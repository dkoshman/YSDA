import os
import pandas as pd
import pickle
import string
import torch


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset


def get_data():
    data_dir = "/external2/dkkoshman/Notebooks/ML"

    train = pickle.load(open(os.path.join(data_dir, "train.pickle"), "rb"))
    queries = pickle.load(open(os.path.join(data_dir, "queries.pickle"), "rb"))
    orgs = pickle.load(open(os.path.join(data_dir, "orgs.pickle"), "rb"))

    return train, queries, orgs


class Preprocessor:
    def __init__(self):
        self.sklearn_preprocessor = TfidfVectorizer(
            strip_accents="unicode", lowercase=True
        ).build_preprocessor()

    def __call__(self, text):
        text = self.sklearn_preprocessor(text)
        text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
        return text


def get_numeric_data(train):
    X = train.select_dtypes("number").drop(
        [
            "query_id",
            "org_id",
            "queries_geo_id",
            "orgs_geo_id",
            "queries_window_size_longitude",
            "queries_window_size_latitude",
            "relevance",
        ],
        axis="columns",
    )

    y = train["relevance"]

    return X, y


def split_grouped_data(train, train_size=0.70, random_state=42):
    train_index, test_index = next(
        GroupShuffleSplit(
            n_splits=1, train_size=train_size, random_state=random_state
        ).split(train, groups=train["query_id"])
    )
    return train_index, test_index


def split_data(df, train_size, val_size, test_size):
    if train_size + val_size + test_size != 1:
        raise RuntimeError("Incorrect split sizes")

    train_index, test_index = split_grouped_data(df, train_size=train_size)
    val_index, test_index = split_grouped_data(
        df.loc[test_index], train_size=val_size / (val_size + test_size)
    )
    return df.loc[train_index], df.loc[val_index], df.loc[test_index]


def concatenate_text_features(df: pd.DataFrame) -> pd.Series:
    return df.select_dtypes(object).apply(lambda s: s.str.cat(sep=" "), axis="columns")


def train_vectorizer(queries, orgs, train_relevance, max_features):
    queries_text = concatenate_text_features(queries)
    orgs_text = concatenate_text_features(orgs)

    train_text_data = pd.concat(
        [
            queries_text.loc[train_relevance["query_id"].unique()],
            orgs_text.loc[train_relevance["org_id"].unique()],
        ]
    )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 3),
        preprocessor=Preprocessor(),
        max_features=max_features,
    ).fit(train_text_data)

    queries_text_embedding = pd.DataFrame(
        data=vectorizer.transform(queries_text).toarray(),
        index=queries_text.index,
    )
    orgs_text_embedding = pd.DataFrame(
        data=vectorizer.transform(orgs_text).toarray(),
        index=orgs_text.index,
    )
    return queries_text_embedding, orgs_text_embedding, vectorizer


class RankingDataset(Dataset):
    def __init__(self, queries, orgs, train, dtype=torch.float32):
        def transform(value):
            return torch.tensor(value, dtype=dtype)

        self.text_data = (
            train.groupby("query_id")
            .apply(
                lambda df: {
                    "query": transform(queries.loc[df.artifact_name]),
                    "documents": transform(orgs.loc[df["org_id"]].values),
                    "pairwise_numeric_features": transform(
                        df.drop(
                            ["query_id", "org_id", "relevance"], axis="columns"
                        ).values
                    ),
                    "relevance": transform(df["relevance"].values),
                }
            )
            .to_list()
        )

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, index):
        """
        Returns dict corresponding to single query:
        {
            "query": tensor of shape [text_embedding_dimension],
            "documents": tensor of shape [num_documents, text_embedding_dimension],
            "pairwise_numeric_features: tensor of shape [num_documents, num_features]
            "relevance": tensor of shape [num_documents],
        }
        """
        return self.text_data[index]
