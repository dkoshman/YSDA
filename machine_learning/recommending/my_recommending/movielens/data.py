import abc
import argparse
import functools
import os
import string
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import wandb

from scipy.sparse import coo_matrix

from my_recommending.utils import profile, prepare_artifact

if TYPE_CHECKING:
    from my_tools.utils import SparseTensor
    from my_recommending.movielens.lit import LitRecommenderBase


def csv_imdb_ratings_to_dataframe(
    path_to_imdb_ratings_csv: str, ratings_scale_max_to_convert_to: float = 5
) -> pd.DataFrame:
    dataframe = pd.read_csv(path_to_imdb_ratings_csv)
    dataframe = dataframe.rename(lambda c: c.lower().replace(" ", "_"), axis="columns")
    dataframe = dataframe.rename({"const": "imdb_id"}, axis="columns")
    dataframe["imdb_id"] = dataframe["imdb_id"].str.removeprefix("tt").astype(int)
    dataframe["your_rating"] *= ratings_scale_max_to_convert_to / 10
    assert list(dataframe) == [
        "imdb_id",
        "your_rating",
        "date_rated",
        "title",
        "url",
        "title_type",
        "imdb_rating",
        "runtime_(mins)",
        "year",
        "genres",
        "num_votes",
        "release_date",
        "directors",
    ]
    return dataframe


class RecommendingDatasetInterface(abc.ABC):
    @property
    @abc.abstractmethod
    @functools.lru_cache
    def unique_dataset_user_ids(self) -> np.array:
        """Cached unique sorted user ids that the model can encounter during training."""

    @property
    @abc.abstractmethod
    @functools.lru_cache
    def unique_dataset_item_ids(self) -> np.array:
        """Cached unique sorted item ids that the model can encounter during training."""

    @property
    @functools.lru_cache()
    def shape(self) -> "tuple[int, int]":
        n_users = len(self.unique_dataset_user_ids)
        n_items = len(self.unique_dataset_item_ids)
        return n_users, n_items

    def dense_to_dataset_user_ids(self, dense_user_ids: np.array) -> np.array:
        return self.unique_dataset_user_ids[dense_user_ids]

    def dataset_to_dense_user_ids(self, dataset_user_ids: np.array) -> np.array:
        dataset_to_dense = pd.Series(
            index=self.unique_dataset_user_ids,
            data=np.arange(len(self.unique_dataset_user_ids)),
        )
        return dataset_to_dense.loc[dataset_user_ids].values

    def dense_to_dataset_item_ids(self, dense_item_ids: np.array) -> np.array:
        return self.unique_dataset_item_ids[dense_item_ids]

    def dataset_to_dense_item_ids(self, dataset_item_ids: np.array) -> np.array:
        dataset_to_model = pd.Series(
            index=self.unique_dataset_item_ids,
            data=np.arange(len(self.unique_dataset_item_ids)),
        )
        return dataset_to_model.loc[dataset_item_ids].values


class MovieLensInterface(RecommendingDatasetInterface, abc.ABC):
    def __init__(self, directory):
        self.directory = directory

    def abs_path(self, filename):
        return os.path.join(os.getcwd(), self.directory, filename)

    @property
    def ratings_columns_to_dtypes(self) -> OrderedDict:
        return OrderedDict(
            {
                "user_id": np.int32,
                "item_id": np.int32,
                "rating": np.float32,
                "timestamp": np.int32,
            }
        )

    @abc.abstractmethod
    @functools.lru_cache()
    def __getitem__(self, filename):
        return self.read(filename=filename)

    def read(self, filename, **kwargs):
        path = os.path.join(self.directory, filename)
        dataframe = pd.read_csv(path, **kwargs)
        return dataframe.squeeze()

    def explicit_feedback_scipy_csr(self, name):
        return self.ratings_dataframe_to_scipy_csr(self[name])

    def ratings_dataframe_to_scipy_csr(self, ratings_dataframe):
        movielens_ratings = ratings_dataframe["rating"].to_numpy()

        dataset_user_ids = ratings_dataframe["user_id"].to_numpy()
        user_ids = self.dataset_to_dense_user_ids(dataset_user_ids)

        dataset_item_ids = ratings_dataframe["item_id"].to_numpy()
        item_ids = self.dataset_to_dense_item_ids(dataset_item_ids)

        data = movielens_ratings
        row_ids = user_ids
        col_ids = item_ids
        explicit_feedback = coo_matrix((data, (row_ids, col_ids)), shape=self.shape)
        return explicit_feedback.tocsr()

    @abc.abstractmethod
    def items_description(self, dense_item_ids: np.array) -> pd.DataFrame:
        """Generates dataframe of items description with an item per row in free form."""

    @staticmethod
    def construct_1d_explicit(
        item_ids: np.array, ratings: np.array, n_items
    ) -> "SparseTensor":
        return torch.sparse_coo_tensor(
            indices=np.stack([np.zeros_like(item_ids), item_ids]),
            values=ratings,
            size=[1, n_items],
        )

    @abc.abstractmethod
    def imdb_ratings_dataframe_to_explicit(
        self, imdb_ratings: pd.DataFrame
    ) -> "SparseTensor":
        """
        Builds explicit feedback matrix from imdb_ratings DataFrame
        returned by the csv_imdb_ratings_to_dataframe function.
        """


class MovieLens100k(MovieLensInterface):
    def __getitem__(self, filename):
        if filename == "u.info":
            return self.read(
                filename=filename,
                names=["quantity", "index"],
                index_col="index",
                sep=" ",
            )
        elif filename == "u.genre":
            return self.read(filename=filename, names=["name", "id"], index_col="id")
        elif filename == "u.occupation":
            return self.read(filename=filename, names=["occupation"])
        elif filename == "u.user":
            return self.read(
                filename=filename,
                names=["user_id", "age", "gender", "occupation", "zip code"],
                index_col="user_id",
            )
        elif filename == "u.item":
            return self.read(
                filename=filename,
                names=[
                    "movie id",
                    "movie title",
                    "release date",
                    "video release date",
                    "IMDb URL",
                    "unknown",
                    "Action",
                    "Adventure",
                    "Animation",
                    "Children's",
                    "Comedy",
                    "Crime",
                    "Documentary",
                    "Drama",
                    "Fantasy",
                    "Film-Noir",
                    "Horror",
                    "Musical",
                    "Mystery",
                    "Romance",
                    "Sci-Fi",
                    "Thriller",
                    "War",
                    "Western",
                ],
                index_col="movie id",
                encoding_errors="backslashreplace",
            )
        if filename in ["u.data"] + [
            i + j
            for i in ["u1", "u2", "u3", "u4", "u5", "ua", "ub"]
            for j in [".base", ".test"]
        ]:
            return self.read(
                filename=filename,
                names=list(self.ratings_columns_to_dtypes),
                sep="\t",
                dtype=self.ratings_columns_to_dtypes,
            )
        raise FileNotFoundError(f"File {filename} not found.")

    def read(self, filename, sep="|", header=None, **kwargs):
        return super().read(filename=filename, sep=sep, header=header, **kwargs)

    @property
    @functools.lru_cache
    def unique_dataset_user_ids(self) -> np.array:
        return np.unique(self["u.user"].index)

    @property
    @functools.lru_cache
    def unique_dataset_item_ids(self):
        return np.unique(self["u.item"].index)

    def items_description(self, dense_item_ids: np.array):
        assert dense_item_ids.ndim == 1
        dataset_item_ids = self.dense_to_dataset_item_ids(dense_item_ids)
        description = (
            self["u.item"]
            .loc[dataset_item_ids]
            .reset_index(names="movielens_100k_dataset_ids")
        )
        description["dense_item_ids"] = dense_item_ids
        return description

    @staticmethod
    def normalize_titles(titles):
        """
        Normalizes unicode string, removes punctuation, article 'the',
        deduplicates consecutive spaces and strips fringing spaces.
        """
        titles = (
            titles.str.lower()
            .str.normalize("NFC")
            .str.replace(f"[{string.punctuation}]", "", regex=True)
            .str.replace("the", "")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return titles

    def imdb_ratings_dataframe_to_explicit(self, imdb_ratings: pd.DataFrame):
        imdb_ratings["title"] = self.normalize_titles(imdb_ratings["title"])

        movielens_titles = self["u.item"]["movie title"]
        movielens_titles = self.normalize_titles(movielens_titles.str.split("(").str[0])

        movielens_movie_ids = []
        ratings = []

        for imdb_title, rating in zip(
            imdb_ratings["title"], imdb_ratings["your_rating"]
        ):
            for movielens_movie_id, movielens_title in movielens_titles.items():
                if imdb_title == movielens_title:
                    movielens_movie_ids.append(movielens_movie_id)
                    ratings.append(rating)

        ratings = np.array(ratings)
        movielens_movie_ids = np.array(movielens_movie_ids)
        item_ids = self.dataset_to_dense_item_ids(movielens_movie_ids)

        return self.construct_1d_explicit(
            item_ids=item_ids, ratings=ratings, n_items=self.shape[1]
        )


class MovieLens25m(MovieLensInterface):
    @profile()
    def prepare_splits(
        self, train_time_quantile: float, test_user_fraction: float, seed=42
    ):
        time_split_ante, time_split_post = dataframe_quantiles_deterministic_split(
            quantiles=[train_time_quantile, 1 - train_time_quantile],
            dataframe=self["ratings"],
            column_to_split_by="timestamp",
        )
        unique_user_ids = time_split_post["user_id"].unique()
        rng = np.random.default_rng(seed)
        test_user_ids = rng.choice(
            unique_user_ids,
            size=int(len(unique_user_ids) * test_user_fraction),
            replace=False,
        )
        val_split = time_split_post.query(
            "user_id not in @test_user_ids",
            local_dict=dict(test_user_ids=test_user_ids),
        )
        test_split = time_split_post.query(
            "user_id in @test_user_ids",
            local_dict=dict(test_user_ids=test_user_ids),
        )
        splits = dict(train=time_split_ante, val=val_split, test=test_split)
        for kind, split in splits.items():
            filename = f"ratings_{kind}.csv"
            split.to_csv(path_or_buf=self.abs_path(filename=filename), index=False)

    @functools.lru_cache(maxsize=20)
    def __getitem__(self, filename):
        if not filename.endswith(".csv"):
            filename += ".csv"

        if filename == "ratings.csv":
            return self.read(
                filename=filename,
                names=list(self.ratings_columns_to_dtypes),
                header=0,
                dtype=self.ratings_columns_to_dtypes,
            )
        elif filename == "links.csv":
            nullable_int64 = pd.Int64Dtype()
            return self.read(
                filename=filename,
                dtype={
                    "movieId": np.int32,
                    "imdbId": np.int32,
                    "tmdbId": nullable_int64,
                },
            ).rename({"movieId": "movielensId"}, axis="columns")
        elif filename == "movies.csv":
            return self.read(
                filename=filename,
                index_col="movieId",
                dtype={"movieId": np.int32, "title": str, "genres": str},
            ).rename_axis("item_id")
        elif filename == "tags.csv":
            dtypes = self.ratings_columns_to_dtypes
            dtypes.pop("rating")
            return (
                self.read(filename=filename, dtype={"tag": str})
                .rename({"userId": "user_id", "movieId": "item_id"}, axis="columns")
                .astype(dtypes)
                .dropna()
            )
        else:
            return self.read(filename=filename)

    @property
    @functools.lru_cache
    def unique_dataset_user_ids(self) -> np.array:
        return np.unique(self["ratings"]["user_id"])

    @property
    @functools.lru_cache
    def unique_dataset_item_ids(self):
        return np.unique(self["ratings"]["item_id"])

    @property
    @functools.lru_cache()
    def unique_imdb_ids(self):
        return np.unique(self["links"]["imdbId"])

    def dataset_to_imdb_movie_ids(self, dataset_item_ids: np.array):
        movielens_to_imdb = self["links"].set_index("movielensId")["imdbId"]
        return movielens_to_imdb.loc[dataset_item_ids].values

    def imdb_to_dataset_item_ids(self, imdb_movie_ids: np.array):
        imdb_to_movielens = self["links"].set_index("imdbId")["movielensId"]
        return imdb_to_movielens.loc[imdb_movie_ids].values

    def items_description(self, dense_item_ids: np.array):
        assert dense_item_ids.ndim == 1
        dataset_item_ids = self.dense_to_dataset_item_ids(dense_item_ids)
        description = (
            self["movies"]
            .loc[dataset_item_ids]
            .reset_index(names="movielens_movie_ids")
        )
        description["dense_item_ids"] = dense_item_ids
        description = pd.merge(
            description,
            self["links"],
            how="left",
            left_on="movielens_movie_ids",
            right_on="movielensId",
        )
        return description

    def imdb_ratings_dataframe_to_explicit(self, imdb_ratings: pd.DataFrame):
        imdb_ids = imdb_ratings["imdb_id"].values
        imdb_ids = np.intersect1d(self.unique_imdb_ids, imdb_ids)
        dataset_ids = self.imdb_to_dataset_item_ids(imdb_ids)
        item_ids = self.dataset_to_dense_item_ids(dataset_ids)
        ratings = imdb_ratings.set_index("imdb_id").loc[imdb_ids]["your_rating"].values
        return self.construct_1d_explicit(
            item_ids=item_ids, ratings=ratings, n_items=self.shape[1]
        )


class MovieLensMixin:
    @property
    def movielens(self: "LitRecommenderBase") -> MovieLensInterface:
        return self.build_class(**self.hparams["datamodule"]["movielens"])

    @property
    def class_candidates(self):
        return super().class_candidates + [MovieLens100k, MovieLens25m]

    def common_explicit(self: "LitRecommenderBase", filename):
        if (movielens_inner_name := self.hparams["datamodule"][filename]) is None:
            return
        try:
            return self.movielens.explicit_feedback_scipy_csr(name=movielens_inner_name)
        except FileNotFoundError:
            return

    def train_explicit(self):
        return self.common_explicit(filename="train_explicit_file")

    def val_explicit(self):
        return self.common_explicit(filename="val_explicit_file")

    def test_explicit(self):
        return self.common_explicit(filename="test_explicit_file")


def dataframe_quantiles_deterministic_split(
    dataframe: pd.DataFrame, quantiles: "list[float]", column_to_split_by: str
) -> "list[pd.DataFrame]":
    if sum(quantiles) != 1:
        raise ValueError(f"Quantiles must sum up to one, received {quantiles}.")
    if not pd.api.types.is_numeric_dtype(dataframe[column_to_split_by].dtype):
        raise ValueError(
            f"Splitting column's dtype isn't numeric: "
            f"{column_to_split_by} -> {dataframe[column_to_split_by].dtype} "
        )
    if any(dataframe[column_to_split_by].isna()):
        raise ValueError(f"Splitting column {column_to_split_by} contains nans.")

    quantile_thresholds = np.cumsum(quantiles)
    thresholds = dataframe[column_to_split_by].quantile(
        quantile_thresholds, interpolation="lower"
    )
    thresholds = [-np.inf] + list(thresholds)
    splits = []
    for begin, end in zip(thresholds, thresholds[1:]):
        splits.append(dataframe.query(f"{begin} < {column_to_split_by} <= {end}"))

    assert sum(map(len, splits)) == len(dataframe), "Split lengths don't sum up."
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Prepare movielens25m artifact and splits."
    )
    parser.add_argument("directory", type=str, help="movielens25m directory")
    parser.add_argument("train_time_quantile", type=float)
    parser.add_argument("test_user_fraction", type=float)
    parser.add_argument(
        "update_with_local_directory", type=bool, nargs="?", default=False
    )
    args = parser.parse_args()

    movielens25m = MovieLens25m(directory=args.directory)
    wandb.init(project="Recommending", job_type="data")
    artifact = prepare_artifact(
        full_artifact_name="dkoshman/Recommending/movielens25m:latest",
        directory=args.directory,
        artifact_type="data",
        update_with_local_directory=args.update_with_local_directory,
    )
    artifact.wait()
    movielens25m.prepare_splits(
        train_time_quantile=args.train_time_quantile,
        test_user_fraction=args.test_user_fraction,
    )
    artifact.metadata["train_time_quantile"] = args.train_time_quantile
    artifact.metadata["test_user_fraction"] = args.test_user_fraction
    artifact.save()


if __name__ == "__main__":
    main()
