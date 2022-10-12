import pathlib
import pickle

import pandas as pd

from dataset import get_data, RankingDataset, split_data, train_vectorizer


def top_k_dummies(series, top_k=5, **kwargs):
    series = series.astype("category").cat.remove_categories(
        series.value_counts().index[top_k:]
    )
    return pd.get_dummies(series, **kwargs)


def replace_one_hot_columns_with_categorical(X, train, queries, orgs):
    one_hot_column_prefixes = [
        "queries_locale_",
        "queries_geo_id_",
        "orgs_locale_",
        "orgs_org_names_",
        "orgs_rubric_",
        "orgs_region_",
        "orgs_geo_id_",
    ]

    X_with_categorical_columns = (
        X[
            X.columns[
                ~X.columns.str.match(
                    "|".join(f"(^{prefix})" for prefix in one_hot_column_prefixes)
                )
            ]
        ]
        .join(train[["query_id", "org_id"]])
        .join(
            queries[["geo_id", "locale"]].add_prefix("queries_"),
            on="query_id",
        )
        .join(
            orgs[["region_code", "geo_id", "address_locale"]].add_prefix("orgs_"),
            on="org_id",
        )
        .drop(["query_id", "org_id"], axis="columns")
        .astype(
            {
                c: "category"
                for c in [
                    "queries_geo_id",
                    "queries_locale",
                    "orgs_region_code",
                    "orgs_geo_id",
                    "orgs_address_locale",
                ]
            }
        )
    )

    return X_with_categorical_columns


def prepare_extended_data(max_features, one_hot_per_category):
    train, queries, orgs = get_data()
    train_relevance, val_relevance, test_relevance = split_data(
        train,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
    )

    X = train.select_dtypes("number").drop(
        [
            "query_id",
            "org_id",
            "queries_geo_id",
            "orgs_geo_id",
            "queries_window_size_longitude",
            "queries_window_size_latitude",
        ],
        axis="columns",
    )
    X = replace_one_hot_columns_with_categorical(X, train, queries, orgs)
    X = X.fillna(X.select_dtypes("number").mean())
    X = pd.concat([train[["org_id", "query_id"]], X], axis="columns")

    queries, orgs, vectorizer = train_vectorizer(
        queries=queries,
        orgs=orgs,
        train_relevance=train_relevance,
        max_features=max_features,
    )

    for one_hot in one_hot_per_category:
        train = pd.concat(
            [X.select_dtypes("number")]
            + [
                top_k_dummies(column, top_k=one_hot, prefix=column_name)
                for column_name, column in X.select_dtypes("category").iteritems()
            ],
            axis="columns",
        )

        train_dataset = RankingDataset(queries, orgs, train.loc[train_relevance.index])
        val_dataset = RankingDataset(queries, orgs, train.loc[val_relevance.index])
        test_dataset = RankingDataset(queries, orgs, train.loc[test_relevance.index])

        pathlib.Path("data").mkdir(exist_ok=True)
        data_name = f"max_features_{max_features}_one_hot_{one_hot}"
        pickle.dump(
            {
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "test_dataset": test_dataset,
                "vectorizer": vectorizer,
            },
            open(f"data/{data_name}.pickle", "wb"),
        )
        print(f"Data {data_name} ready")


def main():
    max_features = 4096
    one_hot_per_category = [-1, 10, 100, 1000]
    prepare_extended_data(max_features, one_hot_per_category)


if __name__ == "__main__":
    main()
