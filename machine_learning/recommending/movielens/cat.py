import functools
from typing import Literal

import pandas as pd

from ..models.cat import CatboostAggregatorFromArtifacts, CatboostRecommenderBase
from .data import MovieLens100k, MovieLens25m


def ratings_stats(
    ratings: pd.DataFrame, kind: 'Literal["user", "item"]'
) -> pd.DataFrame:
    """
    Calculate some statistics from ratings dataframe
    with at least "user_ids", "item_ids" and "rating" in columns
    """
    mean_ratings = (
        ratings.groupby(f"{kind}_id")["rating"].mean().rename(f"mean_{kind}_ratings")
    )
    n_ratings = ratings.groupby(f"{kind}_id").size().rename([f"{kind}_n_ratings"])
    return pd.concat([mean_ratings, n_ratings], axis="columns")


class CatboostMovieLens100kFeatureRecommender(CatboostRecommenderBase):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens100k(movielens_directory)

    @functools.lru_cache()
    def user_features(self):
        user_features = self.movielens["u.user"].reset_index(names="user_ids")

        user_ratings_stats = ratings_stats(self.movielens["u.data"], kind="user")
        user_features = pd.merge(user_features, user_ratings_stats, on="user_ids")

        user_features["user_ids"] = self.movielens.movielens_user_ids_to_model_user_ids(
            movielens_user_ids=user_features["user_ids"].values
        )
        self.cat_features["user"] |= {"user_ids", "gender", "occupation", "zip code"}
        return self.maybe_none_merge(
            user_features, super().user_features(), on="user_ids"
        )

    @functools.lru_cache()
    def item_features(self):
        item_features = (
            self.movielens["u.item"]
            .reset_index(names="item_ids")
            .drop(["movie title", "video release date", "IMDb URL"], axis="columns")
        )
        item_features["release date"] = pd.to_datetime(
            item_features["release date"]
        ).astype(int)

        item_ratings_stats = ratings_stats(self.movielens["u.data"], kind="item")
        item_features = pd.merge(item_features, item_ratings_stats, on="item_ids")

        item_features["item_ids"] = self.movielens.movielens_movie_to_model_item_ids(
            movielens_movie_ids=item_features["item_ids"].values
        )
        self.cat_features["item"] |= set(item_features.columns.drop("release date"))
        return self.maybe_none_merge(
            item_features, super().item_features(), on="item_ids"
        )

    @functools.lru_cache()
    def user_item_features(self):
        return self.maybe_none_merge(
            self.movielens["u.data"][["user_ids", "item_ids", "timestamp"]],
            super().user_item_features(),
            on=["user_ids", "item_ids"],
        )


class CatboostMovieLens100kFeatureAggregatorFromArtifacts(
    CatboostMovieLens100kFeatureRecommender, CatboostAggregatorFromArtifacts
):
    pass


# TODO: if catboost ooms on gpu, try batches or look up settings
class CatboostMovieLens25mFeatureRecommender(CatboostRecommenderBase):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens25m(movielens_directory)

    @functools.lru_cache()
    def user_features(self):
        user_features = ratings_stats(self.movielens["ratings"], kind="user")
        return self.maybe_none_merge(
            user_features, super().user_features(), on="user_ids"
        )

    @functools.lru_cache()
    def item_features(self):
        item_features = self.movielens["movies"].reset_index(names="item_ids")

        item_ratings_stats = ratings_stats(self.movielens["ratings"], kind="item")
        item_features = pd.merge(item_features, item_ratings_stats, on="item_ids")

        genres = item_features["genres"].str.get_dummies()
        item_features = pd.concat(
            [item_features.drop(["title", "genres"], axis="columns"), genres],
            axis="columns",
        )

        item_features["item_ids"] = self.movielens.movielens_movie_to_model_item_ids(
            item_features["item_ids"].values
        )
        self.cat_features["item"] |= set(item_features.columns)
        return self.maybe_none_merge(
            item_features, super().item_features(), on="item_ids"
        )

    @functools.lru_cache()
    def user_item_features(self):
        user_item_timestamps = self.movielens["ratings"][
            ["user_ids", "item_ids", "timestamp"]
        ]
        user_item_tags = (
            self.movielens["tags"]
            .rename({"userId": "user_ids", "movieId": "item_ids"}, axis="columns")
            .groupby(["user_ids", "item_ids"])["tag"]
            .apply(", ".join)
            .reset_index()
        )
        user_item_features = pd.merge(
            left=user_item_timestamps, right=user_item_tags, on=["user_ids", "item_ids"]
        )
        self.text_features["user_item"] |= {"tag"}
        return self.maybe_none_merge(
            user_item_features,
            super().user_item_features(),
            on=["user_ids", "item_ids"],
        )
