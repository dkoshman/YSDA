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
        ratings.groupby(f"{kind}_ids")["rating"].mean().rename(f"mean_{kind}_ratings")
    )
    n_ratings = ratings.groupby(f"{kind}_ids").size().rename(f"{kind}_n_ratings")
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
        user_features["user_ids"] = self.movielens.dataset_to_dense_user_ids(
            dataset_user_ids=user_features["user_ids"].values
        )

        kind = self.FeatureKind.user
        self.update_features(
            kind=kind, cat_features={"user_ids", "gender", "occupation", "zip code"}
        )
        return self.maybe_none_left_merge(
            user_features, super().user_features(), on=kind.merge_on
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
        kind = self.FeatureKind.item
        self.update_features(
            kind=kind, cat_features=set(item_features.columns.drop(["release date"]))
        )
        item_features = pd.merge(item_features, item_ratings_stats, on="item_ids")
        item_features["item_ids"] = self.movielens.dataset_to_dense_item_ids(
            dataset_item_ids=item_features["item_ids"].values
        )
        return self.maybe_none_left_merge(
            item_features, super().item_features(), on=kind.merge_on
        )

    @functools.lru_cache()
    def user_item_features(self):
        user_item_features = self.movielens["u.data"][
            ["user_ids", "item_ids", "timestamp"]
        ]
        kind = self.FeatureKind.user_item
        return self.maybe_none_left_merge(
            user_item_features, super().user_item_features(), on=kind.merge_on
        )


class CatboostMovieLens100kFeatureAggregatorFromArtifacts(
    CatboostAggregatorFromArtifacts, CatboostMovieLens100kFeatureRecommender
):
    def build_pool_kwargs(self, drop_user_ids=None, drop_item_ids=None, **kwargs):
        return super().build_pool_kwargs(
            drop_user_ids=False if drop_user_ids is None else drop_user_ids,
            drop_item_ids=False if drop_item_ids is None else drop_item_ids,
            **kwargs,
        )


# TODO: if catboost ooms on gpu, try batches or look up settings
class CatboostMovieLens25mFeatureRecommender(CatboostRecommenderBase):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens25m(movielens_directory)

    @functools.lru_cache()
    def user_features(self):
        kind = self.FeatureKind.user
        user_features = ratings_stats(self.movielens["ratings"], kind=kind.value)
        return self.maybe_none_left_merge(
            user_features, super().user_features(), on=kind.merge_on
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
        item_features["item_ids"] = self.movielens.dataset_to_dense_item_ids(
            dataset_item_ids=item_features["item_ids"].values
        )

        kind = self.FeatureKind.item
        self.update_features(kind=kind, cat_features=set(item_features.columns))
        return self.maybe_none_left_merge(
            item_features, super().item_features(), on=kind.merge_on
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
        kind = self.FeatureKind.user_item
        user_item_features = pd.merge(
            left=user_item_timestamps, right=user_item_tags, on=kind.merge_on
        )
        self.update_features(kind=kind, text_features={"tag"})
        return self.maybe_none_left_merge(
            user_item_features, super().user_item_features(), on=kind.merge_on
        )
