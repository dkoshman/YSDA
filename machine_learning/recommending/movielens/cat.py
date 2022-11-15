import functools
from typing import Literal

import catboost
import pandas as pd

from ..models import als, baseline, cat, mf, slim
from ..models.cat import CatboostAggregatorFromArtifacts, CatboostFeatureRecommenderBase
from .data import MovieLens100k, MovieLens25m


class CatboostMovieLensFeatureRecommenderMixin:
    @staticmethod
    def common_ratings_features(
        features: pd.DataFrame,
        ratings: pd.DataFrame,
        kind: 'Literal["user", "item"]',
    ) -> pd.DataFrame:
        features[f"mean_{kind}_ratings"] = ratings.groupby(f"{kind}_id")[
            "rating"
        ].mean()
        features[f"{kind}_n_ratings"] = ratings.groupby(f"{kind}_id").size()
        return features

    def merge_user_ratings_features(
        self, user_features: pd.DataFrame, ratings: pd.DataFrame
    ) -> pd.DataFrame:
        return self.common_ratings_features(
            features=user_features, ratings=ratings, kind="user"
        )

    def merge_item_ratings_features(
        self, item_features: pd.DataFrame, ratings: pd.DataFrame
    ) -> pd.DataFrame:
        return self.common_ratings_features(
            features=item_features, ratings=ratings, kind="item"
        )

    @staticmethod
    def user_item_timestamps(ratings: pd.DataFrame) -> pd.DataFrame:
        user_item_features = ratings.rename(
            {"user_id": "user_ids", "item_id": "item_ids"}, axis="columns"
        )[["user_ids", "item_ids", "timestamp"]]
        return user_item_features


class CatboostMovieLens100kFeatureRecommender(
    CatboostFeatureRecommenderBase, CatboostMovieLensFeatureRecommenderMixin
):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens100k(movielens_directory)

    @property
    @functools.lru_cache()
    def user_features(self):
        user_features = self.movielens["u.user"]
        user_features = self.merge_user_ratings_features(
            user_features=user_features, ratings=self.movielens["u.data"]
        )
        user_features = user_features.reset_index(names="user_ids")
        user_features["user_ids"] = self.movielens.movielens_user_ids_to_model_user_ids(
            movielens_user_ids=user_features["user_ids"].values
        )
        return user_features

    @property
    @functools.lru_cache()
    def item_features(self):
        item_features = self.movielens["u.item"]

        item_features = item_features.drop(
            ["movie title", "video release date", "IMDb URL"], axis="columns"
        )
        item_features["release date"] = pd.to_datetime(
            item_features["release date"]
        ).astype(int)
        item_features = self.merge_item_ratings_features(
            item_features=item_features, ratings=self.movielens["u.data"]
        )
        item_features = item_features.reset_index(names="item_ids")
        item_features["item_ids"] = self.movielens.movielens_movie_to_model_item_ids(
            movielens_movie_ids=item_features["item_ids"].values
        )
        return item_features

    @property
    @functools.lru_cache()
    def user_item_features(self):
        return self.user_item_timestamps(ratings=self.movielens["u.data"])

    def pool(self, dataframe, **kwargs) -> catboost.Pool:
        user_cat_features = ["user_ids", "gender", "occupation", "zip code"]
        item_cat_features = list(
            self.item_features.columns.drop(
                ["release date", "mean_item_ratings", "item_n_ratings"]
            )
        )
        cat_features = user_cat_features + item_cat_features
        return super().pool(dataframe=dataframe, cat_features=cat_features, **kwargs)


# TODO: if catboost ooms on gpu, try batches or look up settings
class CatboostMovieLens25mFeatureRecommender(
    CatboostFeatureRecommenderBase, CatboostMovieLensFeatureRecommenderMixin
):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens25m(movielens_directory)

    @property
    @functools.lru_cache()
    def item_features(self):
        item_features = self.movielens["movies"].reset_index(names="item_ids")
        item_features["item_ids"] = self.movielens.movielens_movie_to_model_item_ids(
            item_features["item_ids"].values
        )
        genres = item_features["genres"].str.get_dummies()
        item_features = pd.concat(
            [item_features.drop(["title", "genres"], axis="columns"), genres],
            axis="columns",
        )
        return item_features

    @property
    @functools.lru_cache()
    def user_item_features(self):
        user_item_timestamps = self.user_item_timestamps(
            ratings=self.movielens["ratings"]
        )
        user_item_tags = self.movielens["tags"].rename(
            {"userId": "user_ids", "movieId": "item_ids"}, axis="columns"
        )
        user_item_tags = (
            user_item_tags.groupby(["user_ids", "item_ids"])["tag"]
            .apply(", ".join)
            .reset_index()
        )
        user_item_features = pd.merge(
            left=user_item_timestamps, right=user_item_tags, on=["user_ids", "item_ids"]
        )
        return user_item_features

    def pool(self, dataframe, **kwargs) -> catboost.Pool:
        user_cat_features = ["user_ids"]
        item_cat_features = list(
            self.item_features.columns.drop(
                ["release date", "mean_item_ratings", "item_n_ratings"]
            )
        )
        return super().pool(
            dataframe=dataframe,
            cat_features=user_cat_features + item_cat_features,
            text_features=["tag"],
            **kwargs,
        )


# class CatboostMovieLensFeatureAggregatorRecommender(CatboostAggregatorRecommender):
#     def __init__(self, *, movielens_directory=None, **kwargs):
#         super().__init__(**kwargs)
#         self.movielens_directory = movielens_directory
#
#     @property
#     def movielens(self):
#         return MovieLens25m(self.movielens_directory)
#


class CatboostMovieLensAggregatorFromArtifacts(CatboostAggregatorFromArtifacts):
    @property
    def module_candidates(self):
        return super().module_candidates + [als, baseline, cat, mf, slim]

    @property
    def class_candidates(self):
        return super().class_candidates + [CatboostMovieLens100kFeatureRecommender]
