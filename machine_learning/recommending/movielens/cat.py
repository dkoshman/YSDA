import catboost
import numpy as np
import pandas as pd

from ..models import als, baseline, cat, mf, slim
from ..models.cat import (
    CatboostInterface,
    CatboostAggregatorFromArtifacts,
    CatboostAggregatorRecommender,
)
from .data import MovieLens100k, MovieLens25m


class CatboostMovieLens100kFeatureRecommender(CatboostInterface):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens_directory = movielens_directory

    @property
    def movielens(self):
        return MovieLens100k(self.movielens_directory)

    def pool(self, dataframe, **kwargs) -> catboost.Pool:
        user_ids = dataframe["user_ids"].values
        dataframe = dataframe.drop(["user_ids", "item_ids"], axis="columns")
        user_cat_features = ["gender", "occupation", "zip code"]
        item_cat_features = list(self.item_features.columns.drop(["release date"]))
        return super().pool(
            dataframe=dataframe,
            cat_features=user_cat_features + item_cat_features,
            group_id=user_ids,
            **kwargs
        )

    def merge_features(self, dataframe):
        dataframe = pd.merge(
            dataframe,
            self.user_features,
            how="left",
            left_on="user_ids",
            right_index=True,
        )
        dataframe = pd.merge(
            dataframe,
            self.item_features,
            how="left",
            left_on="item_ids",
            right_index=True,
        )
        return dataframe

    @property
    def user_features(self):
        user_features = self.movielens["u.user"]
        user_features.index -= 1
        assert list(user_features) == ["age", "gender", "occupation", "zip code"]
        assert all(user_features.index.values == np.arange(len(user_features)))
        return user_features

    @property
    def item_features(self):
        item_features = self.movielens["u.item"]
        item_features.index -= 1
        item_features = item_features.drop(
            ["movie title", "video release date", "IMDb URL"], axis="columns"
        )
        item_features["release date"] = pd.to_datetime(
            item_features["release date"]
        ).astype(int)
        assert list(item_features) == [
            "release date",
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
        ]
        assert all(item_features.index.values == np.arange(len(item_features)))
        return item_features

    def train_pool_kwargs(self, explicit):
        dataframe, explicit_data = self.dataframe_and_explicit_data(explicit=explicit)
        dataframe = self.merge_features(dataframe=dataframe)
        return dict(dataframe=dataframe, label=explicit_data)

    def predict_pool_kwargs(self, user_ids, n_recommendations, users_explicit=None):
        dataframe = pd.DataFrame(
            dict(
                user_ids=np.repeat(user_ids.numpy(), self.n_items),
                item_ids=np.tile(np.arange(self.n_items), len(user_ids)),
            )
        )
        dataframe = self.merge_features(dataframe=dataframe)
        return dict(dataframe=dataframe)


# # TODO: use as features number of ratings, genres, timestamps
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
