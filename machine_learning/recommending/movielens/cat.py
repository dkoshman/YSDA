import pandas as pd

from ..models import als, baseline, cat, mf, slim
from ..models.cat import (
    CatboostInterface,
    CatboostAggregatorFromArtifacts,
    CatboostAggregatorRecommender,
)
from .data import MovieLens100k, MovieLens25m


class CatboostMovieLensFeatureRecommender(CatboostInterface):
    def __init__(self, *, movielens_directory=None, **kwargs):
        super().__init__(**kwargs)
        self.movielens_directory = movielens_directory

    @property
    def movielens(self):
        return MovieLens100k(self.movielens_directory)

    @property
    def pass_user_ids_to_pool(self):
        return True

    @property
    def user_features(self):
        user_features = self.movielens["u.user"]
        user_features.index -= 1
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
        return item_features

    @property
    def cat_features(self):
        user_cat_features = ["gender", "occupation", "zip code"]
        item_cat_features = list(self.item_features.columns.drop(["release date"]))
        return ["user_ids", "item_ids"] + user_cat_features + item_cat_features

    def train_dataframe(self, explicit):
        dataframe = self.explicit_dataframe(explicit)
        dataframe = self.merge_features(dataframe)
        return dataframe

    def predict_dataframe(self, user_ids, item_ids):
        dataframe = self.dense_dataframe(user_ids, item_ids)
        dataframe = self.merge_features(dataframe)
        return dataframe

    def get_extra_state(self):
        binary_bytes = super().get_extra_state()
        state = dict(
            binary_bytes=binary_bytes,
            movielens_directory=self.movielens_directory,
        )
        return state

    def set_extra_state(self, state):
        self.movielens_directory = state["movielens_directory"]
        super().set_extra_state(state["binary_bytes"])


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
#     @property
#     def pass_user_ids_to_pool(self):
#         return False
#
#     @property
#     def user_features(self):
#         # user_features = self.movielens["u.user"]
#         # user_features.index -= 1
#         return user_features
#
#     @property
#     def item_features(self):
#         # item_features = self.movielens["u.item"]
#         # item_features.index -= 1
#         # item_features = item_features.drop(
#         #     ["movie title", "video release date", "IMDb URL"], axis="columns"
#         # )
#         # item_features["release date"] = pd.to_datetime(
#         #     item_features["release date"]
#         # ).astype(int)
#         return item_features
#
#     @property
#     def cat_features(self):
#         # user_cat_features = ["gender", "occupation", "zip code"]
#         # item_cat_features = list(self.item_features.columns.drop(["release date"]))
#         # return ["user_ids", "item_ids"] + user_cat_features + item_cat_features
#
#
#     def train_dataframe(self, explicit):
#         # dataframe = self.explicit_dataframe(explicit)
#         # dataframe = self.merge_features(dataframe)
#         return dataframe
#
#     def predict_dataframe(self, user_ids, item_ids):
#         # dataframe = self.dense_dataframe(user_ids, item_ids)
#         # dataframe = self.merge_features(dataframe)
#         return dataframe
#
#     def get_extra_state(self):
#         binary_bytes = super().get_extra_state()
#         state = dict(
#             binary_bytes=binary_bytes,
#             movielens_directory=self.movielens_directory,
#         )
#         return state
#
#     def set_extra_state(self, state):
#         super().set_extra_state(state["binary_bytes"])
#         self.movielens_directory = state["movielens_directory"]


class CatboostMovieLensAggregatorFromArtifacts(CatboostAggregatorFromArtifacts):
    @property
    def module_candidates(self):
        return super().module_candidates + [als, baseline, cat, mf, slim]

    @property
    def class_candidates(self):
        return super().class_candidates + [CatboostMovieLensFeatureRecommender]
