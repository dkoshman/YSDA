from collections import Counter

import catboost
import pandas as pd
import torch

# from ..data import MovieLens
from ..interface import RecommenderModuleInterface

from ..utils import torch_sparse_to_scipy_coo


class CatboostBase(RecommenderModuleInterface):
    def explicit_dataframe(self, explicit_feedback):
        explicit_feedback = explicit_feedback.tocoo()
        return pd.DataFrame(
            dict(
                user_id=explicit_feedback.row,
                item_id=explicit_feedback.col,
                explicit=explicit_feedback.data,
            )
        )

    @property
    def cat_features(self):
        return []

    @property
    def text_features(self):
        return []

    def save(self):
        self.model.save_model("tmp")
        with open("tmp", "rb") as f:
            bytes = f.read()
        return bytes

    def load(self, bytes):
        with open("tmp", "wb") as f:
            f.write(bytes)
        self.model.load_model("tmp")


class CatboostModule(CatboostMixin):
    def __init__(self, n_users, n_items, **cb_params):
        super().__init__(n_users=n_users, n_items=n_items)
        self.model = catboost.CatBoostRanker(**cb_params)

    def fit(self, explicit_feedback):
        super().fit(explicit_feedback)
        explicit_feedback = explicit_feedback.tocoo()
        pool = catboost.Pool(
            pd.DataFrame(
                dict(user_id=explicit_feedback.row, item_id=explicit_feedback.col)
            ),
            label=explicit_feedback.data,
            group_id=explicit_feedback.row,
            cat_features=["user_id", "item_id"],
        )
        self.model.fit(pool, verbose=100)

    def forward(self, user_ids=None, item_ids=None):
        if not torch.is_tensor(user_ids):
            user_ids = torch.from_numpy(user_ids)
        explicit_feedback = torch_sparse_to_scipy_coo(self.explicit_feedback)
        batch = self.topk_data(user_ids.to(torch.int64))
        pool = self.pool(**batch)
        grouped_ratings = self.model.predict(pool)
        grouped_ratings = (
            torch.from_numpy(grouped_ratings)
            .reshape(len(user_ids), -1)
            .to(torch.float32)
        )
        grouped_item_ids = torch.from_numpy(
            batch["dataframe"]["item_id"]
            .values.reshape(len(user_ids), -1)
            .astype("int64")
        )
        ratings = torch.full(
            (len(user_ids), self.n_items),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        ratings = ratings.scatter(1, grouped_item_ids, grouped_ratings)
        return ratings

    def feature_importance(self, user_ids, users_explicit_feedback):
        pool = self.pool(
            **self.topk_data(
                user_ids=user_ids,
                users_explicit_feedback=users_explicit_feedback,
            )
        )
        feature_importance = self.model.get_feature_importance(pool)
        dataframe = (
            pd.Series(feature_importance, self.model.feature_names_)
            .sort_values(ascending=False)
            .to_frame()
            .T
        )
        return dataframe


class CatboostMovieLensRecommenderModule(CatboostRecommenderModule):
    def __init__(self, *args, movielens_directory, **kwargs):
        super().__init__(*args, **kwargs)
        self.movielens = MovieLens(movielens_directory)

    @property
    def user_info(self):
        user_info = self.movielens["u.user"]
        user_info.index -= 1
        return dict(
            dataframe=user_info,
            cat_features=["gender", "occupation", "zip code"],
            text_features=[],
        )

    @property
    def item_info(self):
        item_info = self.movielens["u.item"]
        item_info.index -= 1
        item_info = item_info.drop(["video release date", "IMDb URL"], axis="columns")
        item_info["release date"] = pd.to_datetime(item_info["release date"])
        return dict(
            dataframe=item_info,
            cat_features=list(item_info.columns.drop(["release date", "movie title"])),
            text_features=["movie title"],
        )
