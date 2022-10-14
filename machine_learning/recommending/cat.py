from collections import Counter

import catboost
import pandas as pd
import torch

from .data import MovieLens
from .entrypoints import NonGradientRecommenderMixin, LitRecommenderBase
from .interface import RecommenderModuleInterface
from .movielens import (
    MovieLensNonGradientRecommender,
    MovieLensPMFRecommender,
    MovieLensSLIMRecommender,
    MovieLensRecommender,
)
from .utils import pl_module_from_checkpoint_artifact, torch_sparse_to_scipy_coo


class CatboostMixin:
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


class CatboostRecommenderModule(CatboostMixin, RecommenderModuleInterface):
    def __init__(
        self,
        n_users,
        n_items,
        models,
        model_names=None,
        n_recommendations=10,
        **cb_params,
    ):
        if model_names is not None and len(models) != len(model_names):
            raise ValueError("Models and their names must be of equal length.")
        super().__init__(n_users=n_users, n_items=n_items)
        self.models = models
        self.model_names = (
            self.generate_model_names(models) if model_names is None else model_names
        )
        self.n_recommendations = n_recommendations
        self.model = catboost.CatBoostRanker(**cb_params)

    @staticmethod
    def generate_model_names(models):
        names = []
        counter = Counter()
        for model in models:
            name = model.__class__.__name__
            names.append(f"{name}_{counter[name]}")
            counter.update([name])
        return names

    def skeleton_dataframe(self):
        return pd.DataFrame(
            columns=["user_id", "item_id", "explicit"] + self.model_names
        )

    def topk_data(self, user_ids, users_explicit_feedback=None):
        ratings = []
        topk_item_ids = []
        for model in self.models:
            with torch.inference_mode():
                model_ratings = model(user_ids=user_ids).to_dense()
            ratings.append(model_ratings)
            values, item_ids = torch.topk(model_ratings, k=self.n_recommendations)
            topk_item_ids.append(item_ids)

        per_user_topk_item_ids = []
        for user_id, item_ids in zip(user_ids, torch.cat(topk_item_ids, dim=1)):
            unique_item_ids, counts = item_ids.unique(return_counts=True)
            _, indices = torch.topk(counts, k=self.n_recommendations)
            per_user_topk_item_ids.append(unique_item_ids[indices])
        per_user_topk_item_ids = torch.stack(per_user_topk_item_ids)

        dataframe = self.skeleton_dataframe()
        for i, (user_id, item_ids) in enumerate(zip(user_ids, per_user_topk_item_ids)):
            df = self.skeleton_dataframe()
            df["item_id"] = item_ids.numpy()
            df["user_id"] = user_id.numpy()
            if users_explicit_feedback is not None:
                df["explicit"] = (
                    users_explicit_feedback[i, item_ids].toarray().squeeze()
                )
            item_ratings = [rating[i, item_ids] for rating in ratings]
            df.iloc[:, 3:] = torch.stack(item_ratings, dim=0).T.numpy()
            dataframe = pd.concat([dataframe, df])

        return dict(
            dataframe=dataframe,
            cat_features=["user_id", "item_id"],
            text_features=[],
        )

    def pool(self, dataframe, cat_features=None, text_features=None):
        cat_features = cat_features or []
        text_features = text_features or []
        dataframe = dataframe.astype(
            {c: "string" for c in cat_features + text_features}
        )
        for column in dataframe.columns.drop(cat_features + text_features):
            dataframe[column] = pd.to_numeric(dataframe[column])
        pool = catboost.Pool(
            data=dataframe.drop(["explicit"], axis="columns"),
            cat_features=cat_features,
            text_features=text_features,
            label=None
            if dataframe["explicit"].isna().any()
            else dataframe["explicit"].to_numpy(),
            group_id=dataframe["user_id"].to_numpy(),
        )
        return pool

    def batched_topk_data(self, user_ids=None, explicit_feedback=None, batch_size=100):
        if user_ids is None:
            user_ids = torch.arange(self.n_users)
        dataframe = self.skeleton_dataframe()
        for batch_user_ids in user_ids.split(batch_size):
            batch_explicit = (
                None if explicit_feedback is None else explicit_feedback[user_ids]
            )
            batch = self.topk_data(batch_user_ids, batch_explicit)
            dataframe = pd.concat([dataframe, batch["dataframe"]])
        return dict(
            dataframe=dataframe,
            cat_features=batch["cat_features"],
            text_features=batch["text_features"],
        )

    def fit(self, explicit_feedback):
        data = self.batched_topk_data(explicit_feedback=explicit_feedback)
        pool = self.pool(**data)
        self.model.fit(pool, verbose=50)

    def forward(self, user_ids=None, item_ids=None):
        if not torch.is_tensor(user_ids):
            user_ids = torch.from_numpy(user_ids)
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

    def topk_data(self, user_ids, users_explicit_feedback=None):
        batch = super().topk_data(user_ids, users_explicit_feedback)
        dataframe = batch["dataframe"]
        user_info = self.user_info
        dataframe = pd.merge(
            dataframe,
            user_info["dataframe"],
            how="left",
            left_on="user_id",
            right_index=True,
        )
        item_info = self.item_info
        dataframe = pd.merge(
            dataframe,
            item_info["dataframe"],
            how="left",
            left_on="item_id",
            right_index=True,
        )
        return dict(
            dataframe=dataframe,
            cat_features=batch["cat_features"]
            + user_info["cat_features"]
            + item_info["cat_features"],
            text_features=batch["text_features"]
            + user_info["text_features"]
            + item_info["text_features"],
        )


class CatboostRecommenderModuleFromArtifacts(CatboostMovieLensRecommenderModule):
    def __init__(self, *args, model_artifact_names, **kwargs):
        models = []
        for artifact_name in model_artifact_names:
            models.append(
                pl_module_from_checkpoint_artifact(
                    artifact_name=artifact_name,
                    class_candidates=[
                        MovieLensNonGradientRecommender,
                        MovieLensPMFRecommender,
                        MovieLensSLIMRecommender,
                    ],
                )
            )
        super().__init__(
            *args, models=models, model_names=model_artifact_names, **kwargs
        )


class CatboostRecommender(NonGradientRecommenderMixin, LitRecommenderBase):
    @property
    def class_candidates(self):
        return super().class_candidates + [
            CatboostRecommenderModule,
            CatboostMovieLensRecommenderModule,
            CatboostRecommenderModuleFromArtifacts,
        ]


class MovieLensCatBoostRecommender(CatboostRecommender, MovieLensRecommender):
    pass
