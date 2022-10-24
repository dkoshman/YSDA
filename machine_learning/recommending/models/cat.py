import warnings

from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

import catboost
import numpy as np
import pandas as pd
import shap
import torch

from my_tools.utils import BuilderMixin, torch_sparse_slice
from ..interface import FitExplicitInterfaceMixin, RecommenderModuleBase
from ..utils import fetch_artifact, load_path_from_artifact

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class CatboostInterface(RecommenderModuleBase, FitExplicitInterfaceMixin, ABC):
    model: catboost.CatBoost

    def __init__(self, explicit=None, **cb_params):
        super().__init__(explicit=explicit)
        self.model = catboost.CatBoostRanker(**cb_params)

    def explicit_dataframe(self, explicit: "spmatrix"):
        if explicit.shape != self.explicit_scipy_coo().shape:
            raise ValueError(
                "Provided explicit feedback shape doesn't match "
                "train shape, user and item ids may be incorrect."
            )
        explicit = explicit.tocoo()
        dataframe = pd.DataFrame(
            dict(
                user_ids=explicit.row,
                item_ids=explicit.col,
                explicit=explicit.data,
            )
        )
        return dataframe

    @staticmethod
    def dense_dataframe(user_ids, item_ids):
        dataframe = pd.DataFrame(
            dict(
                user_ids=np.repeat(user_ids.numpy(), len(item_ids)),
                item_ids=np.tile(item_ids.numpy(), len(user_ids)),
            )
        )
        return dataframe

    @abstractmethod
    def train_dataframe(self, explicit) -> pd.DataFrame:
        ...

    @abstractmethod
    def predict_dataframe(self, user_ids, item_ids) -> pd.DataFrame:
        ...

    @property
    def cat_features(self) -> list[str]:
        """The categorical columns in dataframe being passed to pool."""
        return []

    @property
    def text_features(self) -> list[str]:
        """The text columns in dataframe being passed to pool."""
        return []

    @property
    @abstractmethod
    def pass_user_ids_to_pool(self) -> bool:
        """Whether to pass user ids as feature to pool."""

    def pool(
        self, dataframe: pd.DataFrame, pass_user_ids_to_pool: bool = None
    ) -> catboost.Pool:
        if "user_ids" not in dataframe:
            raise ValueError("Passed dataframe must at least have a user_ids column.")

        if pass_user_ids_to_pool is None:
            pass_user_ids_to_pool = self.pass_user_ids_to_pool

        for column in dataframe:
            if column in self.cat_features + self.text_features:
                dataframe[column] = dataframe[column].astype("string")
            else:
                dataframe[column] = pd.to_numeric(dataframe[column])

        label = None
        if "explicit" in dataframe:
            label = dataframe["explicit"].to_numpy()
            dataframe = dataframe.drop("explicit", axis="columns")
        group_id = dataframe["user_ids"].to_numpy()
        if not pass_user_ids_to_pool:
            dataframe = dataframe.drop("user_ids", axis="columns")
        pool = catboost.Pool(
            data=dataframe,
            cat_features=self.cat_features,
            text_features=self.text_features,
            label=label,
            group_id=group_id,
        )
        return pool

    def fit(self):
        explicit = self.explicit_scipy_coo()
        dataframe = self.train_dataframe(explicit)
        pool = self.pool(dataframe=dataframe)
        self.model.fit(pool, verbose=100)

    def forward(self, user_ids, item_ids):
        dataframe = self.predict_dataframe(user_ids, item_ids)
        pool = self.pool(dataframe=dataframe)
        ratings = self.model.predict(pool)
        ratings = torch.from_numpy(ratings.reshape(len(user_ids), -1))
        return ratings.to(torch.float32)

    def feature_importance(self, explicit: "spmatrix" or None = None) -> pd.Series:
        """
        Returns series with feature names in index and
        their importance in values, sorted by decreasing importance.
        """
        explicit = explicit if explicit is not None else self.explicit_scipy_coo()
        dataframe = self.train_dataframe(explicit)
        pool = self.pool(dataframe)
        feature_importance = self.model.get_feature_importance(pool, prettified=True)
        return feature_importance

    def shap(self, explicit=None) -> tuple[np.ndarray, float, pd.DataFrame]:
        """
        :param explicit: the explicit feedback for which to calculate shap values
        :return shap_values, expected_value, features
        shap_values: matrix of shape [explicit.numel(), n_features] with
        shap values for each feature of each sample
        expected_value: average prediction value on the dataset
        features: dataframe with same shape as shap_values
        with feature values and names
        """
        explainer = shap.TreeExplainer(self.model)
        dataframe = self.train_dataframe(explicit)
        pool = self.pool(dataframe)
        shap_values = explainer.shap_values(pool)
        features = dataframe.drop("explicit", axis="columns")
        if not self.pass_user_ids_to_pool:
            features = features.drop("user_ids", axis="columns")
        return shap_values, explainer.expected_value, features

    def get_extra_state(self):
        self.model.save_model("tmp")
        with open("tmp", "rb") as f:
            binary_bytes = f.read()
        return binary_bytes

    def set_extra_state(self, binary_bytes):
        with open("tmp", "wb") as f:
            f.write(binary_bytes)
        self.model.load_model("tmp")


class CatboostExplicitRecommender(CatboostInterface):
    @property
    def pass_user_ids_to_pool(self):
        return True

    @property
    def cat_features(self):
        return ["user_ids", "item_ids"]

    def train_dataframe(self, explicit):
        return self.explicit_dataframe(explicit)

    def predict_dataframe(self, user_ids, item_ids):
        return self.dense_dataframe(user_ids, item_ids)


class CatboostAggregatorRecommender(CatboostInterface):
    def __init__(
        self,
        fit_recommenders=None,
        explicit=None,
        recommender_names=None,
        train_n_recommendations=10,
        batch_size=100,
        **cb_params,
    ):
        if recommender_names is not None and len(fit_recommenders) != len(
            recommender_names
        ):
            raise ValueError("Models and their names must be of equal length.")
        super().__init__(explicit=explicit, **cb_params)
        self.fit_recommenders = fit_recommenders
        self.recommender_names = recommender_names
        self.train_n_recommendations = train_n_recommendations
        self.batch_size = batch_size

    @staticmethod
    def generate_model_names(models):
        names = []
        counter = Counter()
        for model in models:
            name = model.__class__.__name__
            names.append(f"{name}_{counter[name]}")
            counter.update([name])
        return names

    @property
    def pass_user_ids_to_pool(self):
        return False

    def skeleton_dataframe(self):
        dataframe = self.explicit_dataframe(self.explicit_scipy_coo())
        if self.recommender_names is None:
            self.recommender_names = self.generate_model_names(self.fit_recommenders)
        return pd.DataFrame(columns=list(dataframe.columns) + self.recommender_names)

    def poll_fit_recommenders(self, user_ids, users_explicit=None):
        ratings = []
        for model in self.fit_recommenders:
            with torch.inference_mode():
                if users_explicit is not None:
                    model_ratings = model.online_ratings(users_explicit=users_explicit)
                else:
                    model_ratings = model(
                        user_ids=user_ids, item_ids=torch.arange(self.n_items)
                    )
            ratings.append(model_ratings.to(self.device))
        return ratings

    def aggregate_topk_recommendations(
        self,
        user_ids,
        users_explicit=None,
        filter_already_liked_items=False,
        n_recommendations=None,
    ):
        """
        :return: tensor of shape [topk_item_ids.shape[0], self.n_recommendations]
        with most frequent item_ids among recommendations in no particular order
        """
        topk_item_ids = []
        for recommender in self.fit_recommenders:
            if users_explicit is not None:
                item_ids = recommender.online_recommend(
                    users_explicit=users_explicit, n_recommendations=n_recommendations
                )
            else:
                assert isinstance(n_recommendations, int)
                item_ids = recommender.recommend(
                    user_ids=user_ids,
                    n_recommendations=n_recommendations,
                    filter_already_liked_items=filter_already_liked_items,
                )
            topk_item_ids.append(item_ids.to(self.device))
        per_user_all_recommendations = torch.cat(topk_item_ids, dim=1)

        per_user_topk_item_ids = []
        for user_all_recommendations in per_user_all_recommendations:
            unique_item_ids, counts = user_all_recommendations.unique(
                return_counts=True
            )
            _, indices = torch.topk(counts, k=n_recommendations)
            per_user_topk_item_ids.append(unique_item_ids[indices])
        per_user_topk_item_ids = torch.stack(per_user_topk_item_ids)
        return per_user_topk_item_ids

    def dataframe(self, user_ids, per_user_item_ids, ratings, explicit=None):
        dataframes = []
        for i, (user_id, item_ids) in enumerate(zip(user_ids, per_user_item_ids)):
            user_dataframe = self.skeleton_dataframe()
            user_dataframe["item_ids"] = item_ids.numpy()
            user_dataframe["user_ids"] = user_id.numpy()
            if explicit is not None:
                explicit = explicit.tocsr()
                user_dataframe["explicit"] = (
                    explicit[user_id, item_ids].toarray().squeeze()
                )
            item_ratings = [rating[i, item_ids] for rating in ratings]
            user_dataframe.iloc[:, 3:] = torch.stack(item_ratings, dim=0).T.numpy()
            dataframes.append(user_dataframe)

        dataframe = pd.concat(dataframes)
        if explicit is None:
            dataframe = dataframe.drop("explicit", axis="columns")
        return dataframe

    def dataframe_from_batches(
        self,
        user_ids,
        users_explicit=None,
        item_ids=None,
        explicit=None,
        filter_already_liked_items=None,
        n_recommendations=None,
    ):
        dataframe = self.skeleton_dataframe()
        for batch_indices in torch.arange(len(user_ids)).split(self.batch_size):
            batch_user_ids = user_ids[batch_indices]
            batch_users_explicit = None
            if users_explicit is not None:
                batch_users_explicit = torch_sparse_slice(
                    users_explicit, row_ids=batch_indices
                )
            batch_ratings = self.poll_fit_recommenders(
                user_ids=batch_user_ids,
                users_explicit=batch_users_explicit,
            )
            if item_ids is None:
                per_user_item_ids = self.aggregate_topk_recommendations(
                    user_ids=batch_user_ids,
                    users_explicit=batch_users_explicit,
                    filter_already_liked_items=filter_already_liked_items,
                    n_recommendations=n_recommendations,
                )
            else:
                per_user_item_ids = item_ids.repeat(len(batch_indices), 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = self.dataframe(
                    user_ids=batch_user_ids,
                    per_user_item_ids=per_user_item_ids,
                    ratings=batch_ratings,
                    explicit=explicit,
                )
            dataframe = pd.concat([dataframe, df])

        return dataframe

    def train_dataframe(self, explicit):
        dataframe = self.dataframe_from_batches(
            user_ids=torch.arange(self.n_users),
            n_recommendations=self.train_n_recommendations,
            filter_already_liked_items=False,
            explicit=explicit,
        )
        dataframe = dataframe.drop("item_ids", axis="columns")
        return dataframe

    def predict_dataframe(self, user_ids, item_ids):
        dataframe = self.dataframe_from_batches(user_ids=user_ids, item_ids=item_ids)
        dataframe = dataframe.drop(["item_ids", "explicit"], axis="columns")
        return dataframe

    def recommend(
        self,
        user_ids: torch.IntTensor,
        n_recommendations: int or None = None,
        filter_already_liked_items=True,
    ) -> torch.IntTensor:
        dataframe = self.dataframe_from_batches(
            user_ids=user_ids,
            n_recommendations=n_recommendations or self.n_items,
            filter_already_liked_items=filter_already_liked_items,
        )
        recommendations = self.common_recommend_logic(
            dataframe=dataframe,
            n_users=len(user_ids),
        )
        return recommendations

    def common_recommend_logic(self, dataframe, n_users):
        item_ids = dataframe["item_ids"].to_numpy().astype(np.int64)
        dataframe = dataframe.drop(["item_ids", "explicit"], axis="columns")
        pool = self.pool(dataframe)
        ratings = self.model.predict(pool)
        item_ids = torch.from_numpy(item_ids.reshape(n_users, -1))
        ratings = torch.from_numpy(ratings.reshape(n_users, -1))
        item_ids = torch.take_along_dim(
            item_ids, ratings.argsort(descending=True), dim=1
        )
        return item_ids.to(torch.int64)

    def online_recommend(self, users_explicit, n_recommendations: int or None = None):
        fictive_user_ids = torch.arange(
            self.n_users, self.n_users + users_explicit.shape[0]
        )
        dataframe = self.dataframe_from_batches(
            user_ids=fictive_user_ids,
            users_explicit=users_explicit,
            n_recommendations=n_recommendations or self.n_items,
            filter_already_liked_items=True,
        )
        recommendations = self.common_recommend_logic(
            dataframe=dataframe,
            n_users=len(fictive_user_ids),
        )
        return recommendations


class CatboostAggregatorFromArtifacts(BuilderMixin, CatboostAggregatorRecommender):
    def __init__(
        self, *args, entity=None, project=None, recommender_artifact_names=(), **kwargs
    ):
        fit_recommenders = []
        for artifact_name in recommender_artifact_names:
            artifact = fetch_artifact(
                entity=entity, project=project, artifact_name=artifact_name
            )
            checkpoint_path = load_path_from_artifact(
                artifact, path_inside_artifact="checkpoint"
            )
            checkpoint = torch.load(checkpoint_path)
            recommender = self.build_class(
                class_name=checkpoint["hyper_parameters"]["model_config"]["class_name"]
            )
            state_dict = {
                k.split(".")[1]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.")
            }
            recommender.load_state_dict(state_dict)
            fit_recommenders.append(recommender)
        super().__init__(
            *args,
            fit_recommenders=fit_recommenders,
            recommender_names=recommender_artifact_names,
            **kwargs,
        )

    def get_extra_state(self):
        state = dict(
            super_extra_state=super().get_extra_state(),
            fit_recommenders=[
                dict(class_name=i.__class__.__name__, state_dict=i.state_dict())
                for i in self.fit_recommenders
            ],
            recommender_names=self.recommender_names,
            train_n_recommendations=self.train_n_recommendations,
            batch_size=self.batch_size,
        )
        return state

    def set_extra_state(self, state):
        self.recommender_names = state["recommender_names"]
        self.train_n_recommendations = state["train_n_recommendations"]
        self.batch_size = state["batch_size"]
        self.fit_recommenders = []
        for i in state["fit_recommenders"]:
            recommender = self.build_class(class_name=i["class_name"])
            recommender.load_state_dict(i["state_dict"])
            self.fit_recommenders.append(recommender)
        super().set_extra_state(state["super_extra_state"])
