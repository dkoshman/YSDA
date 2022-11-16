import abc
import re
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Literal, Union

import catboost
import einops
import numpy as np
import pandas as pd
import shap
import torch
import wandb

from scipy.sparse import csr_array, coo_array
from tqdm.auto import tqdm

from my_tools.models import WandbLoggerMixin
from my_tools.utils import BuilderMixin, torch_sparse_slice

from ..interface import FitExplicitInterfaceMixin, RecommenderModuleBase
from ..movielens import lit as movielens_lit
from ..utils import filter_warnings, wandb_timeit

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from my_tools.utils import SparseTensor


# TODO: add explanations to imdb, train svd(10, 100, 1000 ?), mf, add explanations to app(maybe),


class CatboostInterface(
    RecommenderModuleBase, FitExplicitInterfaceMixin, WandbLoggerMixin, abc.ABC
):
    def __init__(self, cb_params=None, **kwargs):
        super().__init__(**kwargs)
        self.model = catboost.CatBoostRanker(**(cb_params or {}))
        self._full_train_dataframe_label = None
        self.cat_features: "defaultdict[str, set[str]]" = defaultdict(set)
        self.text_features: "defaultdict[str, set[str]]" = defaultdict(set)

    @abc.abstractmethod
    def train_user_item_dataframe_label(
        self, explicit: "spmatrix"
    ) -> "tuple[pd.DataFrame, np.array]":
        """
        Builds dataframe with at least user_ids and item_ids columns,
        which determine what user-item pairs will be in the train dataset
        induced by explicit matrix. Any extra features to add to this
        dataframe can be specified in *_features methods.
        """

    def full_train_dataframe_label(self):
        """Cached version of previous method built from whole self.explicit matrix."""
        if self._full_train_dataframe_label is None:
            self._full_train_dataframe_label = self.train_user_item_dataframe_label(
                explicit=self.to_scipy_coo(self.explicit)
            )
        return self._full_train_dataframe_label

    @abc.abstractmethod
    def predict_user_item_dataframe(
        self, user_ids, n_recommendations, users_explicit=None
    ) -> pd.DataFrame:
        """
        Builds dataframe with at least user_ids and item_ids columns,
        induced by users_explicit matrix if passed, otherwise by user_ids.

        :param user_ids: tensor with real user_ids (corresponding to self.explicit)
            if no users_explicit is passed, and fictive_user_ids otherwise
        :param n_recommendations: n_recommendations requested in calling recommending method
        :param users_explicit: explicit feedback passed to online_recommend
        """

    def user_features(self) -> pd.DataFrame or None:
        """
        Returns user features dataframe with "user_ids" column, also
        updates self.cat_features, self.text_features with user features names.
        """

    def item_features(self) -> pd.DataFrame or None:
        """
        Returns item features dataframe with "item_ids" column, also
        updates self.cat_features, self.text_features with item feature names.
        """

    def user_item_features(self) -> pd.DataFrame or None:
        """
        Returns user-item pair features dataframe with "user_ids" and "item_ids" columns,
        also updates self.cat_features, self.text_features with user-item feature names.
        """

    @staticmethod
    def maybe_none_merge(left, right, on) -> pd.DataFrame or None:
        if left is None:
            return right
        if right is None:
            return left
        return pd.merge(left=left, right=right, how="left", on=on)

    def merge_features(self, dataframe):
        """Merge all extra features train/predict user-items dataframe."""
        dataframe = self.maybe_none_merge(
            left=dataframe, right=self.user_features(), on="user_ids"
        )
        dataframe = self.maybe_none_merge(
            left=dataframe, right=self.item_features(), on="item_ids"
        )
        dataframe = self.maybe_none_merge(
            left=dataframe, right=self.user_item_features(), on=["user_ids", "item_ids"]
        )
        return dataframe

    @staticmethod
    def pool(
        dataframe: pd.DataFrame,
        group_id,
        label=None,
        cat_features=(),
        text_features=(),
    ) -> catboost.Pool:
        """
        Only minor preprocessing is done in this method to
        ensure that dataframe received is structurally
        the same as one passed into pool. This method is not
        meant to be overridden by child classes.
        """
        dataframe = dataframe.copy()
        for column in dataframe:
            if column in list(cat_features) + list(text_features):
                dataframe[column] = dataframe[column].astype("string")
            else:
                dataframe[column] = pd.to_numeric(dataframe[column])
        with filter_warnings(action="ignore", category=FutureWarning):
            pool = catboost.Pool(
                data=dataframe,
                cat_features=cat_features,
                text_features=text_features,
                label=label,
                group_id=group_id,
            )
        return pool

    def build_pool(self, dataframe, label=None, group_id=None) -> catboost.Pool:
        """Method to override for custom pool creation."""
        return self.pool(
            dataframe=self.merge_features(dataframe),
            group_id=dataframe["user_ids"].values if group_id is None else group_id,
            label=label,
            cat_features=set.union(*self.cat_features.values()),
            text_features=set.union(*self.text_features.values()),
        )

    def fit(self):
        dataframe, label = self.full_train_dataframe_label()
        self.log_user_features_dataframe(dataframe=dataframe, label=label, stage="fit")
        pool = self.build_pool(dataframe=dataframe, label=label)
        self.model.fit(pool, verbose=100)

    def log_user_features_dataframe(self, dataframe, stage, label=None, ratings=None):
        """Logs features dataframe slice corresponding to single random user."""
        user_id = dataframe["user_ids"].sample(1).values[0]
        indices = dataframe.query(f"user_ids == @user_id").index.values
        user_dataframe = dataframe.iloc[indices].copy()
        user_dataframe = self.merge_features(dataframe=user_dataframe)
        if stage == "fit":
            self.log(
                dict(cat_features=self.cat_features, text_features=self.text_features)
            )
        if label is not None:
            with filter_warnings(
                action="ignore", category=pd.errors.SettingWithCopyWarning
            ):
                user_dataframe["explicit"] = label[indices]
        if ratings is not None:
            user_dataframe["ratings"] = ratings[indices]
        self.log(
            {
                f"{stage} user dataframe": wandb.Table(dataframe=user_dataframe),
                f"{stage} dataframe size": float(len(dataframe)),
            }
        )

    def forward(self, user_ids, item_ids):
        raise RuntimeError(
            "Catboost works well when recommending small number of items, "
            "and is not meant to generate ratings for all items."
        )

    def common_recommend(
        self, user_ids, users_explicit: "SparseTensor" = None, n_recommendations=None
    ):
        n_recommendations = n_recommendations or self.n_items
        dataframe = self.predict_user_item_dataframe(
            user_ids=user_ids,
            n_recommendations=n_recommendations,
            users_explicit=users_explicit,
        )
        row_ids = dataframe["user_ids"].values.astype(np.int32)
        col_ids = dataframe["item_ids"].values.astype(np.int32)
        pool = self.build_pool(dataframe=dataframe)
        ratings = self.model.predict(pool)
        self.log_user_features_dataframe(
            dataframe=dataframe, stage="recommend", ratings=ratings
        )

        ratings = coo_array(
            (ratings, (row_ids, col_ids)), shape=[row_ids.max() + 1, self.n_items]
        )
        ratings = ratings.tocsr()[np.unique(row_ids)]
        assert ratings.shape == (len(user_ids), self.n_items)
        ratings = self.to_torch_coo(ratings).to(torch.float32).to_dense()
        ratings[ratings == 0] = -torch.inf

        if users_explicit is None:
            users_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids)
        ratings = self.filter_already_liked_items(
            explicit=users_explicit, ratings=ratings
        )
        recommendations = self.ratings_to_recommendations(ratings, n_recommendations)
        return recommendations

    def recommend(self, user_ids, n_recommendations=None):
        return self.common_recommend(
            user_ids=user_ids, n_recommendations=n_recommendations
        )

    def fictive_user_ids(self, n_users):
        """
        Returns fictive user ids to use as group ids during online recommend,
        which do not intersect with user ids the model saw during fit.
        """
        return torch.arange(self.n_users, self.n_users + n_users)

    def online_recommend(self, users_explicit, n_recommendations=None):
        return self.common_recommend(
            user_ids=self.fictive_user_ids(n_users=users_explicit.shape[0]),
            users_explicit=self.to_torch_coo(users_explicit).to(self.device),
            n_recommendations=n_recommendations,
        )

    def feature_importance(self, pool: catboost.Pool) -> pd.Series:
        """
        Returns series with feature names in index and
        their importance in values, sorted by decreasing importance.
        """
        return self.model.get_feature_importance(pool, prettified=True)

    def shap(self, dataframe, label) -> "tuple[np.ndarray, float, pd.DataFrame]":
        """
        :return shap_values, expected_value, features: everything necessary to draw shap plots
        shap_values – matrix of shape [explicit.numel(), n_features] with
            shap values for each feature of each sample
        expected_value – average prediction value on the dataset
        features – dataframe with same shape as shap_values with
            feature values and names
        """
        explainer = shap.TreeExplainer(self.model)
        pool = self.build_pool(dataframe=dataframe, label=label)
        shap_values = explainer.shap_values(pool)
        features = dataframe[pool.get_feature_names()]
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


class CatboostRecommenderBase(CatboostInterface):
    """This recommender uses only user and item ids as features."""

    def user_features(self):
        self.cat_features["user"] |= {"user_ids"}
        return super().user_features()

    def item_features(self):
        self.cat_features["item"] = {"item_ids"}
        return super().item_features()

    def train_user_item_dataframe_label(self, explicit):
        explicit = self.to_scipy_coo(explicit)
        dataframe = pd.DataFrame(dict(user_ids=explicit.row, item_ids=explicit.col))
        return dataframe, explicit.data

    def predict_user_item_dataframe(
        self, user_ids, n_recommendations, users_explicit=None
    ):
        return pd.DataFrame(
            dict(
                user_ids=np.repeat(user_ids.numpy(), self.n_items),
                item_ids=np.tile(np.arange(self.n_items), len(user_ids)),
            )
        )


class CatboostAggregatorRecommender(CatboostInterface):
    """This recommender uses other pre-fit models' topk recommendations as features."""

    def __init__(
        self,
        fit_recommenders: "list[RecommenderModuleBase]" = None,
        recommender_names: "list[str]" = None,
        train_n_recommendations=10,
        batch_size=100,
        **kwargs,
    ):
        if (
            fit_recommenders is not None
            and recommender_names is not None
            and len(fit_recommenders) != len(recommender_names)
        ):
            raise ValueError("Recommenders and their names must be of equal length.")
        super().__init__(**kwargs)
        self.fit_recommenders = torch.nn.ModuleList(fit_recommenders)
        self.recommender_names = recommender_names or self.generate_model_names(
            self.fit_recommenders
        )
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

    def user_item_features(self):
        self.cat_features["user_item"] |= set(self.recommender_names)
        return super().user_item_features()

    def build_pool(self, dataframe, label=None, group_id=None):
        group_id = dataframe["user_ids"].values
        dataframe = dataframe.drop(["user_ids", "item_ids"], axis="columns")
        return super().build_pool(dataframe=dataframe, group_id=group_id, label=label)

    def ranks_dataframe(self, user_ids, item_ids, ranks):
        """
        :param user_ids: tensor of shape [n_rows]
        :param item_ids: tensor of shape [n_rows]
        :param ranks: tensor of shape [n_rows, len(self.fit_recommenders)]
        :return: column-wise concatenated DataFrame with n_rows
        """
        return pd.DataFrame(
            data=torch.cat(
                [user_ids.reshape(-1, 1), item_ids.reshape(-1, 1), ranks], dim=1
            ),
            columns=["user_ids", "item_ids"] + self.recommender_names,
        )

    def aggregate_topk_recommendations(
        self, user_ids, n_recommendations, users_explicit=None, train_explicit=None
    ):
        """
        Builds dataframe with [user_ids, item_ids] + self.recommender_names columns
        induced by aggregated topk n_recommendations by self.fit_recommenders for
        users_explicit if passed, otherwise for user_ids. Values in self.recommender_names
        columns at index [user_id, item_id] are equal to rank of item_id in recommender's
        recommendations for user_id. If train_explicit is passed, recommenders' opinions
        about user-item pairs present in train_explicit will be added to resulting dataframe,
        if they are not already present in it.
        """
        ranks = 1 + torch.arange(self.n_items).repeat(len(user_ids), 1)
        reciprocal_ranks_sum = torch.zeros(len(user_ids), self.n_items)
        recommenders_ranks = []
        for recommender, name in zip(self.fit_recommenders, self.recommender_names):
            if users_explicit is not None:
                with wandb_timeit(name=f"{name} online_recommend"):
                    item_ids = recommender.online_recommend(users_explicit)
            else:
                with wandb_timeit(name=f"{name} recommend"):
                    item_ids = recommender.recommend(user_ids)
            recommender_ranks = torch.scatter(
                input=ranks, dim=1, index=item_ids, src=ranks
            )
            reciprocal_ranks_sum += 1 / recommender_ranks
            recommenders_ranks.append(recommender_ranks)

        _, topk_item_ids = reciprocal_ranks_sum.topk(k=n_recommendations, dim=1)
        topk_ranks = [
            torch.take_along_dim(input=recommender_ranks, indices=topk_item_ids, dim=1)
            for recommender_ranks in recommenders_ranks
        ]

        repeated_user_ids = einops.repeat(user_ids, f"u -> u {n_recommendations}")
        topk_item_ids = einops.rearrange(topk_item_ids, "u k -> (u k)")
        topk_ranks = einops.rearrange(topk_ranks, "r u k -> (u k) r")
        dataframe = self.ranks_dataframe(
            user_ids=repeated_user_ids, item_ids=topk_item_ids, ranks=topk_ranks
        )

        if train_explicit is not None:
            user_pos, item_ids = self.to_torch_coo(train_explicit).coalesce().indices()
            train_ranks = [rank[user_pos, item_ids] for rank in recommenders_ranks]
            train_ranks = einops.rearrange(train_ranks, "r n -> n r")
            train_dataframe = self.ranks_dataframe(
                user_ids=user_ids[user_pos], item_ids=item_ids, ranks=train_ranks
            )
            dataframe = pd.concat([dataframe, train_dataframe])
            dataframe = dataframe.sort_values("user_ids").drop_duplicates()
        return dataframe

    @staticmethod
    def tensor_str(tensor, max_size=50):
        """Returns string representation of a tensor, clipped to max_size."""
        string = str(tensor)
        if len(string) >= max_size:
            string = string[: max_size // 2] + " ... " + string[-max_size // 2 :]
        string = re.sub(r"\s+", " ", string)
        return string

    @staticmethod
    def torch_slice(sparse_tensor, row_ids):
        if sparse_tensor is not None:
            return torch_sparse_slice(sparse_tensor, row_ids=row_ids)

    def user_batches(
        self, user_ids, users_explicit=None, train_explicit=None, tqdm_stage=None
    ):
        for batch_indices in tqdm(
            iterable=torch.arange(len(user_ids)).split(self.batch_size),
            desc=f"Building {tqdm_stage or ''} dataframe for users: {self.tensor_str(user_ids)}",
            disable=not bool(tqdm_stage),
        ):
            batch_user_ids = user_ids[batch_indices]
            batch_users_explicit = self.torch_slice(users_explicit, batch_indices)
            batch_train_explicit = self.torch_slice(train_explicit, batch_indices)
            yield batch_user_ids, batch_users_explicit, batch_train_explicit

    def recommender_ranks_dataframe(
        self,
        user_ids,
        n_recommendations,
        users_explicit=None,
        train_explicit=None,
        tqdm_stage=None,
    ):
        dataframes = []
        for (
            batch_user_ids,
            batch_users_explicit,
            batch_train_explicit,
        ) in self.user_batches(
            user_ids=user_ids,
            users_explicit=users_explicit,
            train_explicit=train_explicit,
            tqdm_stage=tqdm_stage,
        ):
            batch_dataframe = self.aggregate_topk_recommendations(
                user_ids=batch_user_ids,
                users_explicit=batch_users_explicit,
                train_explicit=batch_train_explicit,
                n_recommendations=n_recommendations,
            )
            dataframes.append(batch_dataframe)
        dataframe = pd.concat(dataframes)
        return dataframe

    def train_user_item_dataframe_label(self, explicit):
        dataframe = self.recommender_ranks_dataframe(
            user_ids=torch.arange(self.n_users),
            n_recommendations=self.train_n_recommendations,
            train_explicit=explicit,
            tqdm_stage="train",
        )
        explicit_as_column = csr_array(explicit)[
            dataframe["user_ids"].values, dataframe["item_ids"].values
        ]
        return dataframe, explicit_as_column

    def predict_user_item_dataframe(
        self, user_ids, n_recommendations, users_explicit=None
    ):
        return self.recommender_ranks_dataframe(
            user_ids=user_ids,
            n_recommendations=n_recommendations,
            users_explicit=users_explicit,
        )

    # def explain_recommendations(
    #     self,
    #     user_id=None,
    #     user_explicit=None,
    #     n_recommendations=10,
    #     log=False,
    #     logging_prefix="",
    # ):
    #     if user_id is not None:
    #         user_id = self.n_users
    #         recommendations = self.recommend(
    #             user_ids=torch.tensor([user_id]), n_recommendations=n_recommendations
    #         )
    #     dataframe = self.dataframe_from_batches(
    #         user_ids=torch.tensor([user_id]),
    #         users_explicit=user_explicit,
    #         n_recommendations=n_recommendations,
    #         filter_already_liked_items=True,
    #     )
    #     train_dataframe = self.train_dataframe(self.explicit)
    #     shap_values, expected_value, features = self.shap(dataframe=dataframe)
    #     figures = []
    #     # figure_context_manager = wandb_plt_figure if log else plt_figure
    #     for item_id in recommendations.squeeze(0).cpu().numpy():
    #         shap_index = dataframe["use"]


class CatboostAggregatorFromArtifacts(CatboostAggregatorRecommender):
    """
    Wrapper for CatboostAggregatorRecommender to fetch fit recommenders from wandb artifacts.
    To use this module, BuilderMixin *_candidates methods need to be overridden first.
    """

    def __init__(
        self,
        *args,
        recommender_artifacts: "list[str]" = (),
        lit_modules: "list[str]" = (),
        **kwargs,
    ):
        fit_recommenders = []
        for artifact_name, lit_name in (
            tqdm_progress_bar := tqdm(zip(recommender_artifacts, lit_modules))
        ):
            tqdm_progress_bar.set_description(f"Fetching artifact {artifact_name}")
            artifact = wandb.use_artifact(artifact_name)
            checkpoint_path = artifact.file()
            lit_module = getattr(movielens_lit, lit_name)
            lit_module = lit_module.load_from_checkpoint(checkpoint_path)
            recommender = lit_module.model
            fit_recommenders.append(recommender)
        super().__init__(
            *args,
            fit_recommenders=fit_recommenders,
            recommender_names=recommender_artifacts,
            **kwargs,
        )
