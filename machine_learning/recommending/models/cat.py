import abc
import os
import pickle
import re
import sys
import warnings
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Tuple

import catboost
import einops
import numpy as np
import pandas as pd
import shap
import torch
import wandb

from scipy.sparse import csr_array, coo_array
from tqdm.auto import tqdm

from my_tools.utils import torch_sparse_slice, to_scipy_coo

from ..interface import (
    ExplanationMixin,
    FitExplicitInterfaceMixin,
    RecommenderModuleBase,
)
from ..movielens import lit as movielens_lit
from ..utils import save_shap_force_plot, wandb_plt_figure, Timer, profile

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from my_tools.utils import SparseTensor


# TODO: train aggregator, upload, find out what is the heaviest function during val, limit val batches or
# online user features
# split into 0.8 0.199 0.001? tags as cat features
# maybe add extra features, more models to aggregator [svd(10, 100, 1000 ?), mf(100, 1000), mf no bias]


class CatboostInterface(
    RecommenderModuleBase, FitExplicitInterfaceMixin, ExplanationMixin, abc.ABC
):
    class FeatureKind(Enum):
        user = "user"
        item = "item"
        user_item = "user_item"

        @property
        def merge_on(self):
            return {
                self.user: "user_ids",
                self.item: "item_ids",
                self.user_item: ["user_ids", "item_ids"],
            }[self]

    def __init__(
        self,
        cb_params=None,
        cb_cpu_fit_n_users_per_batch=10_000,
        cb_gpu_fit_n_users_per_batch=1000,
        cb_fit_verbose=100,
        unknown_token="__NA__",
        feature_perturbation="interventional",
        max_gpu_query_size=1023,
        **kwargs,
    ):
        """There seems to be a bug in catboost code with gpu training that leads to inferior results."""
        super().__init__(**kwargs)
        self._cb_params = cb_params
        self.cb_cpu_fit_n_users_per_batch = cb_cpu_fit_n_users_per_batch
        self.cb_gpu_fit_n_users_per_batch = cb_gpu_fit_n_users_per_batch
        self.cb_fit_verbose = cb_fit_verbose
        self.unknown_token = unknown_token
        self.feature_perturbation = feature_perturbation
        self.max_gpu_query_size = max_gpu_query_size

        self.use_gpu = self.cb_params.get("task_type") == "GPU"
        if self.use_gpu and "devices" in self.cb_params:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, self.cb_params["devices"])
            )
        self.model = catboost.CatBoostRanker(**self.cb_params)
        self.cat_features: "dict[CatboostInterface.FeatureKind: set[str]]" = {
            i: set() for i in self.FeatureKind
        }
        self.text_features: "dict[CatboostInterface.FeatureKind: set[str]]" = {
            i: set() for i in self.FeatureKind
        }
        self._full_train_dataframe_label = None
        self._shap_explainer = None
        warnings.simplefilter(action="ignore", category=FutureWarning)

    @property
    def cb_params(self) -> dict:
        return self._cb_params.copy() if self._cb_params else {}

    def get_extra_state(self):
        self.model.save_model("tmp")
        with open("tmp", "rb") as f:
            catboost_bytes = f.read()
        shap_explainer_bytes = pickle.dumps(self._shap_explainer)
        return dict(
            catboost_bytes=catboost_bytes, shap_explainer_bytes=shap_explainer_bytes
        )

    def set_extra_state(self, bytes_dict):
        with open("tmp", "wb") as f:
            f.write(bytes_dict["catboost_bytes"])
        self.model.load_model("tmp")
        self._shap_explainer = pickle.loads(bytes_dict["shap_explainer_bytes"])

    @property
    def shap_explainer(self):
        if self._shap_explainer is None and self.model.is_fitted():
            self._shap_explainer = shap.TreeExplainer(
                self.model, feature_perturbation=self.feature_perturbation
            )
        return self._shap_explainer

    def update_features(
        self,
        kind: FeatureKind,
        cat_features: set = None,
        text_features: set = None,
    ):
        if cat_features is not None:
            self.cat_features[kind] |= cat_features
        if text_features is not None:
            self.text_features[kind] |= text_features

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

    @staticmethod
    def maybe_none_left_merge(left, right, on) -> pd.DataFrame or None:
        if left is None:
            return right
        if right is None:
            return left
        return pd.merge(left=left, right=right, how="left", on=on)

    def user_features(self) -> pd.DataFrame or None:
        """Constructs user features dataframe with "user_ids" column."""

    def item_features(self) -> pd.DataFrame or None:
        """Constructs item features dataframe with "item_ids" column."""

    def user_item_features(self) -> pd.DataFrame or None:
        """Constructs user-item pair features dataframe with "user_ids" and "item_ids" columns."""

    def features(self, kind: FeatureKind):
        return getattr(self, f"{kind.value}_features")()

    def merge_features(self, dataframe):
        """Merge all extra features train/predict user-items dataframe."""
        for kind in self.FeatureKind:
            dataframe = self.maybe_none_left_merge(
                left=dataframe, right=self.features(kind=kind), on=kind.merge_on
            )
        return dataframe

    @property
    @abc.abstractmethod
    def use_user_ids_as_features(self) -> bool:
        """Whether to keep item ids in dataframe being passed as one of pool_kwargs."""

    @property
    @abc.abstractmethod
    def use_item_ids_as_features(self) -> bool:
        """Whether to keep user ids in dataframe being passed as one of pool_kwargs."""

    @Timer()
    @profile(log_every=10)
    def build_pool_kwargs(self, user_item_dataframe, label=None) -> dict:
        group_id = user_item_dataframe["user_ids"].values
        dataframe = self.merge_features(user_item_dataframe)
        if not self.use_user_ids_as_features:
            dataframe = dataframe.drop("user_ids", axis="columns")
        if not self.use_item_ids_as_features:
            dataframe = dataframe.drop("item_ids", axis="columns")
        return dict(
            dataframe=dataframe,
            group_id=group_id,
            label=label,
            cat_features=list(set().union(*self.cat_features.values())),
            text_features=list(set().union(*self.text_features.values())),
        )

    def postprocess_pool_kwargs(
        self,
        dataframe: pd.DataFrame,
        group_id,
        label=None,
        cat_features=(),
        text_features=(),
    ) -> dict:
        dataframe = dataframe.copy()
        for column in dataframe:
            if column in list(cat_features):
                dataframe[column] = (
                    dataframe[column]
                    .fillna(self.unknown_token)
                    .astype("str")
                    .astype("category")
                )
            elif column in list(text_features):
                dataframe[column] = (
                    dataframe[column].fillna(self.unknown_token).astype("string")
                )
            else:
                dataframe[column] = pd.to_numeric(dataframe[column])
        return dict(
            data=dataframe,
            cat_features=cat_features,
            text_features=text_features,
            label=label,
            group_id=group_id,
        )

    def fit(self):
        dataframe, label = self.full_train_dataframe_label()
        pool_kwargs = self.build_pool_kwargs(user_item_dataframe=dataframe, label=label)
        pool = catboost.Pool(**self.postprocess_pool_kwargs(**pool_kwargs))
        if not self.use_gpu:
            self.model = batch_fit_catboost_ranker(
                cb_params=self.cb_params,
                pool=pool,
                n_groups_per_batch=self.cb_cpu_fit_n_users_per_batch,
            )
        else:
            self.model = fit_catboost_ranker(
                pool=pool,
                cb_params=self.cb_params,
                cpu_n_users_per_batch=self.cb_cpu_fit_n_users_per_batch,
                gpu_n_users_per_batch=self.cb_gpu_fit_n_users_per_batch,
                max_gpu_query_size=self.max_gpu_query_size,
            )

    def forward(self, user_ids, item_ids):
        ...
        raise NotImplementedError(
            "Catboost works well when recommending small number of items, "
            "and is not meant to generate ratings for all items."
        )

    def common_recommend(
        self, user_ids, users_explicit: "SparseTensor" = None, n_recommendations=None
    ):
        n_recommendations = n_recommendations or self.n_items
        user_item_dataframe = self.predict_user_item_dataframe(
            user_ids=user_ids,
            n_recommendations=n_recommendations,
            users_explicit=users_explicit,
        )
        row_ids = user_item_dataframe["user_ids"].values.astype(np.int32)
        col_ids = user_item_dataframe["item_ids"].values.astype(np.int32)
        pool_kwargs = self.build_pool_kwargs(user_item_dataframe=user_item_dataframe)
        pool = catboost.Pool(**self.postprocess_pool_kwargs(**pool_kwargs))
        with Timer(name="catboost.predict"):
            ratings = self.model.predict(pool)

        ratings = coo_array(
            (ratings, (row_ids, col_ids)), shape=[row_ids.max() + 1, self.n_items]
        )
        ratings = ratings.tocsr()[np.unique(row_ids)]
        assert ratings.shape == (len(user_ids), self.n_items)
        ratings = self.to_torch_coo(ratings).to(torch.float32).to_dense()
        ratings[ratings == 0] = -torch.inf

        if users_explicit is None:
            users_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids)
        recommendations = self.ratings_to_filtered_recommendations(
            explicit=users_explicit,
            ratings=ratings,
            n_recommendations=n_recommendations,
        )
        return recommendations

    @Timer()
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

    @Timer()
    def online_recommend(self, users_explicit, n_recommendations=None):
        return self.common_recommend(
            user_ids=self.fictive_user_ids(n_users=users_explicit.shape[0]),
            users_explicit=self.to_torch_coo(users_explicit).to(self.device),
            n_recommendations=n_recommendations,
        )

    @staticmethod
    def user_item_dataframe_clip_to_size(dataframe, size) -> pd.DataFrame:
        if len(dataframe) <= size:
            return dataframe
        return dataframe.reset_index(drop=True).sample(size).sort_values("user_ids")

    def get_feature_importance(self, pool) -> pd.DataFrame:
        less_indices, more_indices = split_by_group_sizes(
            group_ids=pool.get_group_id_hash(),
            group_size_threshold=self.max_gpu_query_size,
        )
        pool = pool.slice(less_indices)
        return self.model.get_feature_importance(pool, prettified=True)

    def shap_kwargs(self, user_item_dataframe, label=None):
        pool_kwargs = self.build_pool_kwargs(
            user_item_dataframe=user_item_dataframe, label=label
        )
        pool = catboost.Pool(**self.postprocess_pool_kwargs(**pool_kwargs))
        shap_values = self.shap_explainer.shap_values(pool)
        return dict(
            base_value=self.shap_explainer.expected_value,
            shap_values=shap_values,
            features=pool_kwargs["dataframe"],
        )

    @Timer()
    def explain_recommendations_for_user(
        self,
        user_id=None,
        user_explicit=None,
        n_recommendations=10,
        log=False,
        logging_prefix="explanation/",
        plot_waterfalls=False,
        figsize=(8, 5),
    ) -> dict:
        return_explanations_dict = {}
        if user_id is not None:
            recommendations = self.recommend(
                user_ids=torch.IntTensor([user_id]), n_recommendations=n_recommendations
            )
            dataframe, _ = self.full_train_dataframe_label()
            dataframe = dataframe.query("user_ids == @user_id")
        else:
            recommendations = self.online_recommend(
                users_explicit=user_explicit, n_recommendations=n_recommendations
            )
            dataframe = self.predict_user_item_dataframe(
                user_ids=self.fictive_user_ids(n_users=1),
                users_explicit=user_explicit,
                n_recommendations=self.n_items,
            )

        recommendations = recommendations[0].cpu().numpy()
        recommendations_index = (
            dataframe.reset_index(drop=True)
            .query("item_ids in @recommendations")
            .index.values
        )
        assert len(recommendations_index) == len(recommendations)
        shap_kwargs = self.shap_kwargs(user_item_dataframe=dataframe)
        shap_kwargs["features"] = shap_kwargs["features"].iloc[recommendations_index]
        shap_kwargs["shap_values"] = shap_kwargs["shap_values"][recommendations_index]
        return_explanations_dict["features"] = shap_kwargs["features"]
        if log:
            title = logging_prefix + "features"
            wandb.log({title: wandb.Table(dataframe=shap_kwargs["features"])})

        shap_plot = shap.force_plot(
            **shap_kwargs,
            out_names="Predicted relevance of items for users",
            text_rotation=0,
        )
        force_plot_textio = save_shap_force_plot(shap_plot=shap_plot)
        return_explanations_dict["force_plot_textio"] = force_plot_textio
        if log:
            title = logging_prefix + "force plot"
            wandb.log({title: wandb.Html(force_plot_textio)})

        with wandb_plt_figure(
            title=logging_prefix + "decision plot", figsize=figsize, log=log
        ) as decision_plot_figure:
            shap.decision_plot(
                **shap_kwargs, legend_labels=[f"item {i}" for i in recommendations]
            )
        return_explanations_dict["decision_plot_figure"] = decision_plot_figure

        if plot_waterfalls:
            return_explanations_dict["waterfall_figures"] = {}
            postprocessed_pool_kwargs = self.postprocess_pool_kwargs(
                **self.build_pool_kwargs(
                    user_item_dataframe=dataframe.iloc[recommendations_index]
                )
            )
            shap_explanation: shap.Explanation = self.shap_explainer(
                X=postprocessed_pool_kwargs["data"]
            )
            for item_id, item_explanation in zip(recommendations, shap_explanation):
                with wandb_plt_figure(
                    title=logging_prefix + f"waterfall plot for item {item_id}",
                    figsize=figsize,
                    log=log,
                ) as figure:
                    shap.waterfall_plot(item_explanation)
                return_explanations_dict["waterfall_figures"][item_id] = figure

        return return_explanations_dict


def batch_fit_catboost_ranker(
    cb_params: dict,
    pool: catboost.Pool,
    n_groups_per_batch: int,
    cb_fit_verbose: int = 100,
    disable_tqdm: bool = False,
) -> catboost.CatBoostRanker:

    catboost_ranker = catboost.CatBoostRanker(**cb_params)
    if not pool.shape[0]:
        return catboost_ranker
    _, group_indices = np.unique(pool.get_group_id_hash(), return_index=True)
    begins = range(0, len(group_indices), n_groups_per_batch)
    ends = list(begins)[1:] + [len(group_indices)]

    if n_groups_per_batch == -1 or len(begins) <= n_groups_per_batch:
        catboost_ranker.fit(pool, verbose=cb_fit_verbose)
        return catboost_ranker

    if pool.get_text_feature_indices():
        raise ValueError(
            "Models summation is not supported for models with text features, "
            "so either pass n_groups_per_batch=-1, or remove text_features."
        )

    batch_rankers = []
    if not disable_tqdm:
        print(
            f"Fitting catboost on "
            f"{'gpu' if cb_params.get('task_type') == 'GPU' else 'cpu'}"
            f" to pool of shape {pool.shape} ",
            file=sys.stderr,
        )
    for begin, end in tqdm(
        iterable=zip(begins, ends), total=len(begins), disable=disable_tqdm
    ):
        batch_indices = np.arange(group_indices[begin], group_indices[end])
        batch_pool = pool.slice(batch_indices)
        if catboost_ranker.is_fitted():
            batch_pool.set_baseline(baseline=catboost_ranker.predict(batch_pool))
        batch_ranker = catboost.CatBoostRanker(**cb_params)
        batch_ranker.fit(batch_pool, verbose=cb_fit_verbose)
        batch_rankers.append(batch_ranker)

    catboost_ranker = catboost.sum_models(batch_rankers)
    return catboost_ranker


def split_by_group_sizes(
    group_ids: np.array, group_size_threshold: int
) -> "Tuple[np.array, np.array]":
    """
    Returns an array with indices of items that have
    group_ids of groups with no more than
    group_size_threshold members, and the rest indices.
    """
    group_ids = pd.DataFrame(dict(group_id=group_ids))
    less_group_ids = (
        group_ids.groupby("group_id")
        .size()
        .rename("size")
        .to_frame()
        .query(f"size <= {group_size_threshold}")
        .index
    )
    less_indices = group_ids.query("group_id in @less_group_ids").index.values
    more_indices = group_ids.index.difference(less_indices).values
    return less_indices, more_indices


def fit_catboost_ranker(
    pool: catboost.Pool,
    cb_params: dict,
    cpu_n_users_per_batch,
    gpu_n_users_per_batch,
    max_gpu_query_size=1023,
) -> catboost.CatBoostRanker:
    """
    While this function is a great candidate for async - gpu and cpu
    fitting are pretty independent, but creating coroutines in any
    function requires for it to be async itself, and any function that calls
    it, and so forth, propagating up to the entry point. This makes async
    code non-modular, fragile and bug prone.
    """
    gpu_indices, cpu_indices = split_by_group_sizes(
        group_ids=pool.get_group_id_hash(), group_size_threshold=max_gpu_query_size
    )
    gpu_ranker = batch_fit_catboost_ranker(
        cb_params=cb_params,
        pool=pool.slice(gpu_indices),
        n_groups_per_batch=gpu_n_users_per_batch,
    )
    cpu_ranker = batch_fit_catboost_ranker(
        cb_params={**cb_params, "task_type": "CPU"},
        pool=pool.slice(cpu_indices),
        n_groups_per_batch=cpu_n_users_per_batch,
    )
    catboost_ranker = catboost.sum_models(
        [gpu_ranker, cpu_ranker], weights=[len(gpu_indices), len(cpu_indices)]
    )
    return catboost_ranker


class CatboostRecommenderBase(CatboostInterface):
    """This recommender uses only user and item ids as features."""

    def user_features(self):
        kind = self.FeatureKind.user
        self.update_features(kind=kind, cat_features={"user_ids"})
        return super().user_features()

    def item_features(self):
        kind = self.FeatureKind.item
        self.update_features(kind=kind, cat_features={"item_ids"})
        return super().item_features()

    @property
    def use_user_ids_as_features(self):
        return True

    @property
    def use_item_ids_as_features(self):
        return True

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
        dataframe_build_batch_size=10_000,
        **kwargs,
    ):
        if (
            fit_recommenders is not None
            and recommender_names is not None
            and len(fit_recommenders) != len(recommender_names)
        ):
            raise ValueError("Recommenders and their names must be of equal length.")
        super().__init__(**kwargs)
        self.fit_recommenders: "torch.nn.ModuleList[RecommenderModuleBase]" = (
            torch.nn.ModuleList(fit_recommenders)
        )
        self.recommender_names = recommender_names or self.generate_model_names(
            self.fit_recommenders
        )
        self.train_n_recommendations = train_n_recommendations
        self.dataframe_build_batch_size = dataframe_build_batch_size

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
        kind = self.FeatureKind.user_item
        self.update_features(kind=kind, cat_features=set(self.recommender_names))
        return super().user_item_features()

    @property
    def use_user_ids_as_features(self):
        return False

    @property
    def use_item_ids_as_features(self):
        return False

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

    @Timer()
    @profile(log_every=10)
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
        extended_reciprocal_ranks = 1 / (
            1 + torch.arange(self.n_items + 1).repeat(len(user_ids), 1)
        )
        extended_reciprocal_ranks[:, -1] = 0
        items_reciprocal_ranks_sum = torch.zeros(len(user_ids), self.n_items)
        recommenders_ranks = []
        for recommender, name in zip(self.fit_recommenders, self.recommender_names):
            recommender.warn_if_cannot_generate_enough_recommendations = False
            if users_explicit is not None:
                item_ids = recommender.online_recommend(users_explicit)
            else:
                item_ids = recommender.recommend(user_ids)

            item_ids = torch.where(
                torch.BoolTensor(item_ids == recommender.invalid_recommendation_mark),
                self.n_items,
                item_ids,
            )
            recommender_reciprocal_ranks = torch.scatter(
                input=torch.zeros_like(extended_reciprocal_ranks),
                dim=1,
                index=item_ids,
                src=extended_reciprocal_ranks,
            )[:, :-1]
            items_reciprocal_ranks_sum += recommender_reciprocal_ranks
            recommender_ranks = torch.where(
                recommender_reciprocal_ranks == 0, 0, 1 / recommender_reciprocal_ranks
            ).int()
            recommenders_ranks.append(recommender_ranks)

        _, topk_item_ids = items_reciprocal_ranks_sum.topk(k=n_recommendations, dim=1)
        topk_ranks = [
            torch.take_along_dim(input=recommender_rank, indices=topk_item_ids, dim=1)
            for recommender_rank in recommenders_ranks
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
            iterable=torch.arange(len(user_ids)).split(
                split_size=self.dataframe_build_batch_size
            ),
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
        if explicit.shape != (
            expected_shape := torch.Size([self.n_users, self.n_items])
        ):
            raise ValueError(
                "Shape of explicit feedback matrix doesn't "
                "match the expected shape of [self.n_users, self.n_items]:"
                f"{explicit.shape} != {expected_shape}"
            )
        dataframe = self.recommender_ranks_dataframe(
            user_ids=torch.arange(self.n_users),
            n_recommendations=self.train_n_recommendations,
            train_explicit=explicit,
            tqdm_stage="train",
        )
        explicit_as_column = csr_array(to_scipy_coo(explicit))[
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
            tqdm_progress_bar := tqdm(
                iterable=zip(recommender_artifacts, lit_modules),
                total=len(recommender_artifacts),
            )
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
