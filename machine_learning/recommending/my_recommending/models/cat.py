import abc
import functools
import os
import pickle
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Tuple, List, Set

import catboost
import einops
import numpy as np
import pandas as pd
import shap
import torch
import wandb

from scipy.sparse import coo_array
from tqdm.auto import tqdm

from my_tools.utils import torch_sparse_slice

from ..interface import (
    ExplanationMixin,
    FitExplicitInterfaceMixin,
    RecommenderModuleBase,
)
from ..movielens import lit as movielens_lit
from ..utils import (
    save_shap_force_plot,
    wandb_plt_figure,
    Timer,
    profile,
    clipped_tensor_string_representation,
)

if TYPE_CHECKING:
    from my_tools.utils import SparseTensor


def batch_fit_catboost_ranker(
    cb_params: dict,
    pool: catboost.Pool,
    n_groups_per_batch: int,
    cb_fit_verbose: int = 100,
    disable_tqdm: bool = False,
) -> catboost.CatBoostRanker:
    """Fits catboost ranker with cb_params to pool in batches with n non-homogenous groups per batch."""
    catboost_ranker = catboost.CatBoostRanker(**cb_params)
    if pool.shape[0] == 0:
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
            "so either pass n_groups_per_batch=-1, or remove text features."
        )

    batch_rankers = []
    if not disable_tqdm:
        print(
            f"Fitting catboost on "
            f"{'gpu' if cb_params.get('task_type') == 'GPU' else 'cpu'}"
            f" to pool of shape {pool.shape}.",
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
    less_indices = group_ids.query(
        "group_id in @less_group_ids", local_dict=dict(less_group_ids=less_group_ids)
    ).index.values
    more_indices = group_ids.index.difference(less_indices).values
    return less_indices, more_indices


def fit_catboost_ranker(
    pool: catboost.Pool,
    cb_params: dict,
    cpu_n_users_per_batch=10_000,
    gpu_n_users_per_batch=1000,
    max_gpu_query_size=1023,
) -> catboost.CatBoostRanker:
    """
    There seems to be a bug in catboost code with gpu training that leads to inferior results.

    While this function is a great candidate for async - gpu and cpu
    fitting are pretty independent, but creating coroutines in any
    function requires for it to be async itself, and any function that calls
    it, and so forth, propagating up to the entry point. This makes async
    code non-modular, fragile and bug prone.
    """
    if cb_params.get("task_type") != "GPU":
        return batch_fit_catboost_ranker(
            cb_params=cb_params,
            pool=pool,
            n_groups_per_batch=cpu_n_users_per_batch,
        )

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


class CatboostInterface(
    RecommenderModuleBase, FitExplicitInterfaceMixin, ExplanationMixin, abc.ABC
):
    class FeatureKind(Enum):
        user = "user"
        item = "item"
        user_item = "user_item"

        @property
        def index_columns(self):
            return {
                self.user: ["user_id"],
                self.item: ["item_id"],
                self.user_item: ["user_id", "item_id"],
            }[self]

    @dataclass
    class FeatureReturnValue:
        dataframe: pd.DataFrame or None = None
        cat_features: "Set[str]" = field(default_factory=set)
        text_features: "Set[str]" = field(default_factory=set)

    def __init__(
        self,
        cb_params=None,
        cb_cpu_fit_n_users_per_batch=10_000,
        cb_gpu_fit_n_users_per_batch=1000,
        cb_fit_verbose=100,
        use_user_ids_as_features=True,
        use_item_ids_as_features=True,
        use_text_features=True,
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
        self.use_user_ids_as_features = use_user_ids_as_features
        self.use_item_ids_as_features = use_item_ids_as_features
        self.use_text_features = use_text_features
        self.unknown_token = unknown_token
        self.feature_perturbation = feature_perturbation
        self.max_gpu_query_size = max_gpu_query_size

        if self.cb_params.get("task_type") == "GPU" and "devices" in self.cb_params:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, self.cb_params["devices"])
            )
        self.model = catboost.CatBoostRanker(**self.cb_params)
        self.cat_features = {i: set() for i in self.FeatureKind}
        self.text_features = {i: set() for i in self.FeatureKind}
        self._train_user_item_rating_dataframe = None
        self._shap_explainer = None
        warnings.simplefilter(action="ignore", category=FutureWarning)

    @property
    def cb_params(self) -> dict:
        """Convenience property to avoid unwanted inplace changes to self._cb_params."""
        return self._cb_params.copy() if self._cb_params else {}

    @property
    def shap_explainer(self):
        if self._shap_explainer is None:
            if not self.model.is_fitted():
                raise ValueError(
                    "To initialize shap.TreeExplainer, model must first be fitted."
                )
            self._shap_explainer = shap.TreeExplainer(
                self.model, feature_perturbation=self.feature_perturbation
            )
        return self._shap_explainer

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

    def user_features(self) -> FeatureReturnValue:
        """
        Returns FeatureReturnValue with these kwargs:
            dataframe: dataset specific user features with at least a "user_id" column
            cat_features: set of column names with categorical features
            text_features: set of column names with text features
        """
        return self.FeatureReturnValue()

    def item_features(self) -> FeatureReturnValue:
        """Returns dict analogous to user_features, but with "item_id" column."""
        return self.FeatureReturnValue()

    def user_item_features(self) -> FeatureReturnValue:
        """Returns dict analogous to user_features, but with "user_id" and "item_id" columns."""
        return self.FeatureReturnValue()

    @staticmethod
    def set_index(dataframe: pd.DataFrame, kind: FeatureKind):
        if dataframe.index.names != kind.index_columns:
            dataframe = dataframe.set_index(kind.index_columns, verify_integrity=True)
        if not dataframe.index.is_unique:
            raise ValueError(f"DataFrame index is not unique:\n{dataframe.head()}")
        return dataframe.sort_index()

    @functools.lru_cache()
    def cached_indexed_features_dataframe(self, kind: FeatureKind) -> pd.DataFrame:
        features: CatboostInterface.FeatureReturnValue = getattr(
            self, f"{kind.value}_features"
        )()
        self.cat_features[kind] |= features.cat_features
        self.text_features[kind] |= features.text_features
        if features.dataframe is not None:
            features_dataframe = self.set_index(features.dataframe, kind=kind)
            return features_dataframe

    @Timer()
    def merge_features(self, user_item_dataframe):
        """Merges dataset specific features defined in *_features methods into the given user-item dataframe."""
        for kind in self.FeatureKind:
            features_dataframe = self.cached_indexed_features_dataframe(kind=kind)
            if features_dataframe is not None:
                user_item_dataframe = user_item_dataframe.join(features_dataframe)
        return user_item_dataframe

    def rating_stats(self, ratings_dataframe, kind: FeatureKind) -> pd.DataFrame:
        """Returns dataframe with kind.index_columns index or columns and some ratings statistics in columns."""
        return pd.DataFrame(
            columns=CatboostInterface.FeatureKind.user_item.index_columns
        )

    @functools.lru_cache()
    def cached_rating_stats(self, kind: FeatureKind) -> pd.DataFrame:
        ratings_dataframe = self.ratings_user_item_dataframe(explicit=self.explicit)
        rating_stats = self.rating_stats(ratings_dataframe=ratings_dataframe, kind=kind)
        rating_stats = self.set_index(rating_stats, kind=kind)
        return rating_stats

    def merge_rating_stats(
        self, dataframe, users_explicit=None, user_ids: np.array = None
    ) -> pd.DataFrame:
        """
        Merges user and item stats from rating_stats method into the
        received user-item dataframe, using cached stats from train explicit matrix,
        and recalculating only if the online user stats are needed.
        """
        dataframe = self.set_index(dataframe, kind=self.FeatureKind.user_item)
        user_ratings_stats = self.cached_rating_stats(kind=self.FeatureKind.user)
        item_ratings_stats = self.cached_rating_stats(kind=self.FeatureKind.item)
        if user_ids is not None and self.are_user_ids_fictive(user_ids):
            ratings_dataframe = self.ratings_user_item_dataframe(
                explicit=users_explicit, user_ids=user_ids
            )
            user_ratings_stats = self.rating_stats(
                ratings_dataframe=ratings_dataframe, kind=self.FeatureKind.user
            )
        dataframe = dataframe.join(user_ratings_stats).join(item_ratings_stats)
        return dataframe

    def ratings_user_item_dataframe(
        self, explicit: "SparseTensor", user_ids: np.array = None
    ) -> pd.DataFrame:
        """
        Builds a dataframe with user_id, item_id multiindex and rating
        column from user-item pairs present in the explicit feedback matrix.
        """
        assert explicit.shape[1] == self.n_items, "Invalid explicit matrix shape."
        explicit = self.to_scipy_coo(explicit)
        user_ids = np.arange(self.n_users) if user_ids is None else user_ids
        dataframe = pd.DataFrame(
            dict(
                user_id=user_ids[explicit.row],
                item_id=explicit.col,
                rating=explicit.data,
            )
        )
        dataframe = self.set_index(dataframe=dataframe, kind=self.FeatureKind.user_item)
        return dataframe

    def extract_pool_kwargs(self, dataframe) -> dict:
        """
        Builds dict of catboost Pool kwargs from dataframe with at least
        user_id, item_id, *cat_features, *text_features,
        some number of numeric features columns,
        and rating column if it is a labeled dataset.
        """
        dataframe = dataframe.reset_index()
        group_id = dataframe["user_id"].values
        cat_features = set().union(*self.cat_features.values())
        text_features = set().union(*self.text_features.values())
        if "rating" in dataframe:
            label = dataframe["rating"].values
            dataframe = dataframe.drop("rating", axis="columns")
        else:
            label = None
        if self.use_user_ids_as_features:
            cat_features |= {"user_id"}
        else:
            dataframe = dataframe.drop("user_id", axis="columns")
            cat_features -= {"user_id"}
        if self.use_item_ids_as_features:
            cat_features |= {"item_id"}
        else:
            dataframe = dataframe.drop("item_id", axis="columns")
            cat_features -= {"item_id"}
        if not self.use_text_features:
            dataframe = dataframe.drop(text_features, axis="columns")
            text_features = []
        return dict(
            dataframe=dataframe,
            group_id=group_id,
            label=label,
            cat_features=list(cat_features),
            text_features=list(text_features),
        )

    def postprocess_pool_kwargs(
        self,
        dataframe: pd.DataFrame,
        group_id: np.array,
        label: np.array or None = None,
        cat_features: List[str] = (),
        text_features: List[str] = (),
    ) -> dict:
        """
        Only minor datatype manipulations are performed in this method
        to comply with what catboost expects, also missing categories
        and text features are marked with self.unknown_token.
        """
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

    def build_labeled_user_item_dataframe(self, explicit) -> pd.DataFrame:
        """
        Given explicit feedback matrix, return dataframe with user-item pairs
        present in the explicit matrix, plus any extra features, except for dataset
        specific ones, which will be added automatically later in the pipeline.
        """
        dataframe = self.ratings_user_item_dataframe(explicit=explicit)
        dataframe = self.merge_rating_stats(dataframe=dataframe)
        return dataframe

    @property
    @functools.lru_cache()
    def train_user_item_dataframe(self):
        """Returns the dataframe that the model fits to."""
        return self.build_labeled_user_item_dataframe(explicit=self.explicit)

    @abc.abstractmethod
    def build_recommend_user_item_dataframe(
        self, users_explicit: "SparseTensor", user_ids: np.array, n_recommendations: int
    ) -> pd.DataFrame:
        """
        Given users' feedback matrix, their ids and requested number of
        recommendations, build a dataframe with user-item pairs,
        where user ids are the ones from received user_ids, and item_ids
        are the possible candidates for catboost ranker to rank.
        """

    def pool(self, user_item_dataframe):
        """
        Convenience method that consolidates the
        catboost Pool building pipeline from dataframe received from
        either build_labeled_dataframe or build_recommend_dataframe method.
        """
        user_item_dataframe = self.merge_features(user_item_dataframe)
        pool_kwargs = self.extract_pool_kwargs(dataframe=user_item_dataframe)
        pool_kwargs = self.postprocess_pool_kwargs(**pool_kwargs)
        pool = catboost.Pool(**pool_kwargs)
        return pool

    def fit(self):
        pool = self.pool(self.train_user_item_dataframe)
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

    def flat_catboost_predicted_ratings_to_matrix(
        self, user_ids: np.array, item_ids: np.array, flat_predicted_ratings: np.array
    ) -> "SparseTensor":
        """Unravel the user-item rating pairs into a matrix by mapping the coordinates."""
        row_ids = user_ids.astype(np.int32)
        col_ids = item_ids.astype(np.int32)
        ratings = coo_array(
            (flat_predicted_ratings, (row_ids, col_ids)),
            shape=[row_ids.max() + 1, self.n_items],
        )
        ratings = ratings.tocsr()[np.unique(row_ids)]
        ratings = self.to_torch_coo(ratings).to(torch.float32).to_dense()
        ratings[ratings == 0] = -torch.inf
        return ratings

    def common_recommend(
        self,
        users_explicit: "SparseTensor",
        user_ids,
        n_recommendations: int or None,
    ):
        n_recommendations = n_recommendations or self.n_items
        dataframe = self.build_recommend_user_item_dataframe(
            users_explicit=users_explicit,
            user_ids=user_ids,
            n_recommendations=n_recommendations,
        )
        pool = self.pool(dataframe)
        flat_predicted_ratings = self.model.predict(pool)
        ratings = self.flat_catboost_predicted_ratings_to_matrix(
            user_ids=dataframe.index.get_level_values("user_id").values,
            item_ids=dataframe.index.get_level_values("item_id").values,
            flat_predicted_ratings=flat_predicted_ratings,
        )
        recommendations = self.ratings_to_filtered_recommendations(
            explicit=users_explicit,
            ratings=ratings,
            n_recommendations=n_recommendations,
        )
        return recommendations

    @Timer()
    def recommend(self, user_ids, n_recommendations=None):
        return self.common_recommend(
            users_explicit=torch_sparse_slice(self.explicit, row_ids=user_ids),
            user_ids=user_ids,
            n_recommendations=n_recommendations,
        )

    def fictive_user_ids(self, n_users) -> torch.IntTensor:
        """
        Returns fictive user ids to use as group ids during online recommend,
        which do not intersect with user ids the model saw during fit.
        """
        return torch.arange(self.n_users, self.n_users + n_users)

    def are_user_ids_fictive(self, user_ids: np.array) -> bool:
        """
        Predicate to determine if user ids are fictive, or,
        equivalently, if we are performing an online recommend.
        """
        return any(user_ids >= self.n_users)

    @Timer()
    def online_recommend(self, users_explicit, n_recommendations=None):
        return self.common_recommend(
            users_explicit=users_explicit,
            user_ids=self.fictive_user_ids(n_users=users_explicit.shape[0]),
            n_recommendations=n_recommendations,
        )

    def get_feature_importance(self, pool) -> pd.DataFrame:
        """
        Given unlabeled pool, returns the default importance type
        for each feature, which is usually the average feature's
        impact on the prediction value or loss function.
        """
        less_indices, more_indices = split_by_group_sizes(
            group_ids=pool.get_group_id_hash(),
            group_size_threshold=self.max_gpu_query_size,
        )
        pool = pool.slice(less_indices)
        return self.model.get_feature_importance(pool, prettified=True)

    def shap_kwargs(self, user_item_dataframe):
        """
        Given user_item_dataframe, returned by one of the build_*_user_item_dataframe methods,
        returns all the necessary parameters to start drawing graphs using the shap module.
        """
        user_item_dataframe = self.merge_features(
            user_item_dataframe=user_item_dataframe
        )
        pool_kwargs = self.extract_pool_kwargs(dataframe=user_item_dataframe)
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
            user_ids = torch.IntTensor([user_id])
            recommendations = self.recommend(
                user_ids=user_ids, n_recommendations=n_recommendations
            )
            user_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids)
        else:
            recommendations = self.online_recommend(
                users_explicit=user_explicit, n_recommendations=n_recommendations
            )
            user_ids = self.fictive_user_ids(n_users=1)

        recommendations = recommendations[0].cpu().numpy()
        dataframe = self.build_recommend_user_item_dataframe(
            users_explicit=user_explicit,
            user_ids=user_ids,
            n_recommendations=n_recommendations,
        )
        dataframe = dataframe.query(
            "user_id in @user_ids.numpy() and item_id in @recommendations"
        )
        shap_kwargs = self.shap_kwargs(user_item_dataframe=dataframe)
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
            dataframe = self.merge_features(dataframe)
            postprocessed_pool_kwargs = self.postprocess_pool_kwargs(
                **self.extract_pool_kwargs(dataframe=dataframe)
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


class CatboostRecommenderBase(CatboostInterface):
    def rating_stats(
        self, ratings_dataframe, kind: CatboostInterface.FeatureKind
    ) -> pd.DataFrame:
        """Returns dataframe with kind.index_columns index and some ratings statistics in columns."""
        if kind not in [
            self.FeatureKind.user,
            self.FeatureKind.item,
        ]:
            raise ValueError(f"Rating features are undefined for kind {kind}.")
        kind = kind.value
        mean_ratings = (
            ratings_dataframe.groupby(f"{kind}_id")["rating"]
            .mean()
            .rename(f"mean_{kind}_ratings")
        )
        n_ratings = (
            ratings_dataframe.groupby(f"{kind}_id").size().rename(f"{kind}_n_ratings")
        )
        dataframe = pd.concat([mean_ratings, n_ratings], axis="columns")
        return dataframe

    def build_recommend_user_item_dataframe(
        self, users_explicit, user_ids, n_recommendations
    ):
        dataframe = pd.DataFrame(
            dict(
                user_id=np.repeat(user_ids.numpy(), self.n_items),
                item_id=np.tile(np.arange(self.n_items), len(user_ids)),
            )
        )
        dataframe = self.merge_rating_stats(
            dataframe=dataframe, users_explicit=users_explicit, user_ids=user_ids
        )
        return dataframe


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

    # def user_item_features(self):
    #     return self.FeatureReturnValue(cat_features=set(self.recommender_names))

    def ranks_dataframe(self, user_ids, item_ids, ranks):
        """
        :param user_ids: tensor of shape [n_rows]
        :param item_ids: tensor of shape [n_rows]
        :param ranks: tensor of shape [n_rows, len(self.fit_recommenders)]
        :return: dataframe with self.recommender_names columns,
            user_id and item_id multiindex
        """
        dataframe = pd.DataFrame(
            data=torch.cat(
                [user_ids.reshape(-1, 1), item_ids.reshape(-1, 1), ranks], dim=1
            ),
            columns=["user_id", "item_id"] + self.recommender_names,
        )
        dataframe = self.set_index(dataframe, kind=self.FeatureKind.user_item)
        return dataframe

    @profile(log_every=10)
    def poll_fit_recommenders(
        self, users_explicit, user_ids
    ) -> "List[torch.IntTensor]":
        """
        Given users' explicit feedback and their user_ids, returns
        a list of length equal to len(self.fit_recommenders), with
        torch matrices with the ranks of items that each separate fit_recommender
        assigned for each user, if the item was recommended at all,
        and -1 otherwise.
        """
        extended_ranks = 1 + torch.arange(self.n_items + 1).repeat(len(user_ids), 1)
        extended_ranks[:, -1] = -1
        recommenders_ranks = []
        for recommender, name in zip(self.fit_recommenders, self.recommender_names):
            if self.are_user_ids_fictive(user_ids.numpy()):
                item_ids = recommender.online_recommend(users_explicit)
            else:
                item_ids = recommender.recommend(user_ids)

            item_ids = torch.where(
                torch.BoolTensor(item_ids == recommender.invalid_recommendation_mark),
                self.n_items,
                item_ids,
            )
            recommender_ranks = torch.scatter(
                input=-torch.ones_like(extended_ranks),
                dim=1,
                index=item_ids,
                src=extended_ranks,
            )[:, :-1]
            recommenders_ranks.append(recommender_ranks)

        return recommenders_ranks

    @Timer()
    def aggregate_topk_recommendations(
        self, users_explicit, user_ids, n_recommendations, minimum_n_candidates=100
    ):
        """
        Method to determine item recommendation candidates for each user during recommending.
        Returns user-item dataframe with self.recommender_names columns
        induced by aggregated topk n_recommendations by self.fit_recommenders for
        users with corresponding user_ids. Values in self.recommender_names
        columns at index [user_id, item_id] are equal to rank of item_id in recommender's
        recommendations for user_id.
        """
        topk = min(
            self.n_items,
            minimum_n_candidates + n_recommendations * len(self.fit_recommenders),
        )
        recommenders_ranks = self.poll_fit_recommenders(
            user_ids=user_ids, users_explicit=users_explicit
        )
        items_reciprocal_ranks_sum = sum(
            [torch.where(i == -1, 0, 1 / i) for i in recommenders_ranks]
        )
        _, topk_item_ids = items_reciprocal_ranks_sum.topk(k=topk, dim=1)
        topk_ranks = [
            torch.take_along_dim(input=recommender_rank, indices=topk_item_ids, dim=1)
            for recommender_rank in recommenders_ranks
        ]
        repeated_user_ids = einops.repeat(user_ids, f"u -> u {topk}")
        topk_item_ids = einops.rearrange(topk_item_ids, "u k -> (u k)")
        topk_ranks = einops.rearrange(topk_ranks, "r u k -> (u k) r")
        dataframe = self.ranks_dataframe(
            user_ids=repeated_user_ids, item_ids=topk_item_ids, ranks=topk_ranks
        )
        return dataframe

    def user_batches(
        self, user_ids: torch.IntTensor, users_explicit: "SparseTensor", tqdm_stage=None
    ):
        """Batched iterator over synchronized user_ids and users_explicit."""
        for batch_indices in tqdm(
            iterable=torch.arange(len(user_ids)).split(
                split_size=self.dataframe_build_batch_size
            ),
            desc=f"Building {tqdm_stage or ''} dataframe for users: {clipped_tensor_string_representation(user_ids)}",
            disable=not bool(tqdm_stage),
        ):
            batch_user_ids = user_ids[batch_indices]
            batch_users_explicit = torch_sparse_slice(
                users_explicit, row_ids=batch_indices
            )
            yield batch_user_ids, batch_users_explicit

    def build_labeled_user_item_dataframe(self, explicit):
        if explicit.shape != (
            expected_shape := torch.Size([self.n_users, self.n_items])
        ):
            raise ValueError(
                "Shape of explicit feedback matrix doesn't "
                "match the expected shape of [self.n_users, self.n_items]:"
                f"{explicit.shape} != {expected_shape}"
            )

        dataframes = []
        for batch_user_ids, batch_users_explicit in self.user_batches(
            user_ids=torch.arange(self.n_users),
            users_explicit=explicit,
            tqdm_stage="labeled",
        ):
            recommenders_ranks = self.poll_fit_recommenders(
                user_ids=batch_user_ids,
                users_explicit=batch_users_explicit,
            )
            user_pos, item_ids = (
                self.to_torch_coo(batch_users_explicit).coalesce().indices()
            )
            train_ranks = [rank[user_pos, item_ids] for rank in recommenders_ranks]
            train_ranks = einops.rearrange(train_ranks, "r n -> n r")
            train_dataframe = self.ranks_dataframe(
                user_ids=batch_user_ids[user_pos], item_ids=item_ids, ranks=train_ranks
            )
            dataframes.append(train_dataframe)

        dataframe = pd.concat(dataframes)
        super_dataframe = super().build_labeled_user_item_dataframe(explicit=explicit)
        assert (
            len(dataframe.index.symmetric_difference(super_dataframe.index)) == 0
        ), "Index of batch-constructed dataframe doesn't match the expected one."
        dataframe = dataframe.join(super_dataframe)
        return dataframe

    def build_recommend_user_item_dataframe(
        self, users_explicit, user_ids, n_recommendations
    ):
        dataframes = []
        for batch_user_ids, batch_users_explicit in self.user_batches(
            user_ids=user_ids, users_explicit=users_explicit
        ):
            dataframes.append(
                self.aggregate_topk_recommendations(
                    users_explicit=batch_users_explicit,
                    user_ids=batch_user_ids,
                    n_recommendations=n_recommendations,
                )
            )
        dataframe = pd.concat(dataframes)
        dataframe = self.merge_rating_stats(
            dataframe=dataframe, users_explicit=users_explicit, user_ids=user_ids
        )
        return dataframe


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
