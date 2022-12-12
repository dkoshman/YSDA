import abc
import warnings
from abc import ABC
from typing import TYPE_CHECKING, Any

import torch
import wandb

from .maths import Distance, cosine_distance, weighted_average
from .utils import profile, Timer
from my_tools.utils import to_torch_coo, to_scipy_coo, torch_sparse_slice

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from my_tools.utils import SparseTensor, Pickleable


class RecommenderModuleInterface(torch.nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def device(self):
        """Returns the device on which the module resides at the moment."""

    @abc.abstractmethod
    def forward(
        self, user_ids: torch.IntTensor, item_ids: torch.IntTensor
    ) -> torch.FloatTensor:
        ...

    @abc.abstractmethod
    def recommend(
        self, user_ids: torch.IntTensor, n_recommendations: int = None
    ) -> torch.IntTensor:
        """Returns recommended item ids."""

    @abc.abstractmethod
    def online_ratings(self, explicit: "SparseTensor" or "spmatrix") -> torch.Tensor:
        """Generate ratings for new users defined by their explicit feedback"""

    @abc.abstractmethod
    def online_recommend(
        self,
        explicit: "SparseTensor" or "spmatrix",
        n_recommendations: int = None,
    ) -> torch.IntTensor:
        """Generate recommendations for new users defined by their explicit feedback."""


class RecommenderModuleBase(RecommenderModuleInterface, ABC):
    explicit: torch.Tensor

    def __init__(
        self,
        n_users: int,
        n_items: int,
        explicit: "SparseTensor" or "spmatrix" or None = None,
        persistent_explicit=False,
        invalid_recommendation_mark: int = -1,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        if explicit is None:
            explicit = torch.sparse_coo_tensor(size=(n_users, n_items))
        else:
            explicit = to_torch_coo(explicit)
        self.register_buffer(
            name="explicit", tensor=explicit, persistent=persistent_explicit
        )
        self.invalid_recommendation_mark = invalid_recommendation_mark
        self.to_torch_coo = to_torch_coo
        self.to_scipy_coo = to_scipy_coo

    @property
    def device(self):
        if self.explicit is not None:
            return self.explicit.device
        for parameter in self.parameters():
            return parameter.device

    @Timer()
    @profile()
    def ratings_to_filtered_recommendations(
        self, explicit, ratings, n_recommendations=None
    ):
        n_recommendations = n_recommendations or self.n_items
        ratings = torch.where(explicit.to_dense() > 0, -torch.inf, ratings)
        item_ratings, item_ids = torch.topk(input=ratings, k=n_recommendations)
        item_ids[item_ratings == -torch.inf] = self.invalid_recommendation_mark
        self.check_invalid_recommendations(recommendations=item_ids, warn=False)
        return item_ids

    def check_invalid_recommendations(self, recommendations, warn: bool):
        invalid_items_mask = (recommendations == self.invalid_recommendation_mark).to(
            torch.float32
        )
        users_without_enough_recommendations = invalid_items_mask.any(1).to(
            torch.float32
        )
        if wandb.run is not None:
            wandb.log(
                {
                    "sanity_check/average_n_invalid_recommendations_per_user": invalid_items_mask.mean(),
                    "sanity_check/not_enough_recommendations_frequency": users_without_enough_recommendations.mean(),
                }
            )
        if warn and any(users_without_enough_recommendations):
            warning_message = (
                f"Model failed to generate requested number of recommendations for users.\n"
                f"Model was asked for {recommendations.shape[1]} recommendations among "
                f"{self.n_items} items, but {users_without_enough_recommendations.sum()} "
                f"users out of batch total {recommendations.shape[0]} did not get enough recommendations."
            )
            warnings.warn(warning_message, category=RuntimeWarning)

    def recommend(self, user_ids, n_recommendations=None):
        with torch.inference_mode():
            ratings = self(user_ids=user_ids, item_ids=torch.arange(self.n_items))
        users_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids).to(
            self.device
        )
        recommendations = self.ratings_to_filtered_recommendations(
            explicit=users_explicit,
            ratings=ratings,
            n_recommendations=n_recommendations,
        )
        return recommendations

    def online_nn_ratings(
        self,
        explicit: "SparseTensor" or spmatrix,
        distance: Distance or None = cosine_distance,
        n_neighbours: int = 10,
    ):
        explicit = self.to_torch_coo(explicit).to(self.device).detach().clone()
        distances = distance(explicit, self.explicit.detach().clone())
        neighbors_distances, nearest_user_ids = torch.topk(
            distances, k=n_neighbours, largest=False
        )
        ratings = torch.empty(
            *explicit.shape,
            dtype=torch.float32,
            device=self.device,
        )
        for i, (user_ids, distances_to_users) in enumerate(
            zip(nearest_user_ids, neighbors_distances)
        ):
            with torch.inference_mode():
                neighbours_ratings = self(
                    user_ids=user_ids, item_ids=torch.arange(self.n_items)
                )
            user_ratings = weighted_average(
                tensor=neighbours_ratings,
                weights=(distances_to_users + torch.finfo().eps) ** -1,
            )
            ratings[i] = user_ratings
        return ratings

    def online_ratings(self, users_explicit):
        """
        The fallback method is based on nearest neighbours,
        as it can be applied to any model, but may give suboptimal
        recommendations. Feel free to overwrite this
        method if model natively supports online ratings.
        """
        return self.online_nn_ratings(users_explicit)

    def forward(self, user_ids, item_ids):
        users_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids)
        ratings = self.online_ratings(users_explicit=users_explicit)
        return ratings[:, item_ids]

    def online_recommend(self, users_explicit, n_recommendations=None):
        users_explicit = self.to_torch_coo(users_explicit).to(self.device)
        ratings = self.online_ratings(users_explicit)
        recommendations = self.ratings_to_filtered_recommendations(
            explicit=users_explicit,
            ratings=ratings,
            n_recommendations=n_recommendations,
        )
        return recommendations

    def get_extra_state(self) -> "Pickleable":
        """
        Whatever extra non-tensor parameters need to be saved for inference,
        will be taken here on checkpoint save and passed to load method on checkpoint load.
        This is torch.nn.Module method, this stub is just for clarity.
        """

    def set_extra_state(self, state: "Pickleable") -> None:
        """
        Init module from whatever was returned by getter to be ready for inference.
        This is torch.nn.Module method, this stub is just for clarity.
        """


class FitExplicitInterfaceMixin:
    @abc.abstractmethod
    def fit(self) -> None:
        """Fit to self.explicit feedback matrix."""


class ExplanationMixin:
    @abc.abstractmethod
    def explain_recommendations_for_user(
        self,
        user_id: int = None,
        user_explicit: "SparseTensor" = None,
        n_recommendations=10,
        log: bool = False,
        logging_prefix: str = "explanation/",
    ) -> Any:
        """
        Return any data that will aid in understanding why
        the model recommends the way it does.
        :param user_id: the id of user, for whom the
        recommendations are generated, if he has one
        :param user_explicit: the user's feedback of
        shape [1, self.n_items], if he doesn't have an id
        :param n_recommendations: number of recommendations
        :param log: whether to log explanations to wandb
        :param logging_prefix: the prefix for log entries
        """


class UnpopularRecommenderMixin:
    """
    I want to recommend items which the users haven't previously seen,
    so given probability p_ui that user has seen item, I want to decrease
    the expected rating r_ui by an amount depending on that probability:
    predicted_relevance_ui = r_ui - g(p_ui),
    where g is a non-decreasing function.

    For convenience and without much complexity sacrifice, p_ui can be
    factorized as p_u * p_i, and p_u and p_i can be
    estimated as sample frequencies.

    The implementation below uses g(x) = unpopularity_coef * log(p_ui).
    """

    unpopularity_coef: torch.Tensor
    users_activity: torch.Tensor
    items_popularity: torch.Tensor

    def init_unpopular_recommender_mixin(
        self: RecommenderModuleBase, unpopularity_coef=1e-3
    ):
        self.register_buffer(
            name="unpopularity_coef", tensor=torch.tensor(unpopularity_coef)
        )
        self.register_buffer(name="users_activity", tensor=torch.zeros(self.n_users))
        self.register_buffer(name="items_popularity", tensor=torch.zeros(self.n_items))

    def fit_unpopular_recommender_mixin(self: RecommenderModuleBase):
        self.users_activity = torch.tensor(
            (self.to_scipy_coo(self.explicit) > 0).mean(1).A.squeeze(1)
        )
        self.items_popularity = torch.tensor(
            (self.to_scipy_coo(self.explicit) > 0).mean(0).A.squeeze(0)
        )

    def probability_that_user_has_seen_item(self, users_activity):
        return torch.einsum("u, i -> ui", users_activity, self.items_popularity)

    def additive_rating_offset(self, users_activity):
        p_ui = self.probability_that_user_has_seen_item(users_activity=users_activity)
        eps = torch.finfo().eps
        return -self.unpopularity_coef * torch.log(torch.FloatTensor(eps + p_ui))

    @property
    def mean_user_activity(self):
        """Convenience method."""
        return self.users_activity.mean(dim=0, keepdims=True)


class RecommendingLossInterface:
    def __init__(self, explicit):
        pass

    @abc.abstractmethod
    def __call__(
        self,
        explicit: "SparseTensor",
        model_ratings: torch.FloatTensor = None,
        model: "RecommenderModuleBase" = None,
        user_ids: torch.IntTensor = None,
        item_ids: torch.IntTensor = None,
    ) -> torch.FloatTensor:
        ...
