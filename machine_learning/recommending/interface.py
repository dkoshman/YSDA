import abc
from typing import TYPE_CHECKING, Any

import torch
import wandb
from torch import IntTensor

from machine_learning.recommending.maths import (
    Distance,
    cosine_distance,
    weighted_average,
)
from my_tools.models import IgnoreShapeMismatchOnLoadStateDictMixin
from my_tools.utils import to_torch_coo, to_scipy_coo, torch_sparse_slice

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from my_tools.utils import SparseTensor, Pickleable


# TODO: app on huggingface, telegram bot


class RecommenderModuleInterface(torch.nn.Module, abc.ABC):
    @abc.abstractmethod
    def __init__(self, explicit: "SparseTensor" or "spmatrix" or None = None):
        """
        :param explicit: the explicit feedback matrix,
        it is allowed to be None only when loading from state dict
        """
        super().__init__()

    @property
    @abc.abstractmethod
    def n_users(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def n_items(self) -> int:
        ...

    @abc.abstractmethod
    def forward(
        self, user_ids: torch.IntTensor, item_ids: torch.IntTensor
    ) -> torch.FloatTensor:
        ...

    @abc.abstractmethod
    def recommend(
        self,
        user_ids: torch.IntTensor,
        n_recommendations: int or None = None,
        filter_already_liked_items=True,
    ) -> torch.IntTensor:
        """Returns recommended item ids."""

    @abc.abstractmethod
    def online_ratings(self, explicit: "SparseTensor" or "spmatrix") -> torch.Tensor:
        """Generate ratings for new users defined by their explicit feedback"""

    @abc.abstractmethod
    def online_recommend(
        self,
        explicit: "SparseTensor" or "spmatrix",
        n_recommendations: int = 10,
    ) -> torch.IntTensor:
        """Generate recommendations for new users defined by their explicit feedback."""


class RecommenderModuleBase(
    IgnoreShapeMismatchOnLoadStateDictMixin, RecommenderModuleInterface
):
    def __init__(self, explicit: "SparseTensor" or "spmatrix" or None = None):
        super().__init__()
        if explicit is not None:
            explicit = to_torch_coo(explicit)
        self.register_buffer(name="explicit", tensor=explicit)
        self.to_torch_coo = to_torch_coo
        self.to_scipy_coo = to_scipy_coo

    @property
    def device(self):
        return self.explicit.device

    @property
    def n_users(self):
        return 0 if self.explicit is None else self.explicit.shape[0]

    @property
    def n_items(self):
        return 0 if self.explicit is None else self.explicit.shape[1]

    @abc.abstractmethod
    def forward(
        self, user_ids: torch.IntTensor, item_ids: torch.IntTensor
    ) -> torch.FloatTensor:
        ...

    @staticmethod
    def filter_already_liked_items(explicit, ratings):
        return torch.where(explicit.to_dense() > 0, torch.finfo().min, ratings)

    @staticmethod
    def ratings_to_recommendations(ratings, n_recommendations):
        item_ratings, item_ids = torch.topk(input=ratings, k=n_recommendations)
        return item_ids

    def recommend(
        self,
        user_ids: torch.IntTensor,
        n_recommendations: int = 10,
        filter_already_liked_items=True,
    ) -> torch.IntTensor:
        with torch.inference_mode():
            ratings = self(user_ids=user_ids, item_ids=torch.arange(self.n_items))
        if filter_already_liked_items:
            users_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids).to(
                self.device
            )
            ratings = self.filter_already_liked_items(users_explicit, ratings)
        recommendations = self.ratings_to_recommendations(ratings, n_recommendations)
        return recommendations

    def online_nn_ratings(
        self,
        explicit: "SparseTensor" or spmatrix,
        distance: Distance or None = cosine_distance,
        n_neighbours: int or None = 10,
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

    def online_ratings(self, users_explicit: "SparseTensor" or spmatrix):
        """
        The fallback method is based on nearest neighbours,
        as it can be applied to any model, but may give suboptimal
        recommendations. Feel free to overwrite this
        method if model natively supports online ratings.
        """
        return self.online_nn_ratings(users_explicit)

    def online_recommend(
        self,
        users_explicit: "SparseTensor" or spmatrix,
        n_recommendations: int or None = None,
    ) -> torch.IntTensor:
        users_explicit = self.to_torch_coo(users_explicit).to(self.device)
        ratings = self.online_ratings(users_explicit)
        ratings = self.filter_already_liked_items(users_explicit, ratings)
        recommendations = self.ratings_to_recommendations(ratings, n_recommendations)
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

    def save_state_to_artifact(self, artifact_name):
        if wandb.run is None:
            raise ValueError("Wandb run not initialized, can't save artifact.")

        state_dict_path = "tmp.pt"
        torch.save(self.state_dict(), state_dict_path)
        artifact = wandb.Artifact(
            name=artifact_name,
            type="state_dict",
            metadata=dict(class_name=self.__class__.__name__),
        )
        artifact.add_file(local_path=state_dict_path, name="state_dict")
        wandb.run.log_artifact(artifact)


class FitExplicitInterfaceMixin:
    @abc.abstractmethod
    def fit(self) -> None:
        """Fit to self.explicit feedback matrix."""


class ExplanationMixin:
    @abc.abstractmethod
    def explain_recommendations(
        self,
        user_id: int or IntTensor = None,
        user_explicit: "SparseTensor" = None,
        n_recommendations=10,
        log: bool = False,
        logging_prefix: str = "",
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
