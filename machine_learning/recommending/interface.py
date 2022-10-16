import abc
from typing import TYPE_CHECKING, TypeVar

import torch

from machine_learning.recommending.utils import torch_sparse_to_scipy_coo
from my_tools.utils import scipy_to_torch_sparse

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


SparseTensor = TypeVar("SparseTensor", bound=torch.Tensor)
Pickleable = TypeVar("Pickleable")


# TODO: Catboost, shap, update sweeps, partial fit? update lit interface
# when to use torch.cuda.init()?


class RecommenderModuleInterface(torch.nn.Module, abc.ABC):
    def __init__(self, *, n_users, n_items):
        """
        Only lightweight hyperparameters should be passed,
        any heavy objects should be optional.
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

    @abc.abstractmethod
    def forward(
        self, user_ids: torch.IntTensor, item_ids: torch.IntTensor
    ) -> torch.FloatTensor or SparseTensor:
        ...

    @torch.inference_mode()
    def recommend(
        self, user_ids: torch.IntTensor, n_recommendations=10
    ) -> torch.IntTensor:
        """Returns recommended item ids."""
        relevance = self(user_ids=user_ids, item_ids=torch.arange(self.n_items))
        values, recommendations = torch.topk(input=relevance, k=n_recommendations)
        return recommendations

    def partial_fit(self, explicit_feedback):
        ...

    @abc.abstractmethod
    def _new_users(self, n_new_users: int) -> None:
        """Inner logic necessary to be ready to work with new users."""
        ...

    def new_users(self, n_new_users: int) -> torch.IntTensor:
        """Convenience wrapper to add new users."""
        self._new_users(n_new_users)
        self.n_users += n_new_users
        new_user_ids = torch.arange(self.n_users - n_new_users, self.n_users)
        return new_user_ids

    @abc.abstractmethod
    def _new_items(self, n_new_items: int) -> None:
        """Inner logic necessary to be ready to work with new items."""

    def new_items(self, n_new_items: int) -> torch.IntTensor:
        """Convenience wrapper to add new items."""
        self._new_items(n_new_items)
        self.n_items += n_new_items
        new_item_ids = torch.arange(self.n_items - n_new_items, self.n_items)
        return new_item_ids

    def save(self) -> Pickleable:
        """
        Whatever extra non-torch.Parameter parameters need to be saved for inference,
        will be taken here on checkpoint save and passed to load method on checkpoint load.
        """

    def load(self, saved: Pickleable) -> None:
        """Init module from whatever was saved to be ready for inference."""


class FitExplicitInterfaceMixin:
    @abc.abstractmethod
    def fit(self, explicit_feedback: "csr_matrix") -> None:
        ...


class InMemoryRecommender(RecommenderModuleInterface, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            name="explicit_feedback",
            tensor=torch.sparse_coo_tensor(size=(self.n_users, self.n_items)),
        )

    def save_explicit_feedback(self, explicit_feedback: "csr_matrix"):
        self.explicit_feedback = scipy_to_torch_sparse(explicit_feedback)

    @property
    def explicit_feedback_scipy_coo(self):
        return torch_sparse_to_scipy_coo(self.explicit_feedback)

    def new_users(self, n_new_users):
        """_new_users method will receive an already expanded explicit matrix."""
        new_users_feedback = torch.sparse_coo_tensor(size=(n_new_users, self.n_items))
        self.explicit_feedback = torch.cat([self.explicit_feedback, new_users_feedback])
        return super().new_users(n_new_users)

    def new_items(self, n_new_items):
        """_new_items method will receive an already expanded explicit matrix."""
        new_items_feedback = torch.sparse_coo_tensor(size=(self.n_users, n_new_items))
        self.explicit_feedback = torch.cat(
            [self.explicit_feedback, new_items_feedback], dim=1
        )
        return super().new_items(n_new_items)


class RecommendingLossInterface:
    @abc.abstractmethod
    def __call__(
        self, explicit: SparseTensor, model_ratings: torch.FloatTensor or SparseTensor
    ) -> torch.FloatTensor:
        assert torch.is_same_size(explicit, model_ratings)
        loss = ...
        assert loss.numel() == 1
        return loss
