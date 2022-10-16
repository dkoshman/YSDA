import torch

from my_tools.models import register_regularization_hook
from my_tools.utils import scipy_to_torch_sparse

from ..lit import LitRecommenderBase
from ..data import SparseDataset
from ..interface import RecommenderModuleInterface, InMemoryRecommender
from ..utils import build_bias, build_weight, torch_sparse_slice


class MatrixFactorizationBase(RecommenderModuleInterface):
    """Predicted_rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(self, latent_dimension=10, weight_decay=1.0e-3, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay
        self.user_weight = build_weight(self.n_users, latent_dimension)
        self.user_bias = build_bias(self.n_users, 1)
        self.item_weight = build_weight(self.n_items, latent_dimension)
        self.item_bias = build_bias(self.n_items, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]
        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias

        # Need to add regularization here because otherwise optimizer will decay all weights,
        # not only those corresponding to user and item ids. Also, it is important to add gradient
        # hooks after the forward calculations, otherwise decay messes with the model.
        for parameter in [user_weight, item_weight, user_bias, item_bias]:
            register_regularization_hook(parameter, self.weight_decay)

        return rating

    def _new_users(self, n_new_users):
        new_users_weight = self.user_weight.mean(dim=0).repeat(n_new_users, 1)
        user_weight = torch.cat([self.user_weight, new_users_weight])
        self.user_weight = torch.nn.Parameter(user_weight)

        user_bias = torch.cat([self.user_bias, torch.zeros(n_new_users, 1)])
        self.user_bias = torch.nn.Parameter(user_bias)

    def _new_items(self, n_new_items):
        new_items_weight = self.item_weight.mean(dim=0).repeat(n_new_items, 1)
        item_weight = torch.cat([self.item_weight, new_items_weight])
        self.item_weight = torch.nn.Parameter(item_weight)

        item_bias = torch.cat([self.item_bias, torch.zeros(n_new_items, 1)])
        self.item_bias = torch.nn.Parameter(item_bias)


class ConstrainedProbabilityMatrixFactorization(
    MatrixFactorizationBase, InMemoryRecommender
):
    def __init__(self, *args, latent_dimension=10, **kwargs):
        super().__init__(*args, latent_dimension=latent_dimension, **kwargs)
        self.item_rating_effect_weight = build_weight(self.n_items, latent_dimension)
        self.implicit_feedback_normalized = None

    def init_implicit_feedback_normalized(self):
        implicit_feedback = self.explicit_feedback_scipy_coo > 0
        implicit_feedback_normalized = implicit_feedback.multiply(
            1 / (implicit_feedback.sum(axis=1) + 1e-8)
        ).astype("float32")
        self.implicit_feedback_normalized = scipy_to_torch_sparse(
            implicit_feedback_normalized
        )

    def forward(self, user_ids, item_ids):
        if self.implicit_feedback_normalized is None:
            self.init_implicit_feedback_normalized()

        # Need to clone to avoid gradient hook accumulation on same tensor and subsequent memory leak
        item_rating_effect_weight = self.item_rating_effect_weight.clone()

        item_weight = self.item_weight[item_ids]

        users_implicit_feedback = torch_sparse_slice(
            self.implicit_feedback_normalized, row_ids=user_ids, device=self.bias.device
        )
        user_weights_offset_caused_by_their_ratings = (
            users_implicit_feedback @ item_rating_effect_weight
        )

        ratings = super().forward(user_ids=user_ids, item_ids=item_ids)
        ratings += user_weights_offset_caused_by_their_ratings @ item_weight.T

        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        scale_down = self.user_weight.shape[0] / len(user_ids)
        register_regularization_hook(
            item_rating_effect_weight, self.weight_decay / scale_down
        )
        return ratings

    def _new_users(self, n_new_users) -> None:
        self.init_implicit_feedback_normalized()
        super()._new_users(n_new_users)

    def _new_items(self, n_new_items) -> None:
        self.init_implicit_feedback_normalized()
        new_effect = self.item_rating_effect_weight.mean(dim=0).repeat(n_new_items, 1)
        item_rating_effect_weight = torch.cat(
            [self.item_rating_effect_weight, new_effect]
        )
        self.item_rating_effect_weight = torch.nn.Parameter(item_rating_effect_weight)
        super()._new_items(n_new_items)


class MFRecommender(LitRecommenderBase):
    @property
    def class_candidates(self):
        return super().class_candidates + [
            MatrixFactorizationBase,
            ConstrainedProbabilityMatrixFactorization,
        ]

    def train_dataloader(self):
        return self.build_dataloader(
            dataset=SparseDataset(self.train_explicit),
            sampler_type="grid",
            shuffle=True,
        )

    def common_step(self, batch):
        ratings = self(**batch)
        loss = self.loss(explicit=batch["explicit"], model_ratings=ratings)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log("val_loss", loss)
        return loss
