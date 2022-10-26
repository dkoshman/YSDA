import torch

from my_tools.models import register_regularization_hook
from my_tools.utils import to_torch_coo, torch_sparse_slice

from ..lit import LitRecommenderBase
from ..data import SparseDataset
from ..interface import RecommenderModuleBase
from ..utils import build_bias, build_weight


class MatrixFactorization(RecommenderModuleBase):
    """Predicted_rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(self, explicit=None, latent_dimension=10, weight_decay=1.0e-3):
        super().__init__(explicit=explicit)
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


class ConstrainedProbabilityMatrixFactorization(MatrixFactorization):
    def __init__(self, *args, latent_dimension=10, **kwargs):
        super().__init__(*args, latent_dimension=latent_dimension, **kwargs)
        self.item_rating_effect_weight = build_weight(self.n_items, latent_dimension)
        self.implicit_feedback_normalized = None

    def init_implicit_feedback_normalized(self):
        implicit_feedback = self.to_scipy_coo(self.explicit) > 0
        implicit_feedback_normalized = implicit_feedback.multiply(
            1 / (implicit_feedback.sum(axis=1) + 1e-8)
        ).astype("float32")
        self.implicit_feedback_normalized = to_torch_coo(implicit_feedback_normalized)

    def forward(self, user_ids, item_ids):
        if self.implicit_feedback_normalized is None:
            self.init_implicit_feedback_normalized()

        # Need to clone to avoid gradient hook accumulation on same tensor and subsequent memory leak
        item_rating_effect_weight = self.item_rating_effect_weight.clone()

        item_weight = self.item_weight[item_ids]

        users_implicit_feedback = torch_sparse_slice(
            self.implicit_feedback_normalized, row_ids=user_ids
        ).to(self.bias.device)
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


class MyMatrixFactorization(RecommenderModuleBase):
    def __init__(self, explicit=None, latent_dimension=10, weight_decay=1.0e-3):
        super().__init__(explicit=explicit)
        self.ratings_mf = MatrixFactorization(
            latent_dimension=latent_dimension,
            weight_decay=weight_decay,
            explicit=explicit,
        )
        self.confidence_mf = MatrixFactorization(
            latent_dimension=latent_dimension,
            weight_decay=weight_decay,
            explicit=explicit,
        )

    def probability(self, user_ids, item_ids):
        confidence = self.confidence_mf(user_ids=user_ids, item_ids=item_ids)
        probability = torch.sigmoid(confidence)
        probability = probability / probability.sum()
        return probability

    def forward(self, user_ids, item_ids):
        ratings = self.ratings_mf(user_ids=user_ids, item_ids=item_ids)
        probability = self.probability(user_ids=user_ids, item_ids=item_ids)
        return ratings * probability


class MFRecommender(LitRecommenderBase):
    @property
    def class_candidates(self):
        return super().class_candidates + [
            MatrixFactorization,
            ConstrainedProbabilityMatrixFactorization,
        ]

    def train_dataloader(self):
        return self.build_dataloader(
            dataset=SparseDataset(self.train_explicit()),
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


class MyMFRecommender(MFRecommender):
    @property
    def class_candidates(self):
        return super().class_candidates + [MyMatrixFactorization]

    def common_step(self, batch):
        explicit = batch["explicit"].to_dense()
        user_ids = batch["user_ids"]
        item_ids = batch["item_ids"]
        ratings = self.model.ratings_mf(user_ids=user_ids, item_ids=item_ids)
        probability = self.model.probability(user_ids=user_ids, item_ids=item_ids)
        implicit = explicit > 0
        loss = (
            implicit * ((ratings - explicit) ** 2 - torch.log(probability))
        ).sum() / implicit.sum()
        return loss
