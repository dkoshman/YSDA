import abc
import copy

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


class MyMfSlimHybridRecommender(RecommenderModuleBase):
    def __init__(self, explicit=None, latent_dimension=10):
        super().__init__(explicit=explicit)
        self.mf_linear = MFLinear(
            in_dim=self.n_items,
            latent_dim=latent_dimension,
            out_dim=self.n_items,
            bias=True,
        )

    def forward(self, user_ids, item_ids):
        users_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids)
        ratings = self.online_ratings(users_explicit=users_explicit)
        return ratings[:, item_ids]

    def online_ratings(self, users_explicit):
        users_explicit = users_explicit.to(self.device, torch.float32)
        ratings = self.mf_linear(users_explicit)
        return ratings


class MFConfidenceRecommenderBase(RecommenderModuleBase, abc.ABC):
    @abc.abstractmethod
    def ratings(self, user_ids, item_ids):
        ...

    @abc.abstractmethod
    def probability(self, user_ids, item_ids):
        ...

    def forward(self, user_ids, item_ids):
        ratings = self.ratings(user_ids=user_ids, item_ids=item_ids)
        probability = self.probability(user_ids=user_ids, item_ids=item_ids)
        return ratings * probability


class MyMFConfidenceRecommender(MFConfidenceRecommenderBase):
    def __init__(self, explicit=None, latent_dimension=10):
        super().__init__(explicit=explicit)
        self.ratings_mf = MatrixFactorization(
            latent_dimension=latent_dimension,
            weight_decay=0,
            explicit=explicit,
        )
        self.confidence_mf = MatrixFactorization(
            latent_dimension=latent_dimension,
            weight_decay=0,
            explicit=explicit,
        )

    def ratings(self, user_ids, item_ids):
        return self.ratings_mf(user_ids=user_ids, item_ids=item_ids)

    def probability(self, user_ids, item_ids):
        confidence = self.confidence_mf(user_ids=user_ids, item_ids=item_ids)
        confidence = torch.sigmoid(confidence)
        in_sample_probability = confidence / confidence.sum()
        return in_sample_probability


class MFLinear(torch.nn.Module):
    """Factorized linear layer: y = x @ encoder @ decoder + bias"""

    def __init__(self, in_dim, latent_dim, out_dim, bias=True):
        super().__init__()
        self.encoder = build_weight(in_dim, latent_dim)
        self.decoder = build_weight(latent_dim, out_dim)
        self.bias = build_bias(out_dim) if bias else 0

    def forward(self, x):
        out = x @ self.encoder @ self.decoder + self.bias
        return out


class MyMfSlimConfidenceHybridRecommender(MFConfidenceRecommenderBase):
    def __init__(self, explicit=None, latent_dimension=10):
        super().__init__(explicit=explicit)
        self.ratings_mf = MFLinear(
            in_dim=self.n_items,
            latent_dim=latent_dimension,
            out_dim=self.n_items,
            bias=True,
        )
        self.ratings_user_bias = build_bias(self.n_users, 1)
        self.ratings_bias = torch.nn.Parameter(torch.zeros(1))
        self.confidence_mf = copy.deepcopy(self.ratings_mf)
        self.confidence_user_bias = build_bias(self.n_users, 1)
        self.confidence_bias = torch.nn.Parameter(torch.zeros(1))

    def probability(self, user_ids, item_ids):
        probability = self.online_probability(
            users_explicit=torch_sparse_slice(self.explicit, row_ids=user_ids),
            user_bias=self.confidence_user_bias[user_ids],
        )
        return probability[:, item_ids]

    def online_probability(self, users_explicit, user_bias=None):
        users_explicit = users_explicit.to(self.device, torch.float32)
        if user_bias is None:
            user_bias = self.confidence_user_bias.mean(0)
        confidence = (
            self.confidence_mf(users_explicit) + user_bias + self.confidence_bias
        )
        confidence = torch.sigmoid(confidence)
        in_sample_probability = confidence / confidence.sum()
        return in_sample_probability

    def ratings(self, user_ids, item_ids):
        ratings = self.online_ratings(
            users_explicit=torch_sparse_slice(self.explicit, row_ids=user_ids),
            user_bias=self.ratings_user_bias[user_ids],
        )
        return ratings[:, item_ids]

    def online_ratings(self, users_explicit, user_bias=None):
        users_explicit = users_explicit.to(self.device, torch.float32)
        if user_bias is None:
            user_bias = self.ratings_user_bias.mean(0)
        ratings = self.ratings_mf(users_explicit) + user_bias + self.ratings_bias
        return ratings


class MFRecommender(LitRecommenderBase):
    @property
    def class_candidates(self):
        return super().class_candidates + [
            MatrixFactorization,
            ConstrainedProbabilityMatrixFactorization,
            MyMfSlimHybridRecommender,
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


# TODO: add regularization to mfslim, why is regular mf exhibiting double minimum?
#  remove bias from mf? add more logs to find out whats going on, add SWA
class MyMFRecommender(MFRecommender):
    """
    Recommender based on the following data generation model:
    1. Given set of users U and set of items I,
    a user-item pair [u, i] is chosen with probability p_ui.
    2. Rating r_ui is generated from normal distribution N(mean_ui, var)

    Then by maximum likelihood principle, we are seeking the parameters defined by:
        argmax(P_sample) =
        argmax( product_ui( p_ui * N_{mean_ui, var}(r_ui)))) =
        argmax( sum_ui ( log p_ui - (mean_ui - r_ui) ** 2 / (2 * var))) =
        argmin( sum_ui ( (mean_ui - r_ui) ** 2 / ( 2 * var) - log p_ui)))

    And the predicted relevance of item i for user u is:
        relevance_ui = p_ui * mean_ui
    """

    @property
    def class_candidates(self):
        return super().class_candidates + [
            MyMFConfidenceRecommender,
            MyMfSlimConfidenceHybridRecommender,
        ]

    def common_step(self, batch):
        explicit = batch["explicit"].to_dense()
        user_ids = batch["user_ids"]
        item_ids = batch["item_ids"]

        ratings = self.model.ratings(user_ids=user_ids, item_ids=item_ids)
        probability = self.model.probability(user_ids=user_ids, item_ids=item_ids)

        sample_mask = explicit > 0
        var = self.hparams["loss_config"]["ratings_deviation"] ** 2
        loss = (
            sample_mask
            * ((ratings - explicit) ** 2 / (2 * var) - torch.log(probability))
        ).sum() / sample_mask.sum()
        return loss
