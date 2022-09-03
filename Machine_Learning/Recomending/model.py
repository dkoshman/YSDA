import torch

from my_ml_tools.models import weight_decay
from my_ml_tools.utils import scipy_to_torch_sparse


def _build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def _build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


class ProbabilityMatrixFactorization(torch.nn.Module):
    """predicted rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(self, n_users, n_items, latent_dimension, regularization_lambda):
        super().__init__()

        self.regularization_lambda = regularization_lambda

        self.user_weight = _build_weight(n_users, latent_dimension)
        self.user_bias = _build_bias(n_users, 1)

        self.item_weight = _build_weight(n_items, latent_dimension)
        self.item_bias = _build_bias(n_items, 1)

        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.sigmoid = torch.nn.Sigmoid()

    def linear_forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]
        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias

        # Need to add regularization here because otherwise optimizer will decay all weights,
        # not only those corresponding to user and item ids. Also it is important to add gradient
        # hooks after the forward calculations, otherwise decay messes with the model.
        for parameter in [user_weight, item_weight, user_bias, item_bias]:
            weight_decay(parameter, self.regularization_lambda)

        return rating

    def forward(self, user_ids, item_ids):
        rating = self.linear_forward(user_ids, item_ids)
        rating = self.sigmoid(rating)
        return rating


class ConstrainedProbabilityMatrixFactorization(ProbabilityMatrixFactorization):
    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension,
        regularization_lambda,
        implicit_feedback,
    ):
        super().__init__(n_users, n_items, latent_dimension, regularization_lambda)

        self.item_rating_effect_weight = _build_weight(n_items, latent_dimension)

        self.implicit_feedback_normalized = implicit_feedback.multiply(
            1 / (implicit_feedback.sum(axis=1) + 1e-8)
        ).tocsr()

    def forward(self, user_ids, item_ids):
        # Need to clone to avoid gradient hook accumulation on same tensor and subsequent memory leak
        item_rating_effect_weight = self.item_rating_effect_weight.clone()

        users_implicit_feedback = scipy_to_torch_sparse(
            self.implicit_feedback_normalized[user_ids.cpu().numpy()],
            device=self.bias.device,
        )
        user_weights_offset_caused_by_their_ratings = (
            users_implicit_feedback @ item_rating_effect_weight
        )
        item_weight = self.item_weight[item_ids]

        rating = super().linear_forward(user_ids, item_ids)
        rating += user_weights_offset_caused_by_their_ratings @ item_weight.T
        rating = self.sigmoid(rating)

        scale_down = self.user_weight.shape[0] / len(user_ids)
        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        weight_decay(item_rating_effect_weight, self.regularization_lambda / scale_down)

        return rating
