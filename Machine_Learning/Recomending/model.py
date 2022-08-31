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

    def forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]

        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias
        rating = self.sigmoid(rating)

        # Need to add regularization here because otherwise optimizer will decay all weights,
        # not only those corresponding to user and item ids.
        for parameter in [user_weight, item_weight, user_bias, item_bias]:
            weight_decay(parameter, self.regularization_lambda)

        return rating


class ConstrainedProbabilityMatrixFactorization(torch.nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension,
        regularization_lambda,
        implicit_feedback,
    ):
        super().__init__()

        self.regularization_lambda = regularization_lambda

        self.user_weight = _build_weight(n_users, latent_dimension)
        self.item_weight = _build_weight(n_items, latent_dimension)
        self.item_rating_effect_weight = _build_weight(n_items, latent_dimension)
        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.implicit_feedback = implicit_feedback
        self.implicit_feedback_norm = torch.tensor(implicit_feedback.sum(axis=1))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        item_weight = self.item_weight[item_ids]
        implicit_feedback = scipy_to_torch_sparse(
            self.implicit_feedback[user_ids.cpu().numpy()], device=user_weight.device
        )
        implicit_feedback_norm = self.implicit_feedback_norm[user_ids].to(
            user_weight.device
        )

        user_factors = (
            user_weight
            + implicit_feedback
            @ self.item_rating_effect_weight
            / implicit_feedback_norm
        )
        rating = user_factors @ item_weight.T + self.bias
        rating = self.sigmoid(rating)

        for parameter in [user_weight, item_weight, self.item_rating_effect_weight]:
            weight_decay(parameter, self.regularization_lambda)

        return rating
