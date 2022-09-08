import scipy.sparse
import torch

from my_ml_tools.models import weight_decay
from my_ml_tools.utils import scipy_to_torch_sparse

from utils import torch_sparse_slice


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

        item_weight = self.item_weight[item_ids]

        users_implicit_feedback = torch_sparse_slice(
            self.implicit_feedback_normalized, row_ids=user_ids, device=self.bias.device
        )
        user_weights_offset_caused_by_their_ratings = (
            users_implicit_feedback @ item_rating_effect_weight
        )

        rating = super().linear_forward(user_ids, item_ids)
        rating += user_weights_offset_caused_by_their_ratings @ item_weight.T
        rating = self.sigmoid(rating)

        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        scale_down = self.user_weight.shape[0] / len(user_ids)
        weight_decay(item_rating_effect_weight, self.regularization_lambda / scale_down)

        return rating


class SLIMRegularizationGradientHook:
    def __init__(
        self, parameter, l2_coefficient, l1_coefficient, fixed_row_id_in_each_col
    ):
        self.fixed_row_id_in_each_col = fixed_row_id_in_each_col
        self.parameter_sign = parameter.sign()
        self.regularization = (
            l2_coefficient * parameter.detach().clone()
            + l1_coefficient * self.parameter_sign
        )

    def __call__(self, grad):
        new_grad = grad.clone().detach() + self.regularization
        # parameter_sign >= 0 soft regularization
        new_grad[(self.parameter_sign < 0) & (0 < new_grad)] = 0
        # diag(W) = 0 preservation, assuming it's already true
        new_grad[
            self.fixed_row_id_in_each_col,
            torch.arange(len(self.fixed_row_id_in_each_col)),
        ] = 0
        return new_grad


class SLIM(torch.nn.Module):
    def __init__(
        self,
        explicit_feedback: scipy.sparse.csr.csr_matrix,
        l2_coefficient=1.0,
        l1_coefficient=1.0,
    ):
        super().__init__()

        self.register_buffer(
            name="explicit_feedback",
            tensor=scipy_to_torch_sparse(explicit_feedback),
        )
        self.register_buffer(name="_sparse_weight", tensor=torch.empty(0))
        self.dense_weight_slice = torch.nn.parameter.Parameter(data=torch.empty(0))

        self.l2_coefficient = l2_coefficient
        self.l1_coefficient = l1_coefficient
        self.n_items = explicit_feedback.shape[1]

        self.current_item_ids = None
        self._sparse_values = torch.empty(0)
        self._sparse_indices = torch.empty(0, dtype=torch.int32)

    def init_dense_weight_slice(self, item_ids):
        if item_ids is None:
            return
        dense_weight_slice = torch.empty(self.n_items, len(item_ids))
        torch.nn.init.xavier_normal_(dense_weight_slice)
        dense_weight_slice = dense_weight_slice.abs()
        dense_weight_slice[item_ids, torch.arange(len(item_ids))] = 0
        self.dense_weight_slice.data = dense_weight_slice

    def update_current_item_ids(self, item_ids):
        if (
            torch.is_tensor(item_ids)
            and torch.is_tensor(self.current_item_ids)
            and (item_ids == self.current_item_ids).all()
        ):
            return
        self.transform_dense_slice_to_sparse()
        self.init_dense_weight_slice(item_ids)
        self.current_item_ids = item_ids

    def transform_dense_slice_to_sparse(self):
        if self.current_item_ids is None:
            return
        # TODO: to device, bias, this
        # torch.clip(self._dense_weight_slice, 0)
        # threshold = dense.quantile(1 - self.density)
        # dense[dense < threshold] = 0
        sparse = torch.clip(self._dense_weight_slice, 0).cpu().detach().to_sparse_coo()
        self._sparse_values = torch.cat([self._sparse_values, sparse.values()])
        self._sparse_indices = torch.cat([self._sparse_indices, sparse.indices()], 1)
        print(f"Density {len(sparse.values()) / torch.prod(sparse.shape).item()}")
        self.dense_weight_slice.data = torch.empty(0)
        self.current_item_ids = None

    @property
    def sparse_weight(self):
        if len(self._sparse_weight):
            return self._sparse_weight

        self.update_current_item_ids(None)

        self._sparse_weight = torch.sparse_coo_tensor(
            indices=self._sparse_indices,
            values=self._sparse_values,
            size=(self.n_items, self.n_items),
        ).to_sparse_csr()

        return self._sparse_weight

    def add_regularization_hook(self, dense_weight_slice, item_ids):
        hook = SLIMRegularizationGradientHook(
            parameter=dense_weight_slice,
            l2_coefficient=self.l2_coefficient,
            l1_coefficient=self.l1_coefficient,
            fixed_row_id_in_each_col=item_ids,
        )
        dense_weight_slice.register_hook(hook)
        return dense_weight_slice

    def forward(self, user_ids=None, item_ids=None):
        # it may still not be ready even if not training
        if user_ids is not None:
            users_explicit_feedback = torch_sparse_slice(
                self.explicit_feedback, row_ids=user_ids
            )
            items_sparse_weight = torch_sparse_slice(
                self.sparse_weight, col_ids=item_ids
            )
            ratings = users_explicit_feedback @ items_sparse_weight
            return ratings

        self.update_current_item_ids(item_ids)
        ratings = self.explicit_feedback @ self.dense_weight_slice
        self.add_regularization_hook(self.dense_weight_slice, item_ids)

        return ratings
