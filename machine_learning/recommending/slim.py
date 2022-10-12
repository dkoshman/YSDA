import warnings

import scipy.sparse
import torch

from my_tools.models import register_regularization_hook
from my_tools.utils import scipy_to_torch_sparse, StoppingMonitor

from .data import SparseDataset
from .entrypoints import LitRecommenderBase
from .utils import torch_sparse_slice


class SLIM(torch.nn.Module):
    """
    The fitted model has sparse matrix W of shape [n_items, n_items],
    which it uses to predict ratings for users with feedback matrix
    A of shape [n_users, n_items] by calculating AW. It performs well,
    it's parameters are interpretable, and it performs inference quickly
    due to sparsity. It may be beneficial to add biases to this implementation.
    """

    def __init__(
        self,
        explicit_feedback: scipy.sparse.csr_matrix,
        l2_coefficient=1.0e-4,
        l1_coefficient=1.0e-4,
    ):
        super().__init__()

        self.l2_coefficient = l2_coefficient
        self.l1_coefficient = l1_coefficient
        self.n_items = explicit_feedback.shape[1]

        self.register_buffer(
            name="explicit_feedback",
            tensor=scipy_to_torch_sparse(explicit_feedback),
        )
        self.dense_weight_slice = torch.nn.parameter.Parameter(torch.empty(0))
        self.register_buffer(
            name="sparse_weight",
            tensor=torch.sparse_coo_tensor(size=(self.n_items, self.n_items)),
        )

        self._sparse_values = torch.empty(0)
        self._sparse_indices = torch.empty(2, 0, dtype=torch.int32)

    def is_uninitialized(self):
        return self.dense_weight_slice.numel() == 0

    def init_dense_weight_slice(self, item_ids):
        dense_weight_slice = torch.empty(
            self.n_items, len(item_ids), device=self.dense_weight_slice.device
        )
        torch.nn.init.xavier_normal_(dense_weight_slice)
        dense_weight_slice = dense_weight_slice.abs()
        dense_weight_slice[item_ids, torch.arange(len(item_ids))] = 0
        self.dense_weight_slice = torch.nn.parameter.Parameter(dense_weight_slice)

    def transform_dense_slice_to_sparse(self, item_ids):
        sparse = self.dense_weight_slice.cpu().detach().to_sparse_coo()
        self._sparse_values = torch.cat([self._sparse_values, sparse.values()])
        rows, cols = sparse.indices()
        cols = item_ids[cols]
        indices = torch.stack([rows, cols])
        self._sparse_indices = torch.cat([self._sparse_indices, indices], dim=1)

        self.dense_weight_slice = torch.nn.parameter.Parameter(
            torch.empty(0, device=self.dense_weight_slice.device)
        )

        # Return density for logging.
        density = len(sparse.values()) / sparse.numel()
        return density

    def finalize(self):
        self.sparse_weight = torch.sparse_coo_tensor(
            indices=self._sparse_indices,
            values=self._sparse_values,
            size=(self.n_items, self.n_items),
        )

    def clip_parameter(self):
        self.dense_weight_slice = torch.nn.parameter.Parameter(
            self.dense_weight_slice.clip(0)
        )

    def training_forward(self, item_ids):
        assert len(item_ids) == self.dense_weight_slice.shape[1]
        dense_weight_slice = self.dense_weight_slice.clone()
        ratings = self.explicit_feedback.to(torch.float32) @ dense_weight_slice
        register_regularization_hook(
            dense_weight_slice, self.l2_coefficient, self.l1_coefficient
        )
        if dense_weight_slice.requires_grad:
            hook = SLIMHook(
                parameter=dense_weight_slice,
                fixed_row_id_in_each_col=item_ids,
            )
            dense_weight_slice.register_hook(hook)
        return ratings

    def forward(self, user_ids=None, item_ids=None):
        if user_ids is None:
            user_ids = slice(None)

        if item_ids is None:
            item_ids = slice(None)

        explicit_feedback = torch_sparse_slice(self.explicit_feedback, row_ids=user_ids)
        items_sparse_weight = torch_sparse_slice(self.sparse_weight, col_ids=item_ids)
        ratings = explicit_feedback.to(torch.float32) @ items_sparse_weight
        return ratings


class SLIMHook:
    """
    Gradient hook intended to clip gradient to enforce the positive
    parameter values and zero diagonal regularization restrictions.
    It would be easier to clip values after optimizer step, but then
    it would mess up optimizer's running gradient mean estimation
    and parameter weighting. We still need to clip parameters after
    the step though as the gradient step may still make positive
    parameters negative.
    """

    def __init__(self, parameter, fixed_row_id_in_each_col):
        self.fixed_row_id_in_each_col = fixed_row_id_in_each_col
        self.parameter = parameter.clone().detach()

    def soft_positive_regularization(self, grad):
        grad[(self.parameter == 0) & (0 < grad)] = 0

    def zero_parameter_diagonal_preservation(self, grad):
        grad[
            self.fixed_row_id_in_each_col,
            torch.arange(len(self.fixed_row_id_in_each_col)),
        ] = 0

    def __call__(self, grad):
        grad = grad.clone().detach()
        self.soft_positive_regularization(grad)
        self.zero_parameter_diagonal_preservation(grad)
        return grad


class SLIMDataset(SparseDataset):
    def __init__(
        self,
        explicit_feedback: scipy.sparse.csr_matrix,
        explicit_feedback_val: scipy.sparse.csr_matrix,
    ):
        super().__init__(explicit_feedback)
        self.explicit_feedback_val = explicit_feedback_val

    def __len__(self):
        return self.explicit_feedback.shape[1]

    def __getitem__(self, indices):
        item = super().__getitem__(indices)
        explicit_val = torch_sparse_slice(
            self.explicit_feedback_val, item["user_ids"], item["item_ids"]
        )
        item.update(explicit_val=self.pack_sparse_tensor(explicit_val))
        return item


class SLIMRecommender(LitRecommenderBase):
    def __init__(
        self,
        patience=0,
        min_delta=0,
        check_val_every_n_epoch=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=kwargs.keys())
        self.stopping_monitor = None
        self.current_batch = None
        self.batch_is_fitted = False
        self.item_dataloader_iter = None

    def setup(self, stage=None):
        if stage == "fit":
            if self.trainer.limit_val_batches != 0:
                warnings.warn(
                    "In this implementation when training model is iterating over items, "
                    "and the user-wise metrics cannot be calculated while training. "
                    "So validation should be manually disabled, and the model will validate "
                    "when it is fitted."
                )
            self.stopping_monitor = StoppingMonitor(
                self.hparams["patience"], self.hparams["min_delta"]
            )
            self.current_batch = None
            self.batch_is_fitted = False
            slim_dataset = SLIMDataset(self.train_explicit, self.val_explicit)
            item_dataloader = self.build_dataloader(
                dataset=slim_dataset, sampler_type="item"
            )
            self.item_dataloader_iter = iter(item_dataloader)
        super().setup(stage)

    def build_model(self):
        return self.build_class(
            class_candidates=[SLIM],
            explicit_feedback=self.train_explicit,
            **self.hparams["model_config"],
        )

    def train_dataloader(self):
        try:
            self.current_batch = next(self.item_dataloader_iter)
        except StopIteration:
            self.trainer.should_stop = True
            yield {}

        while not self.batch_is_fitted:
            yield self.current_batch

    def on_train_batch_start(self, batch, batch_idx):
        if self.batch_is_fitted:
            self.batch_is_fitted = False
            return -1
        if self.model.is_uninitialized():
            self.model.init_dense_weight_slice(item_ids=batch["item_ids"])

    def training_step(self, batch, batch_idx):
        ratings = self.model.training_forward(item_ids=batch["item_ids"])
        train_loss = self.loss(explicit=batch["explicit"], model_ratings=ratings)
        self.log("train_loss", train_loss)

        if batch_idx % self.hparams["check_val_every_n_epoch"] == 0:
            val_loss = self.loss(explicit=batch["explicit_val"], model_ratings=ratings)
            self.log("val_loss", val_loss)

            if self.stopping_monitor.is_time_to_stop(val_loss):
                self.batch_is_fitted = True
                density = self.model.transform_dense_slice_to_sparse(
                    item_ids=self.current_batch["item_ids"]
                )
                self.log("Items similarity matrix density", density)

        return train_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.clip_parameter()
        # Update parameters passed to optimizer.
        self.trainer.optimizers = [self.configure_optimizers()]

    def on_train_end(self):
        self.model.finalize()
        self.trainer.validate(self)
        super().on_train_end()
