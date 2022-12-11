import warnings

import numpy as np
import scipy.sparse
import shap
import torch

from my_tools.models import (
    register_regularization_hook,
    StoppingMonitor,
    WandbLoggerMixin,
)
from my_tools.utils import torch_sparse_slice

from ..data import SparseDataset
from ..lit import LitRecommenderBase
from ..interface import RecommenderModuleBase, ExplanationMixin
from ..utils import plt_figure, wandb_plt_figure


class SLIM(RecommenderModuleBase, WandbLoggerMixin, ExplanationMixin):
    """
    The fitted model has sparse matrix W of shape [n_items, n_items],
    which it uses to predict ratings for users with feedback matrix
    A of shape [n_users, n_items] by calculating AW. It performs well,
    it's parameters are interpretable, and it performs inference quickly
    due to sparsity. It may be beneficial to add biases to this implementation.
    """

    def __init__(self, l2_coefficient=1.0e-4, l1_coefficient=1.0e-4, **kwargs):
        super().__init__(**kwargs, persistent_explicit=False)
        self.l2_coefficient = l2_coefficient
        self.l1_coefficient = l1_coefficient

        self.register_buffer(
            name="sparse_weight",
            tensor=torch.sparse_coo_tensor(size=(self.n_items, self.n_items)),
        )
        self.item_bias = torch.nn.Parameter(data=torch.zeros(self.n_items))
        self.dense_weight_slice = torch.nn.parameter.Parameter(torch.empty(0))

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

        density = len(sparse.values()) / sparse.numel()
        self.log({"Items similarity matrix density": density})

    def finalize_training(self):
        self.sparse_weight = torch.sparse_coo_tensor(
            indices=self._sparse_indices,
            values=self._sparse_values,
            size=(self.n_items, self.n_items),
        )

    def clip_negative_parameter_values(self):
        self.dense_weight_slice = torch.nn.parameter.Parameter(
            self.dense_weight_slice.clip(0)
        )

    def training_forward(self, item_ids):
        assert len(item_ids) == self.dense_weight_slice.shape[1]
        dense_weight_slice = self.dense_weight_slice.clone()
        ratings = self.explicit.to(torch.float32) @ dense_weight_slice
        ratings = ratings.to_dense() + self.item_bias[item_ids]
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

    def forward(self, user_ids, item_ids):
        explicit = torch_sparse_slice(self.explicit, row_ids=user_ids).to(self.device)
        items_sparse_weight = torch_sparse_slice(
            self.sparse_weight, col_ids=item_ids
        ).to(self.device)
        ratings = explicit.to(torch.float32) @ items_sparse_weight
        ratings = ratings.to_dense() + self.item_bias[item_ids]
        return ratings

    def online_ratings(self, users_explicit):
        users_explicit = self.to_torch_coo(users_explicit)
        ratings = users_explicit.to(torch.float32).to(self.device) @ self.sparse_weight
        ratings = ratings.to_dense() + self.item_bias
        return ratings

    def explain_recommendations_for_user(
        self,
        user_id=None,
        user_explicit=None,
        n_recommendations=10,
        log=False,
        logging_prefix="",
        feature_names=None,
        n_background_samples_for_shap=1000,
    ):
        if feature_names is not None and len(feature_names) != self.n_items:
            raise ValueError(f"Feature names must be of length {self.n_items}")

        if user_id is not None:
            recommendations = self.recommend(
                user_ids=torch.IntTensor([user_id]), n_recommendations=n_recommendations
            )
            user_explicit = self.to_scipy_coo(self.explicit).tocsr()[user_id].toarray()
        else:
            recommendations = self.online_recommend(
                users_explicit=user_explicit, n_recommendations=n_recommendations
            )
            user_explicit = self.to_scipy_coo(user_explicit).toarray()

        background_samples_for_shap = self.to_scipy_coo(self.explicit).tocsr()[
            np.random.choice(
                np.arange(self.n_users),
                replace=False,
                size=min(n_background_samples_for_shap, self.n_users),
            )
        ]

        figures = []
        figure_context_manager = wandb_plt_figure if log else plt_figure
        for item_id in recommendations.squeeze(0).cpu().numpy():
            item_weight = torch_sparse_slice(self.sparse_weight, col_ids=[item_id])
            item_weight = item_weight.to_dense().numpy().squeeze(1)
            coefficient = self.item_bias[item_id].item()
            explainer = shap.explainers.Linear(
                model=(item_weight, coefficient),
                masker=background_samples_for_shap,
                feature_names=feature_names,
            )
            shap_values = explainer(user_explicit)
            title = (
                logging_prefix
                + f" Item {item_id if feature_names is None else feature_names[item_id]}"
            )
            with figure_context_manager(title=title) as figure:
                shap.waterfall_plot(shap_values[0])
            figures.append(figure)

        return figures


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
        explicit: scipy.sparse.csr_matrix,
        explicit_val: scipy.sparse.csr_matrix,
    ):
        super().__init__(explicit)
        self.explicit_val = explicit_val

    def __len__(self):
        return self.explicit.shape[1]

    def __getitem__(self, indices):
        item = super().__getitem__(indices)
        explicit_val = torch_sparse_slice(
            self.explicit_val, item["user_id"], item["item_id"]
        )
        item.update(explicit_val=self.pack_sparse_tensor(explicit_val))
        return item


class SLIMRecommender(LitRecommenderBase):
    def __init__(
        self,
        patience=0,
        min_delta=0,
        check_val_every_n_epoch=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=list(kwargs.keys()))
        self.stopping_monitor = None
        self.current_batch = None
        self.batch_is_fitted = False
        self.item_dataloader_iter = None

    @property
    def class_candidates(self):
        return super().class_candidates + [SLIM]

    def setup(self, stage=None):
        super().setup(stage)
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
            slim_dataset = SLIMDataset(self.train_explicit(), self.val_explicit())
            item_dataloader = self.build_dataloader(
                dataset=slim_dataset, sampler_type="item"
            )
            self.item_dataloader_iter = iter(item_dataloader)

    def train_dataloader(self):
        try:
            self.current_batch = next(self.item_dataloader_iter)
        except StopIteration:
            self.trainer.should_stop = True
            yield self.current_batch

        while not self.batch_is_fitted:
            yield self.current_batch

    def on_train_batch_start(self, batch, batch_idx):
        if self.batch_is_fitted:
            self.batch_is_fitted = False
            return -1
        if self.model.is_uninitialized():
            self.model.init_dense_weight_slice(item_ids=batch["item_id"])

    def training_step(self, batch, batch_idx):
        ratings = self.model.training_forward(item_ids=batch["item_id"])
        train_loss = self.loss(explicit=batch["explicit"], model_ratings=ratings)
        self.log("train_loss", train_loss)

        if batch_idx % self.hparams["check_val_every_n_epoch"] == 0:
            val_loss = self.loss(explicit=batch["explicit_val"], model_ratings=ratings)
            self.log("val_loss", val_loss)

            if self.stopping_monitor.is_time_to_stop(val_loss):
                self.batch_is_fitted = True
                self.model.transform_dense_slice_to_sparse(
                    item_ids=self.current_batch["item_id"]
                )

        return train_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.clip_negative_parameter_values()
        # Update parameters passed to optimizer.
        self.trainer.optimizers = [self.configure_optimizers()]

    def on_train_end(self):
        self.model.finalize_training()
        self.trainer.validate(self)
        super().on_train_end()
