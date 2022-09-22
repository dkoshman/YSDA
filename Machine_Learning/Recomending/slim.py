import pytorch_lightning as pl
import scipy.sparse
import torch

from torch.utils.data import DataLoader, Dataset

from my_ml_tools.entrypoints import ConfigConstructorBase
from my_ml_tools.lightning import ConvenientCheckpointLogCallback
from my_ml_tools.models import register_regularization_hook
from my_ml_tools.utils import scipy_to_torch_sparse, StoppingMonitor

from utils import (
    RecommenderMixin,
    RecommendingConfigDispenser,
    SparseDataModuleMixin,
    SparseDatasetMixin,
    torch_sparse_slice,
)


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
        l2_coefficient=1.0,
        l1_coefficient=1.0,
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
        # If buffer is initialized with None, it won't be saved in state dict.
        # So upon loading checkpoint, you'll have to manually load this buffer,
        # but there's no way around it yet because of limited sparse support.
        self.register_buffer(name="sparse_weight", tensor=None)

        self._sparse_values = torch.empty(0)
        self._sparse_indices = torch.empty(0, dtype=torch.int32)

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
        if self.sparse_weight is not None:
            raise RuntimeError("Model already finalized.")
        self.sparse_weight = torch.sparse_coo_tensor(
            indices=self._sparse_indices,
            values=self._sparse_values,
            size=(self.n_items, self.n_items),
        ).to_sparse_csr()

    def clip_parameter(self):
        self.dense_weight_slice = torch.nn.parameter.Parameter(
            self.dense_weight_slice.clip(0)
        )

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

    def register_slim_hook(self, dense_weight_slice, item_ids):
        if dense_weight_slice.requires_grad:
            hook = self.SLIMHook(
                parameter=dense_weight_slice,
                fixed_row_id_in_each_col=item_ids,
            )
            dense_weight_slice.register_hook(hook)

    def training_forward(self, user_ids, item_ids):
        dense_weight_slice = self.dense_weight_slice.clone()[user_ids]
        ratings = self.explicit_feedback @ dense_weight_slice
        register_regularization_hook(
            dense_weight_slice, self.l2_coefficient, self.l1_coefficient
        )
        self.register_slim_hook(dense_weight_slice, item_ids)
        return ratings

    def predicting_forward(self, user_ids, item_ids):
        explicit_feedback = torch_sparse_slice(self.explicit_feedback, row_ids=user_ids)
        items_sparse_weight = torch_sparse_slice(self.sparse_weight, col_ids=item_ids)
        ratings = explicit_feedback @ items_sparse_weight
        return ratings

    def forward(self, user_ids, item_ids):
        if self.sparse_weight is None:
            return self.training_forward(user_ids, item_ids)
        else:
            return self.predicting_forward(user_ids, item_ids)


class SLIMDataset(SparseDatasetMixin, Dataset):
    def __init__(
        self,
        explicit_train: scipy.sparse.csr_matrix,
        explicit_val: scipy.sparse.csr_matrix,
    ):
        assert explicit_train.shape == explicit_val.shape

        self.explicit_train = explicit_train
        self.explicit_val = explicit_val

    def __len__(self):
        return self.explicit_train.shape[1]

    def __getitem__(self, item_ids):
        explicit_train_kwargs = self.pack_sparse_slice_into_dict(
            "explicit_train", self.explicit_train, item_ids=item_ids
        )
        explicit_val_kwargs = self.pack_sparse_slice_into_dict(
            "explicit_val", self.explicit_val, item_ids=item_ids
        )

        return dict(
            **explicit_train_kwargs,
            **explicit_val_kwargs,
            item_ids=item_ids,
        )


class SLIMSampler:
    def __init__(self, n_items, batch_size, shuffle=True):
        self.n_items = n_items
        self.batch_size = batch_size
        indices = torch.randperm(n_items) if shuffle else torch.arange(n_items)
        self.batch_indices = torch.split(indices, batch_size)

    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        yield from self.batch_indices


class LitSLIM(SparseDataModuleMixin, RecommenderMixin, pl.LightningModule):
    def __init__(
        self,
        *,
        model_config,
        optimizer_config,
        train_path,
        val_path,
        batch_size=100,
        num_workers=1,
        patience=0,
        min_delta=0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = SLIMDataset(
            self.train_explicit(train_path), self.explicit_feedback(val_path)
        )
        model_config["explicit_feedback"] = self.dataset.explicit_train
        self.model = self.build_model(model_config, model_classes=[SLIM])

        self.stopping_monitor = StoppingMonitor(patience, min_delta)
        self.dataloader_iter = iter(self.dataloader)
        self.current_batch = None
        self.batch_is_fitted = False

    @property
    def dataloader(self):
        sampler = SLIMSampler(
            n_items=len(self.dataset), batch_size=self.hparams["batch_size"]
        )
        dataloader = DataLoader(
            dataset=self.dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["num_workers"] > 1,
        )
        return dataloader

    def train_dataloader(self):
        try:
            self.current_batch = next(self.dataloader_iter)
        except StopIteration:
            self.trainer.should_stop = True
            yield {}

        while not self.batch_is_fitted:
            yield self.current_batch

    def val_dataloader(self):
        class SingleValueIterableThatIsDefinitelyNotAList:
            def __iter__(inner_self):
                yield self.current_batch

        return SingleValueIterableThatIsDefinitelyNotAList()

    def forward(self, **batch):
        return self.model(
            user_ids=batch.get("user_ids", slice(None)),
            item_ids=batch.get("item_ids", slice(None)),
        )

    def loss(self, explicit_ratings, model_ratings):
        loss = ((explicit_ratings.to_dense() - model_ratings) ** 2).sum()
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        if self.batch_is_fitted:
            self.batch_is_fitted = False
            return -1
        if self.model.is_uninitialized():
            self.model.init_dense_weight_slice(item_ids=batch["item_ids"])

    def training_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit_train"], ratings)
        self.log("train_loss", (loss / ratings.numel()) ** 0.5)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.model.clip_parameter()
        # Update parameters for optimization.
        self.trainer.optimizers = [self.configure_optimizers()]

    def validation_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit_val"], ratings)
        self.log("val_loss", (loss / ratings.numel()) ** 0.5)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        loss = validation_step_outputs[0]
        if self.stopping_monitor.is_time_to_stop(loss):
            self.batch_is_fitted = True
            density = self.model.transform_dense_slice_to_sparse(
                item_ids=self.current_batch["item_ids"]
            )
            self.log("density", density)

    def on_fit_end(self):
        self.model.finalize()
        self.trainer.save_checkpoint("slim_finalized.ckpt")
        super().on_fit_end()


class SLIMTrainer(ConfigConstructorBase):
    def lightning_module_candidates(self):
        return [LitSLIM]

    def callback_class_candidates(self):
        return [ConvenientCheckpointLogCallback]

    def main(self):
        self.trainer.fit(self.lightning_module)


@RecommendingConfigDispenser
def main(config):
    SLIMTrainer(config).main()


if __name__ == "__main__":
    main()
