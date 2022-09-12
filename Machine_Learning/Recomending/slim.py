import pytorch_lightning as pl
import scipy.sparse
import torch

from torch.utils.data import DataLoader, Dataset

from my_ml_tools.entrypoints import ConfigDispenser, ConfigConstructorBase
from my_ml_tools.lightning import ConvenientCheckpointLogCallback
from my_ml_tools.utils import scipy_to_torch_sparse

from data import SparseDataModuleMixin, SparseDatasetMixin
from lightning import RecommenderMixin
from utils import torch_sparse_slice


class SLIM(torch.nn.Module):
    def __init__(
        self,
        explicit_feedback: scipy.sparse.csr_matrix,
        l2_coefficient=1.0,
        l1_coefficient=1.0,
    ):
        super().__init__()

        self.register_buffer(
            name="explicit_feedback",
            tensor=scipy_to_torch_sparse(explicit_feedback),
        )

        self._dense_weight_slice = torch.nn.parameter.Parameter(data=torch.empty(0))
        self.sparse_weight = None
        self.l2_coefficient = l2_coefficient
        self.l1_coefficient = l1_coefficient
        self.n_items = explicit_feedback.shape[1]

        self._sparse_values = torch.empty(0)
        self._sparse_indices = torch.empty(0, dtype=torch.int32)

    def is_uninitialized(self):
        return self._dense_weight_slice.numel() == 0

    def init_dense_weight_slice(self, item_ids):
        dense_weight_slice = torch.empty(
            self.n_items, len(item_ids), device=self._dense_weight_slice.device
        )
        torch.nn.init.xavier_normal_(dense_weight_slice)
        dense_weight_slice = dense_weight_slice.abs()
        dense_weight_slice[item_ids, torch.arange(len(item_ids))] = 0
        self._dense_weight_slice.data = dense_weight_slice

    def transform_dense_slice_to_sparse(self):
        sparse = self._dense_weight_slice.cpu().detach().to_sparse_coo()
        self._sparse_values = torch.cat([self._sparse_values, sparse.values()])
        self._sparse_indices = torch.cat([self._sparse_indices, sparse.indices()], 1)

        self._dense_weight_slice.data = torch.empty(
            0, device=self._dense_weight_slice.device
        )
        density = len(sparse.values()) / sparse.numel()
        return density

    def finalize(self):
        assert self.sparse_weight is None, "Model already finalized."
        sparse_weight = torch.sparse_coo_tensor(
            indices=self._sparse_indices,
            values=self._sparse_values,
            size=(self.n_items, self.n_items),
        ).to_sparse_csr()
        del self.sparse_weight
        self.register_buffer(name="sparse_weight", tensor=sparse_weight)

    def get_dense_weight_slice(self, user_ids):
        self._dense_weight_slice.data = torch.clip(self._dense_weight_slice, 0)
        return self._dense_weight_slice.clone()[user_ids]

    def add_regularization_hook(self, dense_weight_slice, item_ids):
        if not dense_weight_slice.requires_grad:
            return dense_weight_slice

        hook = SLIMRegularizationGradientHook(
            parameter=dense_weight_slice,
            l2_coefficient=self.l2_coefficient,
            l1_coefficient=self.l1_coefficient,
            fixed_row_id_in_each_col=item_ids,
        )
        dense_weight_slice.register_hook(hook)
        return dense_weight_slice

    def training_forward(self, user_ids, item_ids):
        dense_weight_slice = self.get_dense_weight_slice(user_ids)
        ratings = self.explicit_feedback @ dense_weight_slice
        self.add_regularization_hook(dense_weight_slice, item_ids)
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


class SLIMRegularizationGradientHook:
    def __init__(
        self, parameter, l2_coefficient, l1_coefficient, fixed_row_id_in_each_col
    ):
        self.fixed_row_id_in_each_col = fixed_row_id_in_each_col
        self.parameter = parameter.clone().detach()
        self.regularization = (
            l2_coefficient * self.parameter + l1_coefficient * self.parameter.sign()
        )

    def soft_positive_regularization(self, grad):
        grad[(self.parameter == 0) & (0 < grad)] = 0

    def zero_parameter_diagonal_preservation(self, grad):
        grad[
            self.fixed_row_id_in_each_col,
            torch.arange(len(self.fixed_row_id_in_each_col)),
        ] = 0

    def __call__(self, grad):
        grad = grad.clone().detach() + self.regularization
        self.soft_positive_regularization(grad)
        self.zero_parameter_diagonal_preservation(grad)
        return grad


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


class StoppingMonitor:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.impatience = 0
        self.min_delta = min_delta
        self.lowest_loss = torch.inf

    def is_time_to_stop(self, loss):
        if loss < self.lowest_loss - self.min_delta:
            self.lowest_loss = loss
            self.impatience = 0
            return False
        self.impatience += 1
        if self.impatience > self.patience:
            self.impatience = 0
            self.lowest_loss = torch.inf
            return True
        return False


class LitSLIM(SparseDataModuleMixin, RecommenderMixin, pl.LightningModule):
    def __init__(
        self,
        *,
        model_config,
        optimizer_config,
        batch_size=100,
        num_workers=1,
        patience=0,
        min_delta=0
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset = SLIMDataset(self.train_explicit, self.val_explicit)
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

        self.batch_is_fitted = False
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
        if self.model.is_uninitialized():
            self.model.init_dense_weight_slice(item_ids=batch["item_ids"])

    def training_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit_train"], ratings)
        self.log("train_loss", (loss / ratings.numel()) ** 0.5)
        return loss

    def validation_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit_val"], ratings)
        self.log("val_loss", (loss / ratings.numel()) ** 0.5)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        loss = validation_step_outputs[0]
        if self.stopping_monitor.is_time_to_stop(loss):
            self.batch_is_fitted = True
            density = self.model.transform_dense_slice_to_sparse()
            self.log("density", density)

    def on_fit_end(self):
        self.model.finalize()


class SLIMTrainer(ConfigConstructorBase):
    def lightning_module_candidates(self):
        return [LitSLIM]

    def callback_class_candidates(self):
        return [ConvenientCheckpointLogCallback]

    def main(self):
        self.trainer.fit(self.lightning_module)


@ConfigDispenser
def main(config):
    SLIMTrainer(config).main()


if __name__ == "__main__":
    # TODO: Memory leak
    main()
