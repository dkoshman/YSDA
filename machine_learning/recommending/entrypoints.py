import abc
from typing import Optional, Any, Callable

import pytorch_lightning
import torch

from scipy.sparse import csr_matrix

from my_tools.utils import build_class

from . import losses
from .data import build_recommending_dataloader, SparseDataset, SparseDataModuleMixin
from .interface import RecommenderModuleInterface


class ExtraDataCheckpointingMixin:
    EXTRA_MODEL_DATA = "extra_model_data"

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint[self.EXTRA_MODEL_DATA] = self.model.save()

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        if self.EXTRA_MODEL_DATA not in checkpoint:
            raise ValueError("Malformed checkpoint.")
        self.model.load(checkpoint[self.EXTRA_MODEL_DATA])


class LitRecommenderBase(
    SparseDataModuleMixin,
    ExtraDataCheckpointingMixin,
    pytorch_lightning.LightningModule,
):
    def __init__(
        self,
        model_config: dict,
        datamodule_config: dict = None,
        optimizer_config: dict = None,
        loss_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if "n_users" not in model_config or "n_items" not in model_config:
            n_users, n_items = self.train_explicit.shape
            model_config.update(n_users=n_users, n_items=n_items)
            self.save_hyperparameters()

        self.model: RecommenderModuleInterface = self.build_class(**model_config)
        self.loss: Callable = (
            None if loss_config is None else self.build_class(**loss_config)
        )

    @property
    def module_candidates(self):
        return [losses, torch.optim]

    @property
    def class_candidates(self):
        return []

    def build_class(self, **kwargs):
        return build_class(
            class_candidates=self.class_candidates,
            module_candidates=self.module_candidates,
            **kwargs,
        )

    @property
    def train_explicit(self) -> Optional[csr_matrix]:
        return

    @property
    def val_explicit(self) -> Optional[csr_matrix]:
        return

    @property
    def test_explicit(self) -> Optional[csr_matrix]:
        return

    def build_dataloader(self, **kwargs):
        config = self.hparams["datamodule_config"]
        return build_recommending_dataloader(
            batch_size=config.get("batch_size", 100),
            num_workers=config.get("num_workers", 0),
            persistent_workers=config.get("persistent_workers", False),
            **kwargs,
        )

    def train_dataloader(self):
        return self.build_dataloader(
            dataset=SparseDataset(self.train_explicit),
            sampler_type="user",
            shuffle=True,
        )

    def val_dataloader(self):
        return self.build_dataloader(
            dataset=SparseDataset(self.val_explicit),
            sampler_type="user",
        )

    def test_dataloader(self):
        return self.build_dataloader(
            dataset=SparseDataset(self.test_explicit),
            sampler_type="user",
        )

    def configure_optimizers(self):
        optimizer = self.build_class(
            params=self.parameters(),
            **(self.hparams["optimizer_config"] or {}),
        )
        return optimizer

    def forward(self, **batch):
        return self.model(
            user_ids=batch.get("user_ids"), item_ids=batch.get("item_ids")
        )

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_save_checkpoint(self, checkpoint):
        checkpoint["hyper_parameters"]["model_config"].update(
            n_users=self.model.n_users, n_items=self.model.n_items
        )


class NonGradientRecommenderMixin:
    def on_train_batch_start(self, batch, batch_idx):
        """Skip train dataloader."""
        self.model.fit(explicit_feedback=self.train_explicit)
        self.trainer.should_stop = True
        return -1

    def configure_optimizers(self):
        """Placeholder optimizer."""
        optimizer = torch.optim.Adam(params=[torch.zeros(0)])
        return optimizer
