from typing import Optional

import pytorch_lightning
import torch

from scipy.sparse import csr_matrix

from my_tools.utils import build_class

from . import losses
from .data import build_recommending_dataloader, SparseDataset, SparseDataModuleMixin


class ExtraDataCheckpointingMixin:
    EXTRA_MODEL_DATA = "extra_model_data"

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self.model, "save"):
            checkpoint[self.EXTRA_MODEL_DATA] = self.model.save()

    def on_load_checkpoint(self, checkpoint):
        if hasattr(self.model, "load"):
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
        datamodule_config: dict,
        model_config: dict,
        optimizer_config: dict = None,
        loss_config: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.build_model()
        self.loss = self.build_loss()

    def build_class(self, module_candidates=(), **kwargs):
        return build_class(
            module_candidates=list(module_candidates) + [losses, torch.optim],
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

    def build_model(self):
        return self.build_class(**self.hparams["model_config"])

    def build_loss(self):
        if config := self.hparams["loss_config"]:
            loss = self.build_class(**config)
            return loss

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
