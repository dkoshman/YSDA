from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch

from my_tools.utils import BuilderMixin

from . import losses, models
from .data import build_recommending_dataloader, SparseDataModuleBase
from .utils import prepare_artifacts_from_config

if TYPE_CHECKING:
    from .interface import RecommenderModuleBase, RecommendingLossInterface


class LitRecommenderBase(SparseDataModuleBase, pl.LightningModule, BuilderMixin):
    def __init__(self, n_users=None, n_items=None, **config):
        super().__init__()
        self.save_hyperparameters()
        prepare_artifacts_from_config(config=config)
        if n_users is None or n_items is None:
            n_users, n_items = self.train_explicit().shape
            self.save_hyperparameters()  # This will update n_users, n_items.
        self.model: "RecommenderModuleBase" = self.build_model()
        self.loss: "RecommendingLossInterface" or None = None
        if "loss" in config:
            self.loss = self.build_class(
                explicit=self.train_explicit(), **config["loss"]
            )

    def build_model(self):
        model_config = self.hparams["model"]
        return self.build_class(
            explicit=self.train_explicit(),
            n_users=self.hparams["n_users"],
            n_items=self.hparams["n_items"],
            **model_config,
        )

    @property
    def module_candidates(self):
        return [models, losses, torch.optim, torch.optim.lr_scheduler]

    @property
    def class_candidates(self):
        return [LRLambda]

    def build_dataloader(self, **kwargs):
        config = self.hparams["datamodule"]
        return build_recommending_dataloader(
            batch_size=config.get("batch_size", 100),
            num_workers=config.get("num_workers", 0),
            persistent_workers=config.get("persistent_workers", False),
            **kwargs,
        )

    def configure_optimizers(self):
        optimizer_config = self.hparams["optimizer"].copy()
        optimizer = self.build_class(params=self.parameters(), **optimizer_config)
        if (lr_scheduler_config := self.hparams.get("lr_scheduler")) is None:
            return optimizer

        if lr_scheduler_config["class_name"] == "LambdaLR":
            lr_scheduler_config = lr_scheduler_config.copy()
            lr_lambda_config = lr_scheduler_config.pop["lr_lambda"]
            lr_scheduler_config["lr_lambda"] = self.build_class(**lr_lambda_config)

        lr_scheduler = self.build_class(optimizer=optimizer, **lr_scheduler_config)
        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def forward(self, **batch):
        return self.model(user_ids=batch["user_id"], item_ids=batch["item_id"])

    """Step placeholders to enable hooks without error."""

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


class LRLambda:
    """Learning rate scheduler with linear warmup and subsequent reverse square root fading."""

    def __init__(self, warmup_epochs=20):
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        epoch += 1
        learning_rate_multiplier = min(
            (self.warmup_epochs / epoch) ** 0.5, epoch / self.warmup_epochs
        )
        return learning_rate_multiplier


class NonGradientRecommenderMixin:
    def on_train_batch_start(self: LitRecommenderBase, batch, batch_idx):
        if self.current_epoch == 0:
            if batch_idx == 0:
                self.model.fit()
        else:
            self.trainer.should_stop = True
            return -1

    def configure_optimizers(self: pl.LightningModule):
        return
