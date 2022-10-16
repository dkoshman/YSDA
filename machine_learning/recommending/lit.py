import pytorch_lightning as pl
import torch

from my_tools.utils import BuilderMixin

from . import losses
from .data import build_recommending_dataloader, SparseDataModuleBase
from .interface import (
    RecommenderModuleInterface,
    RecommendingLossInterface,
    InMemoryRecommender,
)


class ExtraDataCheckpointingMixin:
    EXTRA_MODEL_DATA = "extra_model_data"
    model: RecommenderModuleInterface

    def on_save_checkpoint(self: pl.LightningModule, checkpoint: dict) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint[self.EXTRA_MODEL_DATA] = self.model.save()

    def on_load_checkpoint(self: pl.LightningModule, checkpoint: dict) -> None:
        super().on_load_checkpoint(checkpoint)
        if self.EXTRA_MODEL_DATA not in checkpoint:
            raise ValueError("Malformed checkpoint.")
        self.model.load(checkpoint[self.EXTRA_MODEL_DATA])


class LitRecommenderBase(
    SparseDataModuleBase,
    pl.LightningModule,
    BuilderMixin,
    ExtraDataCheckpointingMixin,
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
            self.save_hyperparameters("model_config")

        self.model: RecommenderModuleInterface = self.build_class(**model_config)
        self.loss: RecommendingLossInterface or None = None
        if loss_config is not None:
            self.loss = self.build_class(**loss_config)
        self.recommend = self.model.recommend

    @property
    def module_candidates(self):
        return [losses, torch.optim]

    def build_dataloader(self, **kwargs):
        config = self.hparams["datamodule_config"]
        return build_recommending_dataloader(
            batch_size=config.get("batch_size", 100),
            num_workers=config.get("num_workers", 0),
            persistent_workers=config.get("persistent_workers", False),
            **kwargs,
        )

    def configure_optimizers(self):
        if (config := self.hparams["optimizer_config"]) is None:
            config = dict(name="Adam")
        optimizer = self.build_class(params=self.parameters(), **config)
        return optimizer

    def setup(self, stage=None):
        super().setup(stage=stage)
        if stage == "fit" and isinstance(self.model, InMemoryRecommender):
            self.model.save_explicit_feedback(self.train_explicit)

    def forward(self, **batch):
        if (user_ids := batch.get("user_ids")) is None:
            user_ids = torch.arange(self.model.n_users)
        if (item_ids := batch.get("item_ids")) is None:
            item_ids = torch.arange(self.model.n_items)
        return self.model(user_ids=user_ids, item_ids=item_ids)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        # In case new users/items were added, for proper checkpoint
        # loading parameters need to be initialized in proper shape.
        checkpoint["hyper_parameters"]["model_config"].update(
            n_users=self.model.n_users, n_items=self.model.n_items
        )


class NonGradientRecommenderMixin:
    def on_train_batch_start(self: LitRecommenderBase, batch, batch_idx):
        """Skip train dataloader."""
        self.model.fit(explicit_feedback=self.train_explicit)
        self.trainer.should_stop = True
        return -1

    def configure_optimizers(self: pl.LightningModule):
        """Placeholder optimizer."""
        optimizer = torch.optim.Adam(params=[torch.zeros(0)])
        return optimizer
