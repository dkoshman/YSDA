import pytorch_lightning as pl
import torch

from my_tools.utils import BuilderMixin, SparseTensor

from . import losses
from .data import build_recommending_dataloader, SparseDataModuleBase
from .interface import RecommenderModuleBase


class LitRecommenderBase(
    SparseDataModuleBase,
    pl.LightningModule,
    BuilderMixin,
):
    def __init__(
        self,
        model_config: dict,
        datamodule_config: dict = None,
        optimizer_config: dict = None,
        loss_config: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model: RecommenderModuleBase = self.build_class(
            explicit=self.train_explicit(), **model_config
        )

    def loss(
        self, explicit: SparseTensor, model_ratings: torch.FloatTensor or SparseTensor
    ) -> torch.FloatTensor:
        explicit = explicit.to_dense()
        model_ratings = model_ratings.to_dense()
        loss = ((explicit - model_ratings) ** 2).mean()
        return loss

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
            config = dict(class_name="Adam")
        optimizer = self.build_class(params=self.parameters(), **config)
        return optimizer

    def forward(self, **batch):
        return self.model(user_ids=batch["user_ids"], item_ids=batch["item_ids"])

    """Step placeholders to enable hooks without error."""

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


class NonGradientRecommenderMixin:
    def on_train_batch_start(self: LitRecommenderBase, batch, batch_idx):
        if self.current_epoch == 0:
            if batch_idx == 0:
                self.model.fit()
        else:
            self.trainer.should_stop = True
            return -1

    def configure_optimizers(self: pl.LightningModule):
        """Placeholder optimizer."""
        optimizer = torch.optim.Adam(params=[torch.zeros(0)])
        return optimizer
