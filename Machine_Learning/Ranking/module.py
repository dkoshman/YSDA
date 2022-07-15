import einops
import numpy as np
import pytorch_lightning as pl
import torch

from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader


class DSSMModule(pl.LightningModule):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        vectorizer,
        dssm,
        loss_fn,
        optimizers_config,
        num_workers=4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.vectorizer = vectorizer
        self.dssm = dssm
        self.loss_fn = loss_fn
        self.optimizers_config = optimizers_config
        self.common_dataloader_params = {
            "batch_size": 1,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": True,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, shuffle=True, **self.common_dataloader_params
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.common_dataloader_params)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, **self.common_dataloader_params)

    def configure_optimizers(self):
        optimizer, lr_scheduler = self.optimizers_config(self)
        return optimizer if lr_scheduler is None else ([optimizer], [lr_scheduler])

    def forward(self, batch):
        return self.dssm(batch)

    @torch.inference_mode()
    def predict(self, query: str, documents: list[str]):
        query = torch.tensor(
            self.vectorizer.transform([query]).toarray(),
            dtype=torch.float32,
            device=self.device,
        )
        documents = torch.tensor(
            self.vectorizer.transform(documents).toarray(),
            dtype=torch.float32,
            device=self.device,
        )
        batch = {
            "query": query,
            "documents": einops.rearrange(documents, "n d -> () n d"),
        }
        relevance = self(batch)
        return relevance[0]

    def training_step(self, batch, batch_idx):
        relevance = self(batch)
        loss = self.loss_fn(relevance, batch["relevance"])
        return loss

    def _shared_eval_step(self, batch):
        if batch["relevance"].shape[-1] < 2:
            return np.nan

        relevance = self(batch)
        return ndcg_score(batch["relevance"].cpu(), relevance.cpu())

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch)

    def validation_epoch_end(self, val_step_outputs):
        self.log("val_ndcg", np.nanmean(val_step_outputs))

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch)

    def test_epoch_end(self, test_step_outputs):
        self.log("test_ndcg", np.nanmean(test_step_outputs))
