import numpy as np
import pytorch_lightning as pl
import torch

from model import ProbabilityMatrixFactorization

from my_ml_tools.utils import sparse_dense_multiply


class LitPMF(pl.LightningModule):
    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension=10,
        weight_decay=1e-3,
        learning_rate=5e-3,
        momentum=0.9,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ProbabilityMatrixFactorization(
            n_users=n_users,
            n_items=n_items,
            latent_dimension=latent_dimension,
            regularization_lambda=weight_decay,
        )

    def forward(self, batch):
        return self.model(user_ids=batch["user_ids"], item_ids=batch["item_ids"])

    @staticmethod
    def sparse_pmf_loss(explicit, implicit, model_ratings):
        """Loss = 1/|Explicit| * \sum_{ij}Implicit_{ij} * (Explicit_{ij} - ModelRating_{ij})^2"""

        error = model_ratings - explicit
        error = sparse_dense_multiply(sparse_dense_multiply(implicit, error), error)
        loss = torch.sparse.sum(error) / error._values().numel()
        return loss

    def loss_fn(self, batch, ratings):
        return self.sparse_pmf_loss(batch["explicit"], batch["implicit"], ratings)

    def training_step(self, batch, batch_idx):
        ratings = self(batch)
        loss = self.loss_fn(batch, ratings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ratings = self(batch)
        loss = self.loss_fn(batch, ratings)
        self.log("val_loss", loss)

    @torch.inference_mode()
    def recommend(self, user_ids):
        batch = {"user_ids": user_ids, "item_ids": np.arange(self.hparams["n_items"])}
        recommendations = self(batch)
        return recommendations

    def configure_optimizers(self):
        #         optimizer = torch.optim.SGD(params=self.parameters(), **self.optim_kwargs)
        optimizer = torch.optim.Adam(params=self.parameters())
        return optimizer
