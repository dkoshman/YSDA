import pytorch_lightning as pl
import torch

from my_ml_tools.utils import sparse_dense_multiply


class LitRecommender(pl.LightningModule):
    def loss_fn(self, batch, model_ratings) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, batch):
        return self.model(user_ids=batch["user_ids"], item_ids=batch["item_ids"])

    def training_step(self, batch, batch_idx):
        ratings = self(batch)
        loss = self.loss_fn(batch, ratings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ratings = self(batch)
        loss = self.loss_fn(batch, ratings)
        self.log("val_loss", loss)
        return loss

    @torch.inference_mode()
    def recommend(self, user_ids):
        batch = {"user_ids": user_ids, "item_ids": slice(None)}
        recommendations = self(batch)
        return recommendations


class LitProbabilityMatrixFactorization(LitRecommender):
    def __init__(self, model, optimizer_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model

    @staticmethod
    def sparse_pmf_loss(explicit, implicit, model_ratings):
        """Loss = 1/|Explicit| * \sum_{ij}Implicit_{ij} * (Explicit_{ij} - ModelRating_{ij})^2"""

        error = model_ratings - explicit
        error = sparse_dense_multiply(sparse_dense_multiply(implicit, error), error)
        loss = torch.sparse.sum(error) / error._values().numel()
        return loss

    def loss_fn(self, batch, model_ratings):
        return self.sparse_pmf_loss(batch["explicit"], batch["implicit"], model_ratings)

    def configure_optimizers(self):
        config = self.hparams["optimizer_kwargs"]

        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(params=self.parameters())

        elif config["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=config["sgd"]["learning_rate"],
                momentum=config["sgd"]["momentum"],
            )
        else:
            raise ValueError("Unknown optimizer config.")

        return optimizer


class LitSLIM(LitRecommender):
    def __init__(self, *, model, patience):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.patience_threshold = patience
        self.lowest_batch_loss = torch.inf
        self.patience = 0

    def loss_fn(self, batch, model_ratings):
        loss = ((batch["explicit_val"].to_dense() - model_ratings) ** 2).mean()
        return loss

    def validation_step(self, batch, batch_idx):
        batch = self.trainer.datamodule.current_batch
        loss = super().validation_step(batch, batch_idx)
        if loss < self.lowest_batch_loss:
            self.lowest_batch_loss = loss
            self.patience = 0
        else:
            self.patience += 1
            if self.patience > self.patience_threshold:
                self.trainer.datamodule.batch_is_fitted = True
                self.patience = 0
                self.lowest_batch_loss = torch.inf
                # self.trainer.optimizers = [self.configure_optimizers()]

    def on_fit_end(self):
        print(self.recommend(user_ids=torch.arange(100)))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters())
        return optimizer
