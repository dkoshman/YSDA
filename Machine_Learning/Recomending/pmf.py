import torch

from data import MovieLensDataModule, SparseDataset
from entrypoints import RecommenderBase, MovielensDispatcher

from my_tools.entrypoints import ConfigDispenser
from my_tools.models import register_regularization_hook
from my_tools.utils import build_class
from utils import build_bias, build_weight, torch_sparse_slice


class ProbabilityMatrixFactorization(torch.nn.Module):
    """Predicted_rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension=10,
        weight_decay=1.0e-3,
    ):
        super().__init__()
        self.weight_decay = weight_decay
        self.user_weight = build_weight(n_users, latent_dimension)
        self.user_bias = build_bias(n_users, 1)
        self.item_weight = build_weight(n_items, latent_dimension)
        self.item_bias = build_bias(n_items, 1)
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user_ids=None, item_ids=None):
        if user_ids is None:
            user_ids = slice(None)

        if item_ids is None:
            item_ids = slice(None)

        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]
        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias

        # Need to add regularization here because otherwise optimizer will decay all weights,
        # not only those corresponding to user and item ids. Also, it is important to add gradient
        # hooks after the forward calculations, otherwise decay messes with the model.
        for parameter in [user_weight, item_weight, user_bias, item_bias]:
            register_regularization_hook(parameter, self.weight_decay)

        return rating


class ConstrainedProbabilityMatrixFactorization(ProbabilityMatrixFactorization):
    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension,
        weight_decay,
        implicit_feedback,
    ):
        super().__init__(n_users, n_items, latent_dimension, weight_decay)
        self.item_rating_effect_weight = build_weight(n_items, latent_dimension)
        self.implicit_feedback_normalized = implicit_feedback.multiply(
            1 / (implicit_feedback.sum(axis=1) + 1e-8)
        ).tocsr()

    def forward(self, user_ids=None, item_ids=None):
        if user_ids is None:
            user_ids = slice(None)

        if item_ids is None:
            item_ids = slice(None)

        # Need to clone to avoid gradient hook accumulation on same tensor and subsequent memory leak
        item_rating_effect_weight = self.item_rating_effect_weight.clone()

        item_weight = self.item_weight[item_ids]

        users_implicit_feedback = torch_sparse_slice(
            self.implicit_feedback_normalized, row_ids=user_ids, device=self.bias.device
        )
        user_weights_offset_caused_by_their_ratings = (
            users_implicit_feedback @ item_rating_effect_weight
        )

        rating = super().forward(user_ids, item_ids)
        rating += user_weights_offset_caused_by_their_ratings @ item_weight.T

        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        scale_down = self.user_weight.shape[0] / len(user_ids)
        register_regularization_hook(
            item_rating_effect_weight, self.weight_decay / scale_down
        )
        return rating


class PMFDataModule(MovieLensDataModule):
    def train_dataloader(self):
        if (explicit := self.train_explicit) is not None:
            return self.build_dataloader(
                SparseDataset(explicit), sampler_type="grid", shuffle=True
            )


class LitProbabilityMatrixFactorization(RecommenderBase):
    def build_model(self):
        model_config = self.hparams["model_config"]
        model_candidates = [
            ProbabilityMatrixFactorization,
            ConstrainedProbabilityMatrixFactorization,
        ]

        if model_config["name"] == "ConstrainedProbabilityMatrixFactorization":
            model_config = model_config.copy()
            model_config["implicit_feedback"] = self.train_explicit > 0

        model = build_class(
            class_candidates=model_candidates,
            n_users=self.train_explicit.shape[0],
            n_items=self.train_explicit.shape[1],
            **model_config,
        )
        return model

    def common_step(self, batch):
        ratings = self(**batch)
        loss = self.loss(
            explicit=batch["explicit"],
            implicit=batch["implicit"],
            model_ratings=ratings,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log(f"train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch)
        self.log(f"val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        pass


class PMFDispatcher(MovielensDispatcher):
    def lightning_candidates(self):
        return (LitProbabilityMatrixFactorization,)

    def datamodule_candidates(self):
        return (PMFDataModule,)


@ConfigDispenser
def main(config):
    PMFDispatcher(config).dispatch()


if __name__ == "__main__":
    main()
