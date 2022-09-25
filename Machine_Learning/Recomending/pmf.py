import pytorch_lightning as pl
import torch

from my_tools.entrypoints import ConfigConstructorBase
from my_tools.lightning import ConvenientCheckpointLogCallback
from my_tools.models import register_regularization_hook

from data import MovieLensDataModule
from loss import (
    ImplicitAwareSparseLoss,
    PersonalizedRankingLoss,
    PersonalizedRankingLossMemoryEfficient,
)
from utils import (
    build_bias,
    build_weight,
    MovielensTester,
    MovielensTuner,
    RecommendingConfigDispenser,
    RecommenderMixin,
    torch_sparse_slice,
)


class ProbabilityMatrixFactorization(torch.nn.Module):
    """predicted rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension,
        weight_decay,
        pass_through_sigmoid=False,
    ):
        super().__init__()

        self.weight_decay = weight_decay
        self.pass_through_sigmoid = pass_through_sigmoid

        self.user_weight = build_weight(n_users, latent_dimension)
        self.user_bias = build_bias(n_users, 1)

        self.item_weight = build_weight(n_items, latent_dimension)
        self.item_bias = build_bias(n_items, 1)

        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.sigmoid = torch.nn.Sigmoid()

    def linear_forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]
        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias

        # Need to add regularization here because otherwise optimizer will decay all weights,
        # not only those corresponding to user and item ids. Also it is important to add gradient
        # hooks after the forward calculations, otherwise decay messes with the model.
        for parameter in [user_weight, item_weight, user_bias, item_bias]:
            register_regularization_hook(parameter, self.weight_decay)

        return rating

    def forward(self, user_ids, item_ids):
        rating = self.linear_forward(user_ids, item_ids)
        if self.pass_through_sigmoid:
            rating = self.sigmoid(rating)
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

    def forward(self, user_ids, item_ids):
        # Need to clone to avoid gradient hook accumulation on same tensor and subsequent memory leak
        item_rating_effect_weight = self.item_rating_effect_weight.clone()

        item_weight = self.item_weight[item_ids]

        users_implicit_feedback = torch_sparse_slice(
            self.implicit_feedback_normalized, row_ids=user_ids, device=self.bias.device
        )
        user_weights_offset_caused_by_their_ratings = (
            users_implicit_feedback @ item_rating_effect_weight
        )

        rating = super().linear_forward(user_ids, item_ids)
        rating += user_weights_offset_caused_by_their_ratings @ item_weight.T
        if self.pass_through_sigmoid:
            rating = self.sigmoid(rating)

        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        scale_down = self.user_weight.shape[0] / len(user_ids)
        register_regularization_hook(
            item_rating_effect_weight, self.weight_decay / scale_down
        )

        return rating


class LitProbabilityMatrixFactorization(RecommenderMixin, pl.LightningModule):
    def __init__(self, model_config, optimizer_config, loss_config):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.build_model()
        self.loss = self.build_loss()

    def build_model(self):
        model_config = self.hparams["model_config"].copy()
        model_candidates = [
            ProbabilityMatrixFactorization,
            ConstrainedProbabilityMatrixFactorization,
        ]
        model_config["n_users"] = self.train_explicit_feedback.shape[0]
        model_config["n_items"] = self.train_explicit_feedback.shape[1]
        if model_config["name"] == "ConstrainedProbabilityMatrixFactorization":
            model_config["implicit_feedback"] = self.train_explicit_feedback
        return super().build_class(model_config, model_candidates)

    def build_loss(self):
        loss = super().build_class(
            class_config=self.hparams["loss_config"],
            class_candidates=[
                ImplicitAwareSparseLoss,
                PersonalizedRankingLoss,
                PersonalizedRankingLossMemoryEfficient,
            ],
        )
        return loss

    def forward(self, **batch):
        return self.model(
            user_ids=batch.get("user_ids", slice(None)),
            item_ids=batch.get("item_ids", slice(None)),
        )

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
        ratings = self(**batch)
        return dict(ratings=ratings)


class PMFDispatcher(ConfigConstructorBase):
    def main(self):
        lightning_module = self.build_lightning_module(
            [LitProbabilityMatrixFactorization]
        )
        datamodule = self.build_datamodule([MovieLensDataModule])
        trainer = self.build_trainer()
        trainer.fit(lightning_module, datamodule=datamodule)
        # if test is not None
        trainer.test(lightning_module, datamodule=datamodule)


class PMFTuner(PMFDispatcher, MovielensTuner):
    pass


class PMFTester(PMFDispatcher, MovielensTester):
    pass


class PMFConfigDispenser(RecommendingConfigDispenser):
    def debug_config(self, config):
        config = super().debug_config(config)
        config["lightning_module"].update(dict(batch_size=1e6))
        return config


@PMFConfigDispenser
def main(config):
    match stage := config["stage"]:
        case "tune":
            PMFTuner(config).main()
        case "test":
            PMFTester(config).test()
        case _:
            raise ValueError(f"Unknown stage {stage}.")


if __name__ == "__main__":
    main()
