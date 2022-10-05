import abc
from typing import Literal

import pytorch_lightning as pl
import torch
from my_tools.entrypoints import ConfigDispenser

from my_tools.utils import build_class

import losses, baseline
from baseline import ImplicitNearestNeighbors
from movielens import MovielensDispatcher

# TODO:catboost, transformer, maybe other neural nets


class LitRecommenderBase(pl.LightningModule):
    def __init__(
        self,
        model_config,
        optimizer_config=None,
        loss_config=None,
        recommender_config=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = None
        self.loss = None
        self.recommend = None

    @abc.abstractmethod
    def build_model(self):
        return "model"

    def build_loss(self):
        if config := self.hparams["loss_config"]:
            loss = build_class(modules=[losses], **config)
            return loss

    def build_recommender(self):
        config = self.hparams["recommender_config"]
        if config is None:
            config = dict()

        recommender = Recommender(
            train_explicit=self.trainer.datamodule.train_explicit,
            model=self,
            **config,
        )
        return recommender

    def setup(self, stage=None):
        if self.model is None:
            self.model = self.build_model()
            self.loss = self.build_loss()
            self.recommend = self.build_recommender()

    def configure_optimizers(self):
        optimizer = build_class(
            modules=[torch.optim],
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


class NonLitToLitAdapterRecommender(LitRecommenderBase):
    @abc.abstractmethod
    def build_model(self):
        model = build_class(
            class_candidates=[...],
            modules=[...],
            explicit_feedback=self.trainer.datamodule.train_explicit,
            **self.hparams["model_config"],
        )
        return model

    def on_train_batch_start(self, batch, batch_idx):
        """Skip train dataloader."""
        self.model.fit(explicit_feedback=self.trainer.datamodule.train_explicit)
        self.trainer.should_stop = True
        return -1

    def configure_optimizers(self):
        """Placeholder optimizer."""
        adam = torch.optim.Adam(params=[torch.empty(0)])
        return adam


class Recommender:
    def __init__(
        self,
        train_explicit,
        model,
        nearest_neighbors_model: Literal[
            "BM25Recommender",
            "CosineRecommender",
            "TFIDFRecommender",
        ] = "BM25Recommender",
        num_neighbors=20,
    ):
        """
        :param train_explicit: explicit feedback matrix
        :param model: Callable[[user_ids], ratings]
        :param nearest_neighbors_model: name of nearest neighbors
        model to use for generating recommendations fo new users
        :param num_neighbors: number of nearest neighbors to use
        """
        self.train_explicit = train_explicit
        self.nearest_neighbours = ImplicitNearestNeighbors(
            explicit_feedback=self.train_explicit,
            name=nearest_neighbors_model,
            num_neighbors=num_neighbors,
        )
        self.model = model

    @torch.inference_mode()
    def nearest_neighbours_ratings(self, explicit_feedback):
        """
        Predicts ratings for new user defined by its explicit feedback by
        searching for closest neighbors and taking weighted average of
        their predicted ratings.
        """

        nn_dict = self.nearest_neighbours.similar_users(
            users_feedback=explicit_feedback
        )
        similar_users = nn_dict["similar_users"]
        similarity = torch.from_numpy(nn_dict["similarity"])
        similarity /= similarity.sum(axis=0)

        ratings = torch.empty(*explicit_feedback.shape, dtype=torch.float32)
        for i, (similar_users_row, similarity_row) in enumerate(
            zip(similar_users, similarity)
        ):
            similar_ratings = self.model(user_ids=similar_users_row)
            if not torch.is_tensor(similar_ratings):
                similar_ratings = torch.from_numpy(similar_ratings)

            similar_ratings = similar_ratings.to("cpu", torch.float32)
            ratings[i] = similar_ratings.transpose(0, 1) @ similarity_row
        return ratings

    @torch.inference_mode()
    def __call__(
        self,
        user_ids=None,
        users_explicit_feedback=None,
        filter_already_liked_items=True,
        n_recommendations=10,
    ):
        if user_ids is None and users_explicit_feedback is None:
            raise ValueError(
                "At least one of user_ids, user_explicit_feedback must be passed."
            )

        if user_ids is None:
            ratings = self.nearest_neighbours_ratings(users_explicit_feedback)
        else:
            ratings = self.model(user_ids=user_ids)

        if not torch.is_tensor(ratings):
            ratings = torch.from_numpy(ratings)

        if filter_already_liked_items:
            if user_ids is None:
                already_liked_items = users_explicit_feedback > 0
            else:
                already_liked_items = self.train_explicit[user_ids] > 0

            ratings -= (
                torch.from_numpy(already_liked_items.toarray()) * torch.finfo().max / 2
            )

        if n_recommendations is None:
            recommendations = torch.argsort(ratings, descending=True)
        else:
            values, recommendations = torch.topk(input=ratings, k=n_recommendations)
        return recommendations


class LitBaselineRecommender(NonLitToLitAdapterRecommender):
    def build_model(self):
        model = build_class(
            modules=[baseline],
            explicit_feedback=self.trainer.datamodule.train_explicit,
            **self.hparams["model_config"],
        )
        return model


@ConfigDispenser
def main(config):
    MovielensDispatcher(
        config=config,
        lightning_candidates=[LitBaselineRecommender],
    ).dispatch()


if __name__ == "__main__":
    main()
