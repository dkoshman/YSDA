import abc
from abc import ABC

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from Machine_Learning.Recomending.data import MovieLensDataModule
from my_tools.entrypoints import ConfigConstructorBase, ConfigDispenser
from my_tools.utils import build_class

import losses
from metrics import RecommendingMetricsCallback, RecommendingIMDBCallback, ImdbRatings
from nearest_neighbours import NearestNeighbours


class RecommenderBase(pl.LightningModule):
    def __init__(
        self, model_config, optimizer_config, loss_config, train_explicit=None
    ):
        super().__init__()
        self.save_hyperparameters(ignore="train_explicit")
        self.model = None
        self.loss = None
        self.train_explicit = train_explicit
        self.recommender = (
            None if train_explicit is None else Recommender(train_explicit, self)
        )

    @abc.abstractmethod
    def build_model(self):
        return "model"

    def build_loss(self):
        loss = build_class(modules=[losses], **self.hparams["loss_config"])
        return loss

    def setup(self, stage=None):
        if stage == "fit":
            if self.train_explicit is None:
                self.train_explicit = self.trainer.datamodule.train_explicit
                self.recommender = Recommender(self.train_explicit, self)
            self.model = self.build_model()
            self.loss = self.build_loss()

    def configure_optimizers(self):
        optimizer = build_class(
            modules=[torch.optim],
            **self.hparams["optimizer_config"],
            params=self.parameters(),
        )
        return optimizer

    def forward(self, **batch):
        return self.model(
            user_ids=batch.get("user_ids"), item_ids=batch.get("item_ids")
        )

    def recommend(
        self,
        user_ids=None,
        users_explicit_feedback=None,
        k=None,
        filter_already_liked_items=True,
    ):
        return self.recommender.__call__(
            user_ids=user_ids,
            users_explicit_feedback=users_explicit_feedback,
            k=k,
            filter_already_liked_items=filter_already_liked_items,
        )


class Recommender:
    def __init__(self, train_explicit, model):
        self.train_explicit = train_explicit
        self.nearest_neighbours = NearestNeighbours(self.train_explicit)
        self.model = model

    def nearest_neighbours_ratings(self, explicit_feedback):
        """
        Predicts ratings for new user defined by its explicit feedback by
        searching for closest neighbors and taking weighted average of
        their predicted ratings.
        """

        nn_dict = self.nearest_neighbours(explicit_feedback)
        ratings = torch.empty(*explicit_feedback.shape)
        for i, (similar_users, similarity) in enumerate(
            zip(nn_dict["similar_users"], nn_dict["similarity"])
        ):
            similar_ratings = self.model(user_ids=similar_users)
            if isinstance(similar_ratings, np.ndarray):
                similar_ratings = torch.from_numpy(similar_ratings).to(torch.float32)
            ratings[i] = similar_ratings.T @ similarity
        return ratings

    def __call__(
        self,
        user_ids=None,
        users_explicit_feedback=None,
        k=None,
        filter_already_liked_items=True,
    ):
        if user_ids is None and users_explicit_feedback is None:
            raise ValueError(
                "At least one of user_ids, user_explicit_feedback must be passed."
            )

        if user_ids is None:
            ratings = self.nearest_neighbours_ratings(users_explicit_feedback)
        else:
            ratings = self.model(user_ids=user_ids)

        if filter_already_liked_items:
            if user_ids is None:
                already_liked_items = users_explicit_feedback > 0
            else:
                already_liked_items = self.train_explicit[user_ids] > 0

            ratings -= (
                torch.from_numpy(already_liked_items.toarray()) * torch.finfo().max / 2
            )

        if k is None:
            recommendations = torch.argsort(ratings, descending=True)
        else:
            values, recommendations = torch.topk(input=ratings, k=k)
        return recommendations


class RecommendingConfigDispenser(ConfigDispenser):
    def parser(self, parser):
        parser.add_argument(
            "--stage",
            "-s",
            default="tune",
            type=str,
            help="One of: tune, test.",
            nargs="?",
        )
        return parser

    def update_config(self, config: dict) -> dict:
        if "trainer" in config:
            config["trainer"].update(devices=None, accelerator=None)
        return config


class MovielensDispatcher(ConfigConstructorBase, abc.ABC):
    def datamodule_candidates(self):
        return (MovieLensDataModule,)

    def callback_candidates(self):
        return RecommendingIMDBCallback, RecommendingMetricsCallback

    def test_datasets_iter(self):
        for i in [2, 3, 4, 5]:
            self.config["datamodule"].update(
                dict(
                    train_explicit_file=f"u{i}.base",
                    test_explicit_file=f"u{i}.test",
                    # tmp before other early stopping is implemented
                    val_explicit_file=f"u{i}.test",
                )
            )
            yield

    def update_tune_data(self):
        self.config["datamodule"].update(
            dict(
                train_explicit_file="u1.base",
                val_explicit_file="u1.test",
            )
        )

    def tune(self):
        self.update_tune_data()
        if wandb.run is None and self.config.get("logger") is not None:
            with wandb.init(project=self.config["project"], config=self.config):
                return self.main()
        else:
            return self.main()

    def test(self):
        with wandb.init(project=self.config["project"], config=self.config):
            for _ in self.test_datasets_iter():
                self.main()

    def dispatch(self):
        match stage := self.config.get("stage"):
            case "tune":
                self.tune()
            case "test":
                self.test()
            case _:
                raise ValueError(f"Unknown stage {stage}.")
