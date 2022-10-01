import abc
from typing import Literal

import numpy as np
import pytorch_lightning as pl
import torch
import wandb

from my_tools.entrypoints import ConfigConstructorBase
from my_tools.utils import build_class

import losses

from data import MovieLensDataModule
from metrics import RecommendingMetricsCallback, RecommendingIMDBCallback
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
            ratings[i] = similar_ratings.transpose(0, 1) @ similarity
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


class MovielensDispatcher(ConfigConstructorBase, abc.ABC):
    def __init__(self, config):
        if "callbacks" not in config:
            config["callbacks"] = {}
        config["callbacks"].update(
            RecommendingIMDBCallback=dict(
                path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
                path_to_movielens_folder="local/ml-100k",
                k=10,
            ),
            RecommendingMetricsCallback=dict(directory="local/ml-100k", k=[10, 20]),
        )
        super().__init__(config)

    def datamodule_candidates(self):
        return (MovieLensDataModule,)

    def callback_candidates(self):
        return RecommendingIMDBCallback, RecommendingMetricsCallback

    def update_tune_data(self):
        self.config["datamodule"].update(
            dict(
                train_explicit_file="u1.base",
                val_explicit_file="u1.test",
                test_explicit_file="u1.test",
            )
        )

    def tune(self):
        self.update_tune_data()
        if wandb.run is None and self.config.get("logger") is not None:
            with wandb.init(project=self.config.get("project"), config=self.config):
                return self.main()
        else:
            return self.main()

    def test_datasets_iter(self):
        for i in [2, 3, 4, 5]:
            self.config["datamodule"].update(
                dict(
                    train_explicit_file=f"u{i}.base",
                    val_explicit_file=f"u{i}.base",
                    test_explicit_file=f"u{i}.test",
                )
            )
            yield

    def test(self):
        with wandb.init(project=self.config["project"], config=self.config):
            for _ in self.test_datasets_iter():
                self.main()

    def dispatch(self):
        if self.config.get("stage") == "test":
            self.test()
        else:
            self.tune()


class NonLightningDispatcher(MovielensDispatcher):
    def build_model(self):
        raise NotImplementedError

    def main(self):
        self.datamodule = self.build_datamodule([MovieLensDataModule])
        self.model = self.build_model()
        self.callbacks = self.build_callbacks()

        self.model.fit()
        self.log("val")
        self.log("test")

    def log(self, kind: Literal["val", "test"]):
        match kind:
            case "val":
                dataloader = self.datamodule.val_dataloader()
            case "test":
                dataloader = self.datamodule.test_dataloader()
            case _:
                raise ValueError(f"Unknown epoch type {kind}")

        metrics_callbacks = filter(
            lambda x: isinstance(x, RecommendingMetricsCallback),
            self.callbacks.values(),
        )
        imdb_callbacks = filter(
            lambda x: isinstance(x, RecommendingIMDBCallback),
            self.callbacks.values(),
        )

        if dataloader:
            for batch in dataloader:
                user_ids = batch["user_ids"]
                ratings = self.model(user_ids=user_ids.numpy())
                for callback in metrics_callbacks:
                    callback.log_batch(
                        user_ids=user_ids, ratings=torch.from_numpy(ratings), kind=kind
                    )

            for callback in metrics_callbacks:
                match kind:
                    case "val":
                        callback.on_validation_epoch_end()
                    case "test":
                        callback.on_test_epoch_end()

            for callback in imdb_callbacks:
                recommender = Recommender(self.datamodule.train_explicit, self.model)
                recommendations = recommender(
                    users_explicit_feedback=callback.explicit_feedback
                )
                callback.log_recommendation(recommendations)
