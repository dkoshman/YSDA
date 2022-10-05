import abc

import torch
import wandb

from callbacks import (
    RecommendingIMDBCallback,
    RecommendingDataOverviewCallback,
    WandbWatcher,
)
from data import MovieLens, RecommendingDataModule
from metrics import RecommendingMetricsCallback

from my_tools.entrypoints import ConfigConstructorBase


class MovieLensDataModule(RecommendingDataModule):
    def __init__(self, directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens(directory)
        self.n_users, self.n_items = self.movielens.shape
        self.save_hyperparameters(ignore="movielens")

    @property
    def train_explicit(self):
        if file := self.hparams["train_explicit_file"]:
            return self.movielens.explicit_feedback_scipy_csr(file)

    @property
    def val_explicit(self):
        if file := self.hparams["val_explicit_file"]:
            return self.movielens.explicit_feedback_scipy_csr(file)

    @property
    def test_explicit(self):
        if file := self.hparams["test_explicit_file"]:
            return self.movielens.explicit_feedback_scipy_csr(file)


class MovielensDispatcher(ConfigConstructorBase, abc.ABC):
    def __init__(
        self,
        config,
        lightning_candidates=(),
        datamodule_candidates=(),
        callback_candidates=(),
    ):
        # For some reason with some configuration it is necessary to manually init cuda.
        torch.cuda.init()

        if "callbacks" not in config:
            config["callbacks"] = {}
        config["callbacks"].update(
            RecommendingIMDBCallback=dict(
                path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
                path_to_movielens_folder="local/ml-100k",
                n_recommendations=10,
            ),
            RecommendingMetricsCallback=dict(
                directory="local/ml-100k",
                k=100,
            ),
        )
        datamodule_candidates = list(datamodule_candidates) + [MovieLensDataModule]
        callback_candidates = list(callback_candidates) + [
            RecommendingIMDBCallback,
            RecommendingMetricsCallback,
            RecommendingDataOverviewCallback,
            WandbWatcher,
        ]
        super().__init__(
            config,
            lightning_candidates=lightning_candidates,
            datamodule_candidates=datamodule_candidates,
            callback_candidates=callback_candidates,
        )

    def update_tune_data(self):
        self.config["datamodule"].update(
            dict(
                train_explicit_file="u1.base",
                val_explicit_file="u1.base",
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
