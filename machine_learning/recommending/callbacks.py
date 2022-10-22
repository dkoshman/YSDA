from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from matplotlib import pyplot as plt
from pytorch_lightning import loggers as pl_loggers
from sklearn.decomposition import TruncatedSVD
import matplotlib.patheffects as pe

from .data import SparseDataModuleInterface
from .metrics import RecommendingMetrics
from .utils import save_checkpoint_artifact

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from .interface import RecommenderModuleBase
    from .lit import LitRecommenderBase


class WandbWatcher(pl.callbacks.Callback):
    """Log torch parameter values to wandb."""

    def __init__(
        self,
        log_what: Literal["gradients", "parameters", "all"] = "all",
        log_every_n_steps=100,
        log_graph=True,
    ):
        self.watch_kwargs = dict(
            log=log_what,
            log_freq=log_every_n_steps,
            log_graph=log_graph,
        )
        # Wandb errors if you try to watch same model twice
        self.watch_model_triggered = False

    def setup(self, trainer, pl_module, stage=None):
        if self.watch_model_triggered:
            return
        if not isinstance(trainer.logger, pl_loggers.WandbLogger):
            raise ValueError("Only wandb logger supports watching model.")
        trainer.logger.watch(pl_module, **self.watch_kwargs)
        self.watch_model_triggered = True

    def teardown(self, trainer, pl_module, stage=None):
        trainer.logger.experiment.unwatch(pl_module)


class RecommendingDataOverviewCallback(pl.callbacks.Callback):
    """Logs to wandb some general information about the data."""

    def __init__(self, explicit: "spmatrix" = None):
        self.explicit = explicit
        self.fig = None

    def setup(self, trainer, pl_module, stage=None):
        if self.explicit is None:
            self.explicit = pl_module.train_explicit
            self.log_data_overview()

    def __enter__(self):
        return self

    def __call__(self, title, xlabel, ylabel):
        self.fig = plt.figure()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # For interactive plots, remove wandb.Image, although some features are not supported
        wandb.log({self.fig.axes[0].get_title(): wandb.Image(self.fig)})
        self.fig = None

    @staticmethod
    def plot_hist(data, histtype="stepfilled", bins=50, **kwargs):
        plt.hist(data, histtype=histtype, bins=bins, **kwargs)

    @staticmethod
    def add_quantiles(
        data,
        cmap="hsv",
        quantiles_to_plot=(0.5, 0.75, 0.9, 0.99),
    ):
        for q in quantiles_to_plot:
            q_value = np.quantile(data, q=q)
            plt.axvline(q_value, ls="--", c=plt.get_cmap(name=cmap)(q))
            plt.text(
                x=q_value,
                y=plt.ylim()[1] * 0.95,
                s=f"{100 * q}%",
                ha="left",
                color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    def plot_explained_variance(self):
        max_n_components = min(self.explicit.shape)
        truncated_svd = TruncatedSVD(n_components=max_n_components)
        truncated_svd.fit(self.explicit)
        plt.plot(np.cumsum(truncated_svd.explained_variance_ratio_))

    def log_data_description(self):
        series = pd.Series(
            {
                "n_users": self.explicit.shape[0],
                "n_items": self.explicit.shape[1],
                "density": self.explicit.nnz / np.prod(self.explicit.shape),
            }
        )
        ratings_description = pd.Series(self.explicit.data).describe()
        ratings_description.index = "ratings_" + ratings_description.index
        series = pd.concat([series, ratings_description])
        wandb.log({"Data description": wandb.Table(dataframe=series.to_frame().T)})

    def log_ratings_distributions(self):
        user_reviews_counts = self.explicit.sum(axis=1).A.squeeze()
        item_reviews_counts = self.explicit.sum(axis=0).A.squeeze()
        for what, data in zip(
            ["user", "item"], [user_reviews_counts, item_reviews_counts]
        ):
            with self(
                title=f"Distribution of {what} reviews",
                xlabel=f"Number of reviews per {what}",
                ylabel="Quantity",
            ):
                self.plot_hist(data)
                self.add_quantiles(data)

    def log_data_overview(self):
        self.log_data_description()
        self.log_ratings_distributions()

        with self(
            title=f"Distribution of ratings",
            xlabel=f"Number of reviews per rating value",
            ylabel="Quantity",
        ):
            self.plot_hist(self.explicit.data)

        with self(
            title="Explained variance depending on number of components",
            xlabel="n components",
            ylabel="explained cumulative variance ratio",
        ):
            self.plot_explained_variance()


class WandbCheckpointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, artifact_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_name = artifact_name

    def on_test_end(self, trainer, pl_module):
        if not self.best_model_path:
            self.save_checkpoint(trainer)
        save_checkpoint_artifact(
            artifact_name=self.artifact_name,
            checkpoint_path=self.best_model_path,
            pl_module_class=pl_module.__class__.__name__,
        )
        super().on_test_end(trainer, pl_module)


class CatBoostMetrics(pl.callbacks.Callback):
    """Log some metrics specific to catboost."""

    def on_test_epoch_end(self, trainer=None, pl_module: "LitRecommenderBase" = None):
        self.log_feature_importance(pl_module)

    @staticmethod
    def log_feature_importance(pl_module: "LitRecommenderBase"):
        for explicit, stage in zip(
            [pl_module.train_explicit(), pl_module.test_explicit()], ["train", "test"]
        ):
            dataframe = pl_module.model.feature_importance(explicit=explicit)
            wandb.log(
                {
                    f"Catboost {stage} feature importance": wandb.Table(
                        dataframe=dataframe
                    )
                }
            )


class RecommendingMetricsCallback(pl.callbacks.Callback):
    def __init__(
        self,
        k: int = 10,
        every_train_epoch=1,
        every_val_epoch=1,
        every_test_epoch=1,
    ):
        self.k = k
        self.every_train_epoch = every_train_epoch
        self.every_val_epoch = every_val_epoch
        self.every_test_epoch = every_test_epoch
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None

    def setup(self, trainer, pl_module: SparseDataModuleInterface, stage=None):
        if self.train_metrics is not None:
            return
        if isinstance(trainer.datamodule, SparseDataModuleInterface):
            module = trainer.datamodule
        elif isinstance(pl_module, SparseDataModuleInterface):
            module = pl_module
        else:
            raise ValueError(
                f"One of lightning_module, datamodule must implement "
                f"the {SparseDataModuleInterface.__name__} interface"
            )

        if (explicit := module.train_explicit()) is not None:
            self.train_metrics = RecommendingMetrics(explicit)
        if (explicit := module.val_explicit()) is not None:
            self.val_metrics = RecommendingMetrics(explicit)
        if (explicit := module.test_explicit()) is not None:
            self.test_metrics = RecommendingMetrics(explicit)

    @torch.inference_mode()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (
            self.every_train_epoch != 0
            and pl_module.current_epoch % self.every_train_epoch == 0
        ):
            self.log_batch(
                user_ids=batch["user_ids"], model=pl_module.model, stage="train"
            )

    @torch.inference_mode()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (
            self.every_val_epoch != 0
            and pl_module.current_epoch % self.every_val_epoch == 0
        ):
            self.log_batch(
                user_ids=batch["user_ids"], model=pl_module.model, stage="val"
            )

    @torch.inference_mode()
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if (
            self.every_test_epoch != 0
            and pl_module.current_epoch % self.every_test_epoch == 0
        ):
            self.log_batch(
                user_ids=batch["user_ids"], model=pl_module.model, stage="test"
            )

    def log_batch(
        self,
        user_ids: torch.IntTensor or None,
        model: "RecommenderModuleBase",
        stage: Literal["train", "val", "test"],
    ):
        if user_ids is None:
            user_ids = torch.arange(model.n_users)
        match stage:
            case "train":
                filter_already_liked_items = False
                metrics = self.train_metrics
            case "val":
                filter_already_liked_items = True
                metrics = self.val_metrics
            case "test":
                filter_already_liked_items = True
                metrics = self.test_metrics
            case _:
                raise ValueError(f"Unknown stage {stage}")

        recommendations = model.recommend(
            user_ids=user_ids,
            n_recommendations=self.k,
            filter_already_liked_items=filter_already_liked_items,
        )
        metrics_dict = metrics.batch_metrics(
            user_ids=user_ids, recommendations=recommendations
        )
        self.log_metrics(metrics_dict, stage=stage)

    @staticmethod
    def log_metrics(metrics: dict, stage: Literal["train", "val", "test"]):
        metrics = {stage + "_" + k: v for k, v in metrics.items()}
        for metric_name in metrics:
            wandb.define_metric(metric_name, summary="mean")
        wandb.log(metrics)

    def on_train_epoch_end(self, trainer=None, pl_module=None):
        self.epoch_end(self.train_metrics, kind="train")

    def on_validation_epoch_end(self, trainer=None, pl_module=None):
        self.epoch_end(self.val_metrics, kind="val")

    def on_test_epoch_end(self, trainer=None, pl_module=None):
        self.epoch_end(self.test_metrics, kind="test")

    def epoch_end(self, metrics, kind: Literal["train", "val", "test"]):
        if metrics is not None and len(metrics.unique_recommended_items):
            metrics = metrics.finalize_coverage()
            self.log_metrics(metrics, kind)
