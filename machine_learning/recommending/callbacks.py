import io
from typing import Literal, TYPE_CHECKING, TextIO

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import wandb

from matplotlib import pyplot as plt
from pytorch_lightning import loggers as pl_loggers
from sklearn.decomposition import TruncatedSVD
import matplotlib.patheffects as pe

from my_tools.lightning import ConvenientCheckpointLogCallback
from .data import SparseDataModuleInterface
from .interface import ExplanationMixin
from .metrics import RecommendingMetrics
from .utils import wandb_plt_figure

if TYPE_CHECKING:
    from scipy.sparse import spmatrix
    from .interface import RecommenderModuleBase
    from .lit import LitRecommenderBase
    from .models.cat import CatboostInterface


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

    def setup(self, trainer, pl_module, stage=None):
        if self.explicit is None:
            self.explicit = pl_module.train_explicit
            self.log_data_overview()

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
            with wandb_plt_figure(title=f"Distribution of {what} reviews"):
                plt.xlabel(f"Number of reviews per {what}")
                plt.ylabel("Quantity")
                plt.hist(data, histtype="stepfilled", bins=50)
                self.add_quantiles(data)

    def log_data_overview(self):
        self.log_data_description()
        self.log_ratings_distributions()

        with wandb_plt_figure(title="Distribution of ratings"):
            plt.xlabel("Number of reviews per rating value")
            plt.ylabel("Quantity")
            plt.hist(self.explicit.data, histtype="stepfilled", bins=50)

        with wandb_plt_figure(
            title="Explained variance depending on number of components"
        ):
            plt.xlabel("Number of components")
            plt.ylabel("Explained cumulative variance ratio")
            self.plot_explained_variance()


class WandbCheckpointCallback(ConvenientCheckpointLogCallback):
    def __init__(self, *args, artifact_name, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_name = artifact_name

    def on_test_end(self, trainer, pl_module):
        artifact = wandb.Artifact(
            name=self.artifact_name,
            type="checkpoint",
            metadata=dict(class_name=pl_module.__class__.__name__),
        )
        artifact.add_file(local_path=self.best_model_path, name="checkpoint")
        wandb.run.log_artifact(artifact)
        super().on_test_end(trainer, pl_module)


class CatBoostMetrics(pl.callbacks.Callback):
    """Log some metrics specific to catboost."""

    def __init__(self, *args, plot_dependence=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_dependence = plot_dependence

    def on_test_epoch_end(self, trainer=None, pl_module: "LitRecommenderBase" = None):
        model: "CatboostInterface" = pl_module.model
        for explicit, stage in zip(
            [pl_module.train_explicit(), pl_module.test_explicit()], ["train", "test"]
        ):
            self.log_feature_importance(model, explicit, stage)
            self.log_shap_plots(
                model, explicit, stage, plot_dependence=self.plot_dependence
            )

    @staticmethod
    def log_feature_importance(
        model: "CatboostInterface", explicit: "spmatrix", stage: str
    ):
        dataframe = model.feature_importance(explicit=explicit)
        wandb_table = wandb.Table(dataframe=dataframe)
        wandb.log({f"Catboost {stage} feature importance": wandb_table})

    @staticmethod
    def force_plot(
        shap_values, expected_value, features, subsample_size=1000, **shap_kwargs
    ) -> TextIO:
        subsample = np.random.choice(
            np.arange(shap_values.shape[0]),
            replace=False,
            size=min(subsample_size, shap_values.shape[0]),
        )
        shap_plot = shap.force_plot(
            base_value=expected_value,
            shap_values=shap_values[subsample],
            features=features.iloc[subsample],
            **shap_kwargs,
        )
        textio = io.TextIOWrapper(io.BytesIO())
        shap.save_html(textio, shap_plot)
        textio.seek(0)
        return textio

    def log_shap_plots(
        self,
        model: "CatboostInterface",
        explicit: "spmatrix",
        stage: str,
        plot_dependence=False,
    ):
        shap_values, expected_value, features = model.shap(explicit)
        prefix = f"{stage} shap "

        textio = self.force_plot(shap_values, expected_value, features)
        wandb.log({prefix + "force plot": wandb.Html(textio)})

        with wandb_plt_figure(title=prefix + "summary plot"):
            shap.summary_plot(shap_values, features)

        if plot_dependence:
            for feature_name in features:
                with wandb_plt_figure(
                    title=prefix + "dependence plot for " + feature_name
                ) as figure:
                    shap.dependence_plot(
                        feature_name, shap_values, features, ax=figure.gca()
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
        if stage == "train":
            filter_already_liked_items = False
            metrics = self.train_metrics
        elif stage == "val":
            filter_already_liked_items = True
            metrics = self.val_metrics
        elif stage == "test":
            filter_already_liked_items = True
            metrics = self.test_metrics
        else:
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


class RecommendingExplanationCallback(pl.callbacks.Callback):
    def __init__(
        self,
        user_ids=None,
        users_explicit=None,
        n_recommendations=10,
    ):
        self.user_ids = user_ids
        self.users_explicit = users_explicit
        self.n_recommendations = n_recommendations

    def on_test_epoch_end(self, trainer=None, pl_module=None):
        self.my_on_epoch_end(model=pl_module.model, stage="test")

    def on_validation_epoch_end(self, trainer=None, pl_module=None):
        self.my_on_epoch_end(model=pl_module.model, stage="val")

    def my_on_epoch_end(
        self,
        model: "RecommenderModuleBase",
        stage: Literal["train", "val", "test", "predict"],
    ):
        if not isinstance(model, ExplanationMixin):
            return
        if self.user_ids is not None:
            for user_id in self.user_ids:
                model.explain_recommendations(
                    user_id=user_id,
                    n_recommendations=self.n_recommendations,
                    log=True,
                    logging_prefix=stage,
                )

        if self.users_explicit is not None:
            try:
                for user_explicit in self.users_explicit.detach().clone():
                    model.explain_recommendations(
                        user_explicit=user_explicit[None],
                        n_recommendations=self.n_recommendations,
                        log=True,
                        logging_prefix=stage,
                    )
            except NotImplementedError:
                pass
