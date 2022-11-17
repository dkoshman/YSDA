import io
import os.path
import re
from typing import Literal, TYPE_CHECKING, TextIO

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import torch
import wandb

from matplotlib import pyplot as plt
from pytorch_lightning import loggers, profiler
import matplotlib.patheffects as pe

from .data import SparseDataModuleInterface
from .interface import ExplanationMixin
from .metrics import RecommendingMetrics
from .utils import wandb_plt_figure, filter_warnings

if TYPE_CHECKING:
    import catboost
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
        if not isinstance(trainer.logger, loggers.WandbLogger):
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


class WandbCheckpointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, artifact_name, aliases=None, **kwargs):
        self.artifact_name = artifact_name
        self.aliases = aliases
        super().__init__(*args, **kwargs)

    def on_fit_end(self, trainer, pl_module):
        if not self.best_model_path:
            self.save_checkpoint(trainer)
        trainer.logger.log_text(
            key="checkpoint", columns=["path"], data=[[self.best_model_path]]
        )
        wandb.log_artifact(
            artifact_or_path=self.best_model_path,
            name=self.artifact_name,
            type="checkpoint",
            aliases=self.aliases,
        )


class CatBoostMetrics(pl.callbacks.Callback):
    """Log some metrics specific to catboost."""

    def __init__(self, *args, plot_dependence=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.plot_dependence = plot_dependence

    def on_train_end(self, trainer, pl_module):
        self.log_feature_names(model=pl_module.model)

    def on_train_epoch_end(self, trainer, pl_module: "LitRecommenderBase"):
        self.common_on_epoch_end(pl_module=pl_module, stage="train")

    def on_val_epoch_end(self, trainer, pl_module: "LitRecommenderBase"):
        self.common_on_epoch_end(pl_module=pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module: "LitRecommenderBase"):
        self.common_on_epoch_end(pl_module=pl_module, stage="test")

    def common_on_epoch_end(self, pl_module, stage: 'Literal["train", "val", "test"]'):
        model: "CatboostInterface" = pl_module.model
        if stage == "train":
            user_item_dataframe, label = model.full_train_dataframe_label()
        elif stage == "val":
            user_item_dataframe, label = model.train_user_item_dataframe_label(
                explicit=pl_module.val_explicit()
            )
        elif stage == "test":
            user_item_dataframe, label = model.train_user_item_dataframe_label(
                explicit=pl_module.test_explicit()
            )
        else:
            raise ValueError(f"Unknown stage {stage}")

        wandb.log({f"{stage} dataframe size": float(len(user_item_dataframe))})

        pool_kwargs = model.build_pool_kwargs(
            user_item_dataframe=user_item_dataframe, label=label
        )
        pool = model.pool(**pool_kwargs)
        self.log_feature_importance(
            catboost_model=model.model,
            pool=pool,
            stage=stage,
        )

        shap_values, expected_value, features = model.shap(
            user_item_dataframe=user_item_dataframe, label=label
        )
        self.log_shap_plots(
            shap_values=shap_values,
            expected_value=expected_value,
            features=features,
            stage=stage,
            plot_dependence=self.plot_dependence,
        )

        self.log_single_user_features_dataframe_with_ratings_and_explicit(
            model=model, dataframe=user_item_dataframe, stage=stage, label=label
        )

    @staticmethod
    def log_feature_names(model: "CatboostInterface"):
        cat_features = {k.value: list(v) for k, v in model.cat_features.items()}
        text_features = {k.value: list(v) for k, v in model.text_features.items()}
        non_numeric_features = sum(cat_features.values(), start=[]) + sum(
            text_features.values(), start=[]
        )
        numeric_features = {}
        for kind in model.FeatureKind:
            features = model.features(kind)
            numeric_features[kind.value] = list(
                features.columns.difference(non_numeric_features)
            )
        features_description = pd.DataFrame(
            dict(
                cat_features=cat_features,
                text_features=text_features,
                numeric_features=numeric_features,
            )
        ).reset_index()
        print(features_description)
        wandb.log(dict(features=wandb.Table(dataframe=features_description)))

    @staticmethod
    def log_single_user_features_dataframe_with_ratings_and_explicit(
        model: "CatboostInterface", dataframe: pd.DataFrame, label: np.array, stage: str
    ):
        user_id = dataframe["user_ids"].sample(1).values[0]
        indices = dataframe.query(f"user_ids == @user_id").index.values

        user_dataframe = dataframe.iloc[indices].copy()
        user_label = label[indices]
        user_pool_kwargs = model.build_pool_kwargs(
            user_item_dataframe=user_dataframe, label=user_label
        )
        user_pool = model.pool(**user_pool_kwargs)
        user_dataframe["ratings"] = model.model.predict(user_pool)
        with filter_warnings(
            action="ignore", category=pd.errors.SettingWithCopyWarning
        ):
            user_dataframe["explicit"] = user_label
        wandb.log({f"{stage} user dataframe": wandb.Table(dataframe=user_dataframe)})

    @staticmethod
    def log_feature_importance(
        catboost_model: "catboost.CatBoost", pool: "catboost.Pool", stage: str
    ):
        dataframe = catboost_model.get_feature_importance(pool, prettified=True)
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
        shap_values,
        expected_value: float,
        features: pd.DataFrame,
        stage: str,
        plot_dependence=False,
    ):
        prefix = f"{stage} shap "
        textio = self.force_plot(
            shap_values=shap_values, expected_value=expected_value, features=features
        )
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
        every_val_epoch=1,
        every_test_epoch=1,
    ):
        self.k = k
        self.every_epoch = dict(val=every_val_epoch, test=every_test_epoch)
        self.metrics = dict(val=None, test=None)
        self.log_dict = wandb.log

    def setup(self, trainer, pl_module, stage=None):
        if isinstance(trainer.datamodule, SparseDataModuleInterface):
            module = trainer.datamodule
        elif isinstance(pl_module, SparseDataModuleInterface):
            module = pl_module
        else:
            raise ValueError(
                f"One of lightning_module, datamodule must implement "
                f"the {SparseDataModuleInterface.__name__} interface."
            )
        self.log_dict = pl_module.log_dict
        for stage in ["val", "test"]:
            explicit = getattr(module, f"{stage}_explicit")()
            if explicit is not None:
                self.metrics[stage] = RecommendingMetrics(explicit)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.common_on_batch_end(pl_module=pl_module, batch=batch, stage="val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.common_on_batch_end(pl_module=pl_module, batch=batch, stage="test")

    def common_on_batch_end(self, pl_module, batch, stage):
        every_epoch = self.every_epoch[stage]
        if every_epoch == 0 or pl_module.current_epoch % every_epoch:
            return
        user_ids = batch["user_ids"]
        model = pl_module.model
        if user_ids is None:
            user_ids = torch.arange(model.n_users)
        recommendations = model.recommend(user_ids=user_ids, n_recommendations=self.k)
        metrics_dict = self.metrics[stage].batch_metrics(
            user_ids=user_ids, recommendations=recommendations
        )
        self.log_metrics(metrics_dict=metrics_dict, stage=stage)

    def log_metrics(self, metrics_dict: dict, stage: Literal["val", "test"]):
        metrics_dict = {stage + "_" + k: v for k, v in metrics_dict.items()}
        for metric_name in metrics_dict:
            wandb.define_metric(metric_name, summary="mean")
        self.log_dict(metrics_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.common_epoch_end(stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.common_epoch_end(stage="test")

    def common_epoch_end(self, stage: Literal["val", "test"]):
        metrics = self.metrics[stage]
        if metrics is not None and len(metrics.unique_recommended_items):
            metrics = metrics.finalize_coverage()
            self.log_metrics(metrics, stage)


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

    def on_test_epoch_end(self, trainer, pl_module):
        self.my_on_epoch_end(model=pl_module.model, stage="test")

    def on_validation_epoch_end(self, trainer, pl_module):
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


class WandbSaveCodeOnceCallback(pl.callbacks.Callback):
    def __init__(
        self, root=".", name=None, include_regex=".*\\.py", exclude_regex=None
    ):
        self.root = root
        self.name = name
        self.include_regex = include_regex
        self.exclude_regex = exclude_regex
        self.have_saved_code = False

    def include_fn(self, path_to_code):
        return bool(re.fullmatch(pattern=self.include_regex, string=path_to_code))

    def exclude_fn(self, path_to_code):
        if self.exclude_regex is None:
            return False
        return bool(re.fullmatch(pattern=self.exclude_regex, string=path_to_code))

    def setup(self, trainer, pl_module, stage=None):
        if not self.have_saved_code:
            wandb.run.log_code(
                root=self.root,
                name=self.name,
                include_fn=self.include_fn,
                exclude_fn=self.exclude_fn,
            )
            self.have_saved_code = True


class WandbProfiler(profiler.SimpleProfiler):
    def __init__(self, artifact_name=None, dirpath=None, filename=None, extended=True):
        self.artifact_name = artifact_name or "unnamed"
        super().__init__(
            dirpath=dirpath or "local",
            filename=filename or "profiler_logs",
            extended=extended,
        )

    def get_path(self, stage):
        return os.path.join(self.dirpath, f"{stage}-{self.filename}.txt")

    def teardown(self, stage=None):
        super().teardown(stage=stage)
        path = self.get_path(stage=stage)
        if os.path.exists(path) and wandb.run is not None:
            html = wandb.Html(f"<pre>{open(path).read()}</pre>")
            wandb.log({f"profiler/{stage}": html})
            os.remove(path)


class ParameterStatsLoggerCallback(pl.callbacks.Callback):
    def __init__(self, every_train_batch=1):
        self.every_train_batch = every_train_batch

    @staticmethod
    def parameter_stats(parameter):
        stats = dict(
            numel=float(parameter.numel()),
            mean=parameter.mean(),
            isnan_sum=float(parameter.isnan().sum()),
            abs_max=parameter.abs().max(),
            norm=parameter.norm(),
        )
        if (grad := parameter.grad) is not None:
            stats["grad_norm"] = grad.detach().norm()
        return stats

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_train_batch == 0:
            for parameter_name, parameter in pl_module.named_parameters():
                for stat_name, stat_value in self.parameter_stats(parameter).items():
                    pl_module.log(f"parameter/{stat_name}_{parameter_name}", stat_value)
