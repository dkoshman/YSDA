import os.path
import re
from typing import Literal, TYPE_CHECKING, Dict

import catboost
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import shap
import wandb

from matplotlib import pyplot as plt
from pytorch_lightning import loggers, profiler
from scipy.sparse import coo_array

from my_tools.utils import torch_sparse_slice
from .data import SparseDataModuleInterface
from .interface import ExplanationMixin
from .metrics import RecommendingMetrics
from .utils import wandb_plt_figure, filter_warnings, save_shap_force_plot, Timer

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

    def __init__(
        self,
        *args,
        plot_dependence=False,
        max_dataset_size_for_complex_shap=1000,
        figsize=(8, 5),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.plot_dependence = plot_dependence
        self.max_dataset_size_for_complex_shap = max_dataset_size_for_complex_shap
        self.figsize = figsize

    def on_train_end(self, trainer, pl_module: "LitRecommenderBase"):
        self.log_feature_names(model=pl_module.model)
        self.common_on_end(pl_module=pl_module, stage="train")

    def on_validation_end(self, trainer, pl_module: "LitRecommenderBase"):
        self.common_on_end(pl_module=pl_module, stage="val")

    def on_test_end(self, trainer, pl_module: "LitRecommenderBase"):
        self.common_on_end(pl_module=pl_module, stage="test")

    @Timer()
    def common_on_end(self, pl_module, stage: 'Literal["train", "val", "test"]'):
        model: "CatboostInterface" = pl_module.model
        if stage == "train":
            user_item_dataframe = model.train_user_item_dataframe
        elif stage == "val":
            user_item_dataframe = model.build_labeled_user_item_dataframe(
                explicit=pl_module.val_explicit()
            )
        elif stage == "test":
            user_item_dataframe = model.build_labeled_user_item_dataframe(
                explicit=pl_module.test_explicit()
            )
        else:
            raise ValueError(f"Unknown stage {stage}")

        wandb.log(
            {
                f"{stage} dataframe size": float(len(user_item_dataframe)),
                f"shap expected value": model.shap_explainer.expected_value,
            }
        )

        self.log_single_user_features_dataframe_with_ratings_and_explicit(
            model=model, dataframe=user_item_dataframe, stage=stage
        )

        pool = model.pool(user_item_dataframe=user_item_dataframe)
        self.log_feature_importance(catboost_model=model.model, pool=pool, stage=stage)

        shap_kwargs = model.shap_kwargs(user_item_dataframe=user_item_dataframe)

        self.log_shap_summary_plot(**shap_kwargs, stage=stage, figsize=self.figsize)

        if self.plot_dependence:
            self.log_shap_dependence_plots(
                **shap_kwargs, stage=stage, figsize=self.figsize
            )

        if len(user_item_dataframe) > self.max_dataset_size_for_complex_shap:
            user_item_dataframe = user_item_dataframe.sample(
                self.max_dataset_size_for_complex_shap
            ).sort_index()
            shap_kwargs = model.shap_kwargs(user_item_dataframe=user_item_dataframe)

        self.log_shap_force_plot(**shap_kwargs, stage=stage)

        dataframe = model.merge_features(user_item_dataframe)
        pool_kwargs = model.extract_pool_kwargs(dataframe=dataframe)
        postprocessed_pool_kwargs = model.postprocess_pool_kwargs(**pool_kwargs)
        shap_explanation = model.shap_explainer(X=postprocessed_pool_kwargs["data"])
        self.log_shap_heatmap_plot(
            shap_explanation=shap_explanation, stage=stage, figsize=self.figsize
        )

    @staticmethod
    def log_feature_names(model: "CatboostInterface"):
        dataframe = model.train_user_item_dataframe
        dataframe = model.merge_features(dataframe)
        pool_kwargs = model.extract_pool_kwargs(dataframe=dataframe)
        cat_features = pool_kwargs["cat_features"]
        text_features = pool_kwargs["text_features"]
        numeric_features = (
            pool_kwargs["dataframe"]
            .columns.difference(cat_features)
            .difference(text_features)
        ).tolist()
        features_description = pd.Series(
            dict(
                cat_features=cat_features,
                text_features=text_features,
                numeric_features=numeric_features,
            )
        )
        features_description = features_description.to_frame().T
        wandb.log(dict(features=wandb.Table(dataframe=features_description)))

    @staticmethod
    def log_single_user_features_dataframe_with_ratings_and_explicit(
        model: "CatboostInterface", dataframe: pd.DataFrame, stage: str
    ):
        user_id = dataframe.reset_index()["user_id"].sample(1).values[0]
        offline_dataframe = dataframe.query(
            f"user_id == @user_id", local_dict=dict(user_id=user_id)
        )
        user_explicit = coo_array(
            (
                offline_dataframe["rating"].values,
                (
                    np.zeros(len(offline_dataframe)),
                    offline_dataframe.reset_index()["item_id"].values,
                ),
            ),
            shape=[1, model.n_items],
        )
        online_dataframe = model.build_recommend_user_item_dataframe(
            users_explicit=model.to_torch_coo(user_explicit),
            user_ids=model.fictive_user_ids(1),
            n_recommendations=len(offline_dataframe),
        )
        for kind, dataframe in zip(
            ["offline", "online"], [offline_dataframe, online_dataframe]
        ):
            dataframe = model.merge_features(dataframe)
            pool_kwargs = model.extract_pool_kwargs(dataframe=dataframe)
            postprocessed_pool_kwargs = model.postprocess_pool_kwargs(**pool_kwargs)
            pool = catboost.Pool(**postprocessed_pool_kwargs)
            features_dataframe = pool_kwargs["dataframe"]
            with filter_warnings(
                action="ignore", category=pd.errors.SettingWithCopyWarning
            ):
                features_dataframe["ratings"] = model.model.predict(pool)
            features_dataframe = features_dataframe.sort_values(
                "ratings", ascending=False
            )
            wandb.log(
                {
                    f"{stage} {kind} user dataframe": wandb.Table(
                        dataframe=features_dataframe
                    )
                }
            )

    @staticmethod
    def log_feature_importance(
        catboost_model: "catboost.CatBoost", pool: "catboost.Pool", stage: str
    ):
        dataframe = catboost_model.get_feature_importance(pool, prettified=True)
        wandb_table = wandb.Table(dataframe=dataframe)
        wandb.log({f"Catboost {stage} feature importance": wandb_table})

    @staticmethod
    def log_shap_force_plot(
        base_value: float, shap_values: np.array, features: pd.DataFrame, stage: str
    ):
        shap_plot = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values,
            features=features,
            out_names="Predicted relevance of items for users",
            text_rotation=0,
        )
        textio = save_shap_force_plot(shap_plot=shap_plot)
        wandb.log({f"shap/{stage} force plot": wandb.Html(textio)})

    @staticmethod
    def log_shap_dependence_plots(
        base_value: float,
        shap_values: np.array,
        features: pd.DataFrame,
        stage: str,
        figsize,
    ):
        for feature_name in list(features):
            with wandb_plt_figure(
                title=f"shap/dependence plot/{stage} " + feature_name, figsize=figsize
            ) as figure:
                shap.dependence_plot(
                    feature_name,
                    shap_values=shap_values,
                    features=features,
                    ax=figure.gca(),
                )

    @staticmethod
    def log_shap_summary_plot(
        base_value: float,
        shap_values: np.array,
        features: pd.DataFrame,
        stage: str,
        figsize,
    ):
        with wandb_plt_figure(title=f"shap/{stage} summary plot", figsize=figsize):
            shap.summary_plot(shap_values=shap_values, features=features)

    @staticmethod
    def log_shap_heatmap_plot(shap_explanation: shap.Explanation, stage: str, figsize):
        with wandb_plt_figure(title=f"shap/{stage} heatmap plot", figsize=figsize):
            shap.plots.heatmap(shap_values=shap_explanation)


class RecommendingMetricsCallback(pl.callbacks.Callback):
    def __init__(
        self,
        k: int = 10,
        every_val_epoch=1,
        every_test_epoch=1,
        assess_online_recommendations=False,
    ):
        self.k = k
        self.every_epoch = dict(val=every_val_epoch, test=every_test_epoch)
        self.assess_online_recommendations = assess_online_recommendations
        self.metrics: "Dict[str, RecommendingMetrics]" = dict()
        self.log_dict = wandb.log

    def skip_epoch(self, current_epoch, stage) -> bool:
        every_epoch = self.every_epoch[stage]
        return every_epoch == 0 or current_epoch % every_epoch

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
            for is_online in [True, False]:
                if explicit is not None and (
                    not is_online or self.assess_online_recommendations
                ):
                    kind = self.kind(stage=stage, is_online=is_online)
                    self.metrics[kind] = RecommendingMetrics(explicit)

    @staticmethod
    def kind(stage: str, is_online: bool) -> str:
        assert stage in ["val", "test"], f"Unknown stage {stage}."
        return stage + "_" + ("online" if is_online else "offline")

    @Timer()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.common_on_batch_end(pl_module=pl_module, batch=batch, stage="val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.common_on_batch_end(pl_module=pl_module, batch=batch, stage="test")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.common_epoch_end(stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.common_epoch_end(stage="test")

    def common_on_batch_end(self, pl_module, batch, stage):
        if self.skip_epoch(current_epoch=pl_module.current_epoch, stage=stage):
            return
        user_ids = batch["user_ids"]
        model = pl_module.model
        for is_online in [True, False]:
            kind = self.kind(stage=stage, is_online=is_online)
            if (metrics := self.metrics.get(kind)) is not None:
                if is_online:
                    users_explicit = torch_sparse_slice(
                        model.explicit, row_ids=user_ids
                    )
                    recommendations = model.online_recommend(
                        users_explicit=users_explicit, n_recommendations=self.k
                    )
                else:
                    recommendations = model.recommend(
                        user_ids=user_ids, n_recommendations=self.k
                    )
                metrics_dict = metrics.batch_metrics(
                    user_ids=user_ids, recommendations=recommendations
                )
                self.log_metrics(metrics_dict=metrics_dict, prefix=kind)

    def log_metrics(self, metrics_dict: dict, prefix: str):
        metrics_dict = {prefix + "_" + k: v for k, v in metrics_dict.items()}
        for metric_name in metrics_dict:
            wandb.define_metric(metric_name, summary="mean")
        self.log_dict(metrics_dict)

    def common_epoch_end(self, stage):
        for is_online in [True, False]:
            kind = self.kind(stage=stage, is_online=is_online)
            if (metrics := self.metrics.get(kind)) is not None and len(
                metrics.unique_recommended_items
            ):
                metrics = metrics.finalize_coverage()
                self.log_metrics(metrics, prefix=kind)


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
                model.explain_recommendations_for_user(
                    user_id=user_id,
                    n_recommendations=self.n_recommendations,
                    log=True,
                    logging_prefix=f"explanation/{stage} ",
                )
        if self.users_explicit is not None:
            try:
                for user_explicit in self.users_explicit.detach().clone():
                    model.explain_recommendations_for_user(
                        user_explicit=user_explicit[None],
                        n_recommendations=self.n_recommendations,
                        log=True,
                        logging_prefix=f"explanation/{stage} ",
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

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, **kwargs
    ):
        if batch_idx % self.every_train_batch == 0:
            for parameter_name, parameter in pl_module.named_parameters():
                for stat_name, stat_value in self.parameter_stats(parameter).items():
                    pl_module.log(f"parameter/{stat_name}_{parameter_name}", stat_value)
