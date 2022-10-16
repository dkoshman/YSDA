from typing import Literal

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from ..metrics import RecommendingMetrics
from .data import ImdbRatings, MovieLens


class RecommendingMetricsCallback(pl.callbacks.Callback):
    def __init__(self, directory, k=10):
        movielens = MovieLens(directory)
        explicit_feedback = movielens.explicit_feedback_scipy_csr("u.data")
        if isinstance(k, int):
            k = [k]
        self.metrics = [RecommendingMetrics(explicit_feedback, k1) for k1 in k]

    @torch.inference_mode()
    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_batch(
            user_ids=batch["user_ids"], ratings=pl_module(**batch), kind="test"
        )

    @torch.inference_mode()
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_batch(
            user_ids=batch["user_ids"], ratings=pl_module(**batch), kind="val"
        )

    def log_batch(self, user_ids, ratings, kind: Literal["val", "test"]):
        for atk_metric in self.metrics:
            metrics = atk_metric.batch_metrics_from_ratings(
                user_ids=user_ids, ratings=ratings
            )
            self.log_metrics(metrics, kind=kind)

    def log_metrics(self, metrics: dict, kind: Literal["val", "test"]):
        metrics = {kind + "_" + k: v for k, v in metrics.items()}
        for metric_name in metrics:
            wandb.define_metric(metric_name, summary="mean")
        wandb.log(metrics)

    def on_test_epoch_end(self, trainer=None, pl_module=None):
        self.epoch_end(kind="test")

    def on_validation_epoch_end(self, trainer=None, pl_module=None):
        self.epoch_end(kind="val")

    def epoch_end(self, kind: Literal["val", "test"]):
        for atk_metric in self.metrics:
            if len(atk_metric.unique_recommended_items):
                metrics = atk_metric.finalize_coverage()
                self.log_metrics(metrics, kind)


class RecommendingIMDBCallback(pl.callbacks.Callback):
    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        path_to_movielens_folder="local/ml-100k",
        n_recommendations=10,
    ):
        imdb = ImdbRatings(path_to_imdb_ratings_csv, path_to_movielens_folder)
        self.explicit_feedback = imdb.explicit_feedback_scipy()
        self.item_df = imdb.movielens["u.item"]
        self.n_recommendations = n_recommendations

    def on_test_epoch_end(self, trainer=None, pl_module=None):
        recommendations = pl_module.recommend(
            users_explicit_feedback=self.explicit_feedback,
            n_recommendations=self.n_recommendations,
        )
        self.log_recommendation(recommendations.cpu().numpy())

    def on_validation_epoch_end(self, trainer=None, pl_module=None):
        recommendations = pl_module.recommend(
            users_explicit_feedback=self.explicit_feedback,
            n_recommendations=self.n_recommendations,
        )
        self.log_recommendation(recommendations.cpu().numpy())

    def log_recommendation(self, recommendations):
        recommendations += 1
        for i, recs in enumerate(recommendations):
            items_description = self.item_df.loc[recs]
            imdb_urls = "https://www.imdb.com/find?q=" + items_description[
                "movie title"
            ].str.split("(").str[0].str.replace(r"\s+", "+", regex=True)
            items_description = pd.concat(
                [items_description["movie title"], imdb_urls], axis="columns"
            )
            wandb.log(
                {
                    f"recommended_items_user_{i}": wandb.Table(
                        dataframe=items_description.T
                    )
                }
            )
