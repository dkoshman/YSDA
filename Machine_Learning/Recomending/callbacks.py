import string
from typing import Literal

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy
import torch
import wandb

from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
import matplotlib.patheffects as pe

from data import MovieLens
from utils import scipy_coo_to_torch_sparse, WandbAPI


class WandbWatcher(pl.callbacks.Callback):
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
        self.watch_model_triggered = False

    def setup(self, trainer, pl_module, stage=None):
        if self.watch_model_triggered:
            return
        if not isinstance(trainer.logger, pl.loggers.WandbLogger):
            raise ValueError("Only wandb logger supports watching model.")
        trainer.logger.watch(pl_module, **self.watch_kwargs)
        self.watch_model_triggered = True

    def teardown(self, trainer, pl_module, stage=None):
        trainer.logger.experiment.unwatch(pl_module)


class RecommendingDataOverviewCallback(pl.callbacks.Callback):
    def __init__(self):
        self.explicit = None
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


class ImdbRatings:
    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        path_to_movielens_folder="local/ml-100k",
    ):
        self.path_to_imdb_ratings_csv = path_to_imdb_ratings_csv
        self.movielens = MovieLens(path_to_movielens_folder)

    @property
    def imdb_ratings(self):
        return pd.read_csv(self.path_to_imdb_ratings_csv)

    @staticmethod
    def normalize_titles(titles):
        titles = (
            titles.str.lower()
            .str.normalize("NFC")
            .str.replace(f"[{string.punctuation}]", "", regex=True)
            .str.replace("the", "")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return titles

    def explicit_feedback_scipy(self):
        imdb_ratings = self.imdb_ratings
        imdb_ratings["Title"] = self.normalize_titles(imdb_ratings["Title"])
        imdb_ratings["Your Rating"] = (imdb_ratings["Your Rating"] + 1) // 2

        movielens_titles = self.movielens["u.item"]["movie title"]
        movielens_titles = self.normalize_titles(movielens_titles.str.split("(").str[0])

        liked_items_ids = []
        liked_items_ratings = []
        for title, rating in zip(imdb_ratings["Title"], imdb_ratings["Your Rating"]):
            for movie_id, ml_title in movielens_titles.items():
                if title == ml_title:
                    liked_items_ids.append(movie_id - 1)
                    liked_items_ratings.append(rating)

        data = liked_items_ratings
        row = np.zeros(len(liked_items_ids))
        col = np.array(liked_items_ids)
        shape = (1, self.movielens.shape[1])
        explicit_feedback = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
        return explicit_feedback

    def explicit_feedback_torch(self):
        return scipy_coo_to_torch_sparse(self.explicit_feedback_scipy())


class WandbCheckpointCallback(pl.callbacks.ModelCheckpoint):
    def __init__(self, *args, artifact_name, description=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_name = artifact_name
        self.description = description

    def on_test_end(self, trainer, pl_module):
        if run := wandb.run:
            if not self.best_model_path:
                self.save_checkpoint(trainer)
            WandbAPI().save_checkpoint(
                checkpoint_path=self.best_model_path,
                pl_module_class=pl_module.__class__.__name__,
                artifact_name=self.artifact_name,
                description=self.description,
                metadata=run.config.as_dict(),
            )
        super().on_test_end(trainer, pl_module)


class CatBoostMetrics(pl.callbacks.Callback):
    def on_test_epoch_end(self, trainer=None, pl_module=None):
        self.log_feature_importance(pl_module)

    def log_feature_importance(self, pl_module):
        cb_module = pl_module.model
        cb_model = cb_module.model
        user_ids = torch.arange(min(100, pl_module.train_explicit.shape[0]))
        for explicit, type in zip(
            [pl_module.train_explicit, pl_module.test_explicit], ["train", "test"]
        ):
            dataframe = cb_module.topk_data(
                user_ids=user_ids,
                users_explicit_feedback=explicit[user_ids],
            )
            pool = cb_module.pool(dataframe, training=True)
            feature_importance = cb_model.get_feature_importance(pool)
            dataframe = (
                pd.Series(feature_importance, cb_model.feature_names_)
                .sort_values(ascending=False)
                .to_frame()
                .T
            )
            wandb.log(
                {
                    f"Catboost {type} feature importance": wandb.Table(
                        dataframe=dataframe
                    )
                }
            )
