from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import wandb

from my_tools.utils import BuilderMixin
from .data import (
    read_csv_imdb_ratings,
    MovieLens25m,
    explicit_from_imdb_ratings,
    MovieLens100k,
)
from ..callbacks import RecommendingExplanationCallback


if TYPE_CHECKING:
    from ..interface import RecommenderModuleBase


class RecommendingExplanationIMDBCallback(
    RecommendingExplanationCallback, BuilderMixin
):
    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        movielens25m_directory="local/ml-25m",
        movielens_model_trains_on_class_name="MovieLens25m",
        movielens_model_trains_on_directory=None,
        n_recommendations=100,
        ratings_scale_max_to_convert_to=5,
    ):
        """If movielens_model_trains is not passed, it is assumed to be MovieLens25m."""
        self.imdb_ratings = read_csv_imdb_ratings(
            path_to_imdb_ratings_csv=path_to_imdb_ratings_csv,
            ratings_scale_max_to_convert_to=ratings_scale_max_to_convert_to,
        )
        self.movielens = self.build_class(
            class_name=movielens_model_trains_on_class_name or "",
            directory=movielens_model_trains_on_directory or movielens25m_directory,
        )
        self.movielens25m = MovieLens25m(directory=movielens25m_directory)
        users_explicit = explicit_from_imdb_ratings(
            imdb_ratings=self.imdb_ratings,
            movielens_the_model_trained_on=self.movielens,
            movielens_25m=self.movielens25m,
        )
        super().__init__(
            n_recommendations=n_recommendations, users_explicit=users_explicit
        )

    def on_train_start(self, trainer, pl_module):
        wandb.log(dict(imdb_ratings=wandb.Table(dataframe=self.imdb_ratings)))

    @property
    def class_candidates(self):
        return [MovieLens100k, MovieLens25m]

    def my_on_epoch_end(
        self,
        model: "RecommenderModuleBase",
        stage: 'Literal["train", "val", "test", "predict"]',
    ):
        super().my_on_epoch_end(model=model, stage=stage)
        recommendations = model.online_recommend(
            users_explicit=self.users_explicit,
            n_recommendations=self.n_recommendations,
        )
        recommendations = recommendations.cpu().numpy()[0]
        items_description = self.recommendations_description(recommendations)
        title = f"{stage} recommendations based on imdb ratings"
        wandb.log({title: wandb.Table(dataframe=items_description)})

        _, indices = torch.topk(
            self.users_explicit.to_dense()[0], k=self.n_recommendations
        )
        dataframe = self.recommendations_description(recommendations=indices.numpy())
        title = "sanity_check/imdb_as_recommendations"
        wandb.log({title: wandb.Table(dataframe=dataframe)})

        try:
            online_ratings = model.online_ratings(users_explicit=self.users_explicit)
        except NotImplementedError:
            pass
        else:
            _, indices = torch.topk(online_ratings[0], k=self.n_recommendations)
            dataframe = self.recommendations_description(
                recommendations=indices.cpu().numpy()
            )
            title = "sanity_check/unfiltered_recommendations"
            wandb.log({title: wandb.Table(dataframe=dataframe)})

    def recommendations_description(self, recommendations: np.array) -> pd.DataFrame:
        assert recommendations.ndim == 1
        movielens_ids = self.movielens.dense_item_to_movielens_movie_ids(
            recommendations
        )
        assert all(
            self.movielens.movielens_movie_to_dense_item_ids(movielens_ids)
            == recommendations
        )
        description = pd.DataFrame(dict(movielens_id=movielens_ids))
        known_movielens_ids = np.intersect1d(
            self.movielens25m.unique_movielens_movie_ids, movielens_ids
        )
        imdb_description = self.movielens25m["movies"].loc[known_movielens_ids]
        description = pd.merge(
            description,
            imdb_description,
            how="left",
            left_on="movielens_id",
            right_index=True,
        )
        if len(known_movielens_ids) < len(movielens_ids) and isinstance(
            self.movielens, MovieLens100k
        ):
            unknown_movielens_ids = np.setdiff1d(movielens_ids, known_movielens_ids)
            ml100k_description = self.movielens["u.item"].loc[unknown_movielens_ids]
            description = pd.merge(
                description,
                ml100k_description,
                how="left",
                left_on="movielens_id",
                right_index=True,
            )
        return description
