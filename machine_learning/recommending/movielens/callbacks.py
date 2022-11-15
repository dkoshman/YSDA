from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
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
        movielens_model_trains_on_directory="local/ml-100k",
        movielens_model_trains_on_class_name="MovieLens100k",
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        movielens25m_directory="local/ml-25m",
        n_recommendations=10,
        ratings_scale_max_to_convert_to=5,
    ):
        self.imdb_ratings = read_csv_imdb_ratings(
            path_to_imdb_ratings_csv=path_to_imdb_ratings_csv,
            ratings_scale_max_to_convert_to=ratings_scale_max_to_convert_to,
        )
        self.movielens = self.build_class(
            class_name=movielens_model_trains_on_class_name,
            directory=movielens_model_trains_on_directory,
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

    @property
    def class_candidates(self):
        return [MovieLens100k, MovieLens25m]

    def items_description(self, item_ids: np.array) -> pd.DataFrame:
        movielens_ids = self.movielens25m.model_item_to_movielens_movie_ids(item_ids)
        description = self.movielens25m["movies"].loc[movielens_ids]
        return description

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
        items_description = self.items_description(recommendations)
        title = f"{stage} recommendations based on imdb ratings"
        wandb.log({title: wandb.Table(dataframe=items_description)})
