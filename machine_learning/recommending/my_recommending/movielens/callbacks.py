from typing import Literal, TYPE_CHECKING

import pandas as pd
import torch
import wandb

from my_tools.utils import BuilderMixin
from .data import MovieLens25m, MovieLens100k, csv_imdb_ratings_to_dataframe
from ..callbacks import RecommendingExplanationCallback
from ..lit import LitRecommenderBase

if TYPE_CHECKING:
    from .data import MovieLensInterface
    from ..interface import RecommenderModuleBase


class RecommendingExplanationIMDBCallback(
    RecommendingExplanationCallback, BuilderMixin
):
    def __init__(
        self,
        path_to_imdb_ratings_csv: str,
        n_recommendations=10,
        ratings_scale_max_to_convert_to=5,
    ):
        super().__init__(n_recommendations=n_recommendations)
        self.path_to_imdb_ratings_csv = path_to_imdb_ratings_csv
        self.ratings_scale_max_to_convert_to = ratings_scale_max_to_convert_to
        self.imdb_ratings: pd.DataFrame or None = None
        self.movielens: MovieLensInterface or None = None

    def setup(self, trainer, pl_module: LitRecommenderBase, stage):
        if self.imdb_ratings is None:
            self.imdb_ratings = csv_imdb_ratings_to_dataframe(
                path_to_imdb_ratings_csv=self.path_to_imdb_ratings_csv,
                ratings_scale_max_to_convert_to=self.ratings_scale_max_to_convert_to,
            )
            self.movielens = pl_module.movielens
            self.users_explicit = self.movielens.imdb_ratings_dataframe_to_explicit(
                imdb_ratings=self.imdb_ratings
            )

    @property
    def class_candidates(self):
        return [MovieLens100k, MovieLens25m]

    def on_train_start(self, trainer, pl_module):
        wandb.log(dict(imdb_ratings=wandb.Table(dataframe=self.imdb_ratings)))
        _, indices = torch.topk(
            self.users_explicit.to_dense()[0], k=self.n_recommendations
        )
        dataframe = self.movielens.items_description(dense_item_ids=indices.numpy())
        title = f"sanity_check/imdb_as_recommendations"
        wandb.log({title: wandb.Table(dataframe=dataframe)})

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
        items_description = self.movielens.items_description(
            dense_item_ids=recommendations
        )
        title = f"{stage} recommendations based on imdb ratings"
        wandb.log({title: wandb.Table(dataframe=items_description)})

        try:
            online_ratings = model.online_ratings(users_explicit=self.users_explicit)
        except NotImplementedError:
            pass
        else:
            _, indices = torch.topk(online_ratings[0], k=self.n_recommendations)
            dataframe = self.movielens.items_description(
                dense_item_ids=indices.cpu().numpy()
            )
            title = f"sanity_check/{stage} unfiltered recommendations"
            wandb.log({title: wandb.Table(dataframe=dataframe)})
