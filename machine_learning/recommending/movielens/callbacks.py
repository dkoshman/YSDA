from typing import Literal, TYPE_CHECKING

import torch
import wandb

from my_tools.utils import BuilderMixin
from .data import ImdbRatings, MovieLens25m, MovieLens100k
from ..callbacks import RecommendingExplanationCallback


if TYPE_CHECKING:
    from ..interface import RecommenderModuleBase


class RecommendingExplanationIMDBCallback(
    RecommendingExplanationCallback, BuilderMixin
):
    def __init__(
        self,
        path_to_imdb_ratings_csv: str,
        movielens_class_name: str,
        movielens_directory: str,
        n_recommendations=10,
        ratings_scale_max_to_convert_to=5,
    ):
        """If movielens_model_trains is not passed, it is assumed to be MovieLens25m."""
        self.imdb_ratings = ImdbRatings(
            path_to_imdb_ratings_csv=path_to_imdb_ratings_csv,
            ratings_scale_max_to_convert_to=ratings_scale_max_to_convert_to,
        )
        self.movielens = self.build_class(
            class_name=movielens_class_name, directory=movielens_directory
        )
        users_explicit = self.imdb_ratings.explicit_movielens(self.movielens)
        super().__init__(
            n_recommendations=n_recommendations, users_explicit=users_explicit
        )

    @property
    def class_candidates(self):
        return [MovieLens100k, MovieLens25m]

    def items_description(self, dense_item_ids):
        return self.imdb_ratings.items_description_movielens(
            dense_item_ids=dense_item_ids, movielens=self.movielens
        )

    def on_train_start(self, trainer, pl_module):
        wandb.log(dict(imdb_ratings=wandb.Table(dataframe=self.imdb_ratings.dataframe)))
        _, indices = torch.topk(
            self.users_explicit.to_dense()[0], k=self.n_recommendations
        )
        dataframe = self.items_description(dense_item_ids=indices.numpy())
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
        items_description = self.items_description(dense_item_ids=recommendations)
        title = f"{stage} recommendations based on imdb ratings"
        wandb.log({title: wandb.Table(dataframe=items_description)})

        try:
            online_ratings = model.online_ratings(users_explicit=self.users_explicit)
        except NotImplementedError:
            pass
        else:
            _, indices = torch.topk(online_ratings[0], k=self.n_recommendations)
            dataframe = self.items_description(dense_item_ids=indices.cpu().numpy())
            title = f"sanity_check/{stage} unfiltered recommendations"
            wandb.log({title: wandb.Table(dataframe=dataframe)})
