from typing import Literal

import wandb

from .data import ImdbRatings
from ..callbacks import RecommendingExplanationCallback
from ..interface import RecommenderModuleBase


class RecommendingExplanationIMDBCallback(RecommendingExplanationCallback):
    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        path_to_movielens_folder="local/ml-100k",
        n_recommendations=5,
    ):
        self.imdb = ImdbRatings(path_to_imdb_ratings_csv, path_to_movielens_folder)
        super().__init__(
            users_explicit=self.imdb.explicit_feedback_torch(),
            n_recommendations=n_recommendations,
        )

    def my_on_epoch_end(
        self,
        model: RecommenderModuleBase,
        stage: Literal["train", "val", "test", "predict"],
    ):
        super().my_on_epoch_end(model=model, stage=stage)
        recommendations = model.online_recommend(
            users_explicit=self.users_explicit,
            n_recommendations=self.n_recommendations,
        )
        recommendations = recommendations.cpu().numpy()
        for i, recs in enumerate(recommendations):
            items_description = self.imdb.items_description(recs)
            title = f"{stage}, recommendations for user {i}"
            wandb.log({title: wandb.Table(dataframe=items_description.T)})
