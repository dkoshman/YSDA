import pytorch_lightning as pl
import wandb

from .data import ImdbRatings


class RecommendingIMDBCallback(pl.callbacks.Callback):
    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        path_to_movielens_folder="local/ml-100k",
        n_recommendations=10,
    ):
        self.imdb = ImdbRatings(path_to_imdb_ratings_csv, path_to_movielens_folder)
        self.n_recommendations = n_recommendations

    def on_test_epoch_end(self, trainer=None, pl_module=None):
        recommendations = pl_module.model.online_recommend(
            users_explicit=self.imdb.explicit_feedback_torch(),
            n_recommendations=self.n_recommendations,
        )
        self.log_recommendation(recommendations.cpu().numpy())

    def on_validation_epoch_end(self, trainer=None, pl_module=None):
        recommendations = pl_module.model.online_recommend(
            users_explicit=self.imdb.explicit_feedback_torch(),
            n_recommendations=self.n_recommendations,
        )
        self.log_recommendation(recommendations.cpu().numpy())

    def log_recommendation(self, recommendations):
        for i, recs in enumerate(recommendations):
            items_description = self.imdb.items_description(recs)
            wandb.log(
                {
                    f"recommended_items_user_{i}": wandb.Table(
                        dataframe=items_description.T
                    )
                }
            )
