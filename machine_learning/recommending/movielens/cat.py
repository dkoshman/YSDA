import pandas as pd

from ..models.cat import CatboostAggregatorFromArtifacts, CatboostRecommenderBase
from .data import MovieLens100k, MovieLens25m


class CatboostMovieLens100kFeatureRecommender(CatboostRecommenderBase):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens100k(movielens_directory)

    def user_features(self):
        user_features = self.movielens["u.user"].reset_index(names="user_id")
        user_features["user_id"] = self.movielens.dataset_to_dense_user_ids(
            dataset_user_ids=user_features["user_id"].values
        )
        return self.FeatureReturnValue(
            dataframe=user_features,
            cat_features={"user_id", "gender", "occupation", "zip code"},
        )

    def item_features(self):
        item_features = (
            self.movielens["u.item"]
            .reset_index(names="item_id")
            .drop(["movie title", "video release date", "IMDb URL"], axis="columns")
        )
        item_features["release date"] = pd.to_datetime(
            item_features["release date"]
        ).astype(int)
        return self.FeatureReturnValue(
            dataframe=item_features,
            cat_features=set(item_features.columns.drop(["release date"])),
        )

    def user_item_features(self):
        user_item_features = self.movielens["u.data"][
            ["user_id", "item_id", "timestamp"]
        ]
        return self.FeatureReturnValue(dataframe=user_item_features)


class CatboostMovieLens100kFeatureAggregatorFromArtifacts(
    CatboostAggregatorFromArtifacts, CatboostMovieLens100kFeatureRecommender
):
    pass


class CatboostMovieLens25mFeatureRecommender(CatboostRecommenderBase):
    def __init__(self, *, movielens_directory, **kwargs):
        super().__init__(**kwargs)
        self.movielens = MovieLens25m(movielens_directory)

    def item_features(self):
        item_features = self.movielens["movies"].reset_index(names="item_id")
        item_features = item_features.query(
            "item_id in @rated_item_ids",
            local_dict=dict(rated_item_ids=self.movielens.unique_dataset_item_ids),
        )
        genres = item_features["genres"].str.get_dummies()
        item_features = item_features.drop(["title", "genres"], axis="columns")
        item_features = item_features.join(genres)
        item_features["item_id"] = self.movielens.dataset_to_dense_item_ids(
            dataset_item_ids=item_features["item_id"].values
        )
        return self.FeatureReturnValue(
            dataframe=item_features,
            cat_features=set(item_features.columns),
        )

    def user_item_features(self):
        user_item_features = self.movielens["ratings"][
            ["user_id", "item_id", "timestamp"]
        ]
        user_item_features = self.set_index(
            user_item_features, kind=self.FeatureKind.user_item
        )
        user_item_tags = (
            self.movielens["tags"]
            .groupby(["user_id", "item_id"])["tag"]
            .apply(", ".join)
        )
        user_item_features = user_item_features.join(user_item_tags)
        return self.FeatureReturnValue(
            dataframe=user_item_features,
            text_features={"tag"},
        )


class CatboostMovieLens25mFeatureAggregatorFromArtifacts(
    CatboostAggregatorFromArtifacts, CatboostMovieLens25mFeatureRecommender
):
    pass
