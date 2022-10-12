from collections import Counter

import catboost
import pandas as pd
import torch
from tqdm.auto import tqdm

from .data import MovieLens
from .entrypoints import NonGradientRecommenderMixin, LitRecommenderBase
from .movielens import (
    MovieLensNonGradientRecommender,
    MovieLensPMFRecommender,
    MovieLensSLIMRecommender,
    MovieLensRecommender,
)
from .utils import WandbAPI


class CatboostRecommenderModule(torch.nn.Module):
    def __init__(
        self,
        models,
        model_names=None,
        n_recommendations=10,
        batch_size=100,
        **cb_params,
    ):
        if model_names is not None and len(models) != len(model_names):
            raise ValueError("Models and their names must be of equal length.")
        super().__init__()
        self.n_items = None
        self.models = models
        self.model_names = (
            self.generate_model_names(models) if model_names is None else model_names
        )
        self.n_recommendations = n_recommendations
        self.batch_size = batch_size
        self.cb_params = cb_params
        self.model = catboost.CatBoostRanker(**self.cb_params)

    @staticmethod
    def generate_model_names(models):
        names = []
        counter = Counter()
        for model in models:
            name = model.__class__.__name__
            names.append(f"{name}_{counter[name]}")
            counter.update([name])
        return names

    def skeleton_dataframe(self):
        return pd.DataFrame(
            columns=["user_id", "item_id", "explicit"] + self.model_names
        )

    @staticmethod
    def dataframe_as_type(dataframe, type_to_columns, default_type="float32"):
        leftover_columns = dataframe.columns.drop(
            [i for j in type_to_columns.values() for i in j]
        )
        type_to_columns.update({default_type: leftover_columns})
        reverse = {v: key for key, values in type_to_columns.items() for v in values}
        return dataframe.astype(reverse)

    def topk_data(self, user_ids, users_explicit_feedback=None):
        ratings = []
        topk_item_ids = []
        for model in self.models:
            with torch.inference_mode():
                model_ratings = model(user_ids=user_ids).to_dense()
            ratings.append(model_ratings)
            values, item_ids = torch.topk(model_ratings, k=self.n_recommendations)
            topk_item_ids.append(item_ids)

        per_user_topk_item_ids = []
        for user_id, item_ids in zip(user_ids, torch.cat(topk_item_ids, dim=1)):
            unique_item_ids, counts = item_ids.unique(return_counts=True)
            _, indices = torch.topk(counts, k=self.n_recommendations)
            per_user_topk_item_ids.append(unique_item_ids[indices])
        per_user_topk_item_ids = torch.stack(per_user_topk_item_ids)

        dataframe = self.skeleton_dataframe()
        for i, (user_id, item_ids) in enumerate(zip(user_ids, per_user_topk_item_ids)):
            df = self.skeleton_dataframe()
            df["item_id"] = item_ids.numpy()
            df["user_id"] = user_id.numpy()
            if users_explicit_feedback is not None:
                df["explicit"] = (
                    users_explicit_feedback[i, item_ids].toarray().squeeze()
                )
            item_ratings = [rating[i, item_ids] for rating in ratings]
            df.iloc[:, 3:] = torch.stack(item_ratings, dim=0).T.numpy()
            dataframe = pd.concat([dataframe, df])

        return dict(
            dataframe=dataframe,
            cat_features=["user_id", "item_id"],
            text_features=[],
        )

    def pool(self, dataframe, cat_features=None, text_features=None):
        cat_features = cat_features or []
        text_features = text_features or []
        dataframe = dataframe.astype(
            {c: "string" for c in cat_features + text_features}
        )
        for column in dataframe.columns.drop(cat_features + text_features):
            dataframe[column] = pd.to_numeric(dataframe[column])
        print(
            dataframe.head(10),
            dataframe.shape,
            dataframe.dtypes,
            list(dataframe.select_dtypes("category").columns),
        )
        pool = catboost.Pool(
            data=dataframe.drop(["explicit"], axis="columns"),
            cat_features=cat_features,
            text_features=text_features,
            label=None
            if dataframe["explicit"].isna().any()
            else dataframe["explicit"].to_numpy(),
            group_id=dataframe["user_id"].to_numpy(),
        )
        return pool

    def fit(self, explicit_feedback):
        self.n_items = explicit_feedback.shape[1]
        dataframe = self.skeleton_dataframe()
        for user_ids in tqdm(
            torch.randperm(explicit_feedback.shape[0]).split(self.batch_size),
            "Extracting topk recommendations",
        ):
            batch = self.topk_data(user_ids, explicit_feedback[user_ids])
            dataframe = pd.concat([dataframe, batch["dataframe"]])

        pool = self.pool(
            dataframe,
            cat_features=batch["cat_features"],
            text_features=batch["text_features"],
        )
        self.model.fit(pool, verbose=50)

    def forward(self, user_ids, item_ids=None):
        if not torch.is_tensor(user_ids):
            user_ids = torch.from_numpy(user_ids)
        batch = self.topk_data(user_ids.to(torch.int64))
        pool = self.pool(**batch)
        grouped_ratings = self.model.predict(pool)
        grouped_ratings = (
            torch.from_numpy(grouped_ratings)
            .reshape(len(user_ids), -1)
            .to(torch.float32)
        )
        grouped_item_ids = torch.from_numpy(
            batch["dataframe"]["item_id"]
            .values.reshape(len(user_ids), -1)
            .astype("int64")
        )
        ratings = torch.full(
            (len(user_ids), self.n_items),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        ratings = ratings.scatter(1, grouped_item_ids, grouped_ratings)
        return ratings

    def save(self):
        self.model.save_model("tmp")
        with open("tmp", "rb") as f:
            bytes = f.read()
        return bytes

    def load(self, bytes):
        with open("tmp", "wb") as f:
            f.write(bytes)
        self.model.load_model("tmp")

    def feature_importance(self, user_ids, users_explicit_feedback):
        pool = self.pool(
            **self.topk_data(
                user_ids=user_ids,
                users_explicit_feedback=users_explicit_feedback,
            )
        )
        feature_importance = self.model.get_feature_importance(pool)
        dataframe = (
            pd.Series(feature_importance, self.model.feature_names_)
            .sort_values(ascending=False)
            .to_frame()
            .T
        )
        return dataframe


class CatboostMovieLensRecommenderModule(CatboostRecommenderModule):
    def __init__(self, *args, movielens_directory, **kwargs):
        super().__init__(*args, **kwargs)
        self.movielens = MovieLens(movielens_directory)

    @property
    def user_info(self):
        user_info = self.movielens["u.user"]
        user_info.index -= 1
        return dict(
            dataframe=user_info,
            cat_features=["gender", "occupation", "zip code"],
            text_features=[],
        )

    @property
    def item_info(self):
        item_info = self.movielens["u.item"]
        item_info.index -= 1
        item_info = item_info.drop(["video release date", "IMDb URL"], axis="columns")
        item_info["release date"] = pd.to_datetime(item_info["release date"])
        return dict(
            dataframe=item_info,
            cat_features=list(item_info.columns.drop(["release date", "movie title"])),
            text_features=["movie title"],
        )

    def topk_data(self, user_ids, users_explicit_feedback=None):
        batch = super().topk_data(user_ids, users_explicit_feedback)
        dataframe = batch["dataframe"]
        user_info = self.user_info
        dataframe = pd.merge(
            dataframe,
            user_info["dataframe"],
            how="left",
            left_on="user_id",
            right_index=True,
        )
        item_info = self.item_info
        dataframe = pd.merge(
            dataframe,
            item_info["dataframe"],
            how="left",
            left_on="item_id",
            right_index=True,
        )
        return dict(
            dataframe=dataframe,
            cat_features=batch["cat_features"]
            + user_info["cat_features"]
            + item_info["cat_features"],
            text_features=batch["text_features"]
            + user_info["text_features"]
            + item_info["text_features"],
        )


class CatboostRecommender(NonGradientRecommenderMixin, LitRecommenderBase):
    def build_model(self):
        config = self.hparams["model_config"].copy()
        artifact_names = config.pop("model_artifact_names")
        models = []
        wandb_api = WandbAPI()
        for artifact_name in artifact_names:
            artifact = wandb_api.artifact(artifact_name=artifact_name)
            models.append(
                wandb_api.build_from_checkpoint_artifact(
                    artifact,
                    class_candidates=[
                        MovieLensNonGradientRecommender,
                        MovieLensPMFRecommender,
                        MovieLensSLIMRecommender,
                    ],
                )
            )
        if config["name"] == "CatboostMovieLensRecommenderModule":
            config["movielens_directory"] = self.hparams["datamodule_config"][
                "directory"
            ]
        model = self.build_class(
            class_candidates=[
                CatboostRecommenderModule,
                CatboostMovieLensRecommenderModule,
            ],
            models=models,
            model_names=artifact_names,
            batch_size=self.hparams["datamodule_config"]["batch_size"],
            **config,
        )
        return model


class MovieLensCatBoostRecommender(CatboostRecommender, MovieLensRecommender):
    pass
