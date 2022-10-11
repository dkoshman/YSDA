from collections import Counter

import catboost
import pandas as pd
import torch
from tqdm.auto import tqdm

from entrypoints import NonGradientRecommenderMixin, LitRecommenderBase
from movielens import (
    MovieLensNonGradientRecommender,
    MovieLensPMFRecommender,
    MovieLensSLIMRecommender,
    MovieLensRecommender,
)
from utils import WandbAPI, torch_sparse_to_scipy_coo


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

        return dataframe

    def pool(self, dataframe, training=False):
        pool = catboost.Pool(
            data=dataframe.drop(["explicit", "user_id", "item_id"], axis="columns"),
            label=dataframe["explicit"].values if training else None,
            group_id=dataframe["user_id"].values,
        )
        return pool

    def fit(self, explicit_feedback):
        self.n_items = explicit_feedback.shape[1]
        dataframe = self.skeleton_dataframe()
        for user_ids in tqdm(
            torch.randperm(explicit_feedback.shape[0]).split(self.batch_size),
            "Extracting topk recommendations",
        ):
            dataframe = pd.concat(
                [dataframe, self.topk_data(user_ids, explicit_feedback[user_ids])]
            )
        pool = self.pool(dataframe, training=True)
        self.model.fit(pool, verbose=50)

    def forward(self, user_ids, item_ids=None):
        if not torch.is_tensor(user_ids):
            user_ids = torch.from_numpy(user_ids)
        dataframe = self.topk_data(user_ids)
        pool = self.pool(dataframe)
        topk_ratings = self.model.predict(pool)
        topk_ratings = (
            torch.from_numpy(topk_ratings).reshape(len(user_ids), -1).to(torch.float32)
        )
        per_user_topk_item_ids = torch.from_numpy(
            dataframe["item_id"].values.reshape(len(user_ids), -1).astype("int64")
        )
        ratings = torch.full(
            (len(user_ids), self.n_items),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
        )
        ratings = ratings.scatter(1, per_user_topk_item_ids, topk_ratings)
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
        model = self.build_class(
            class_candidates=[CatboostRecommenderModule],
            models=models,
            model_names=artifact_names,
            batch_size=self.hparams["datamodule_config"]["batch_size"],
            **config,
        )
        return model


class MovieLensCatBoostRecommender(CatboostRecommender, MovieLensRecommender):
    pass
