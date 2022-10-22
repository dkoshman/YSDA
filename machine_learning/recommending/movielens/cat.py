import pandas as pd

from .data import MovieLens
from ..models.cat import CatboostInterface


class CatboostMovieLensFeatureRecommender(CatboostInterface):
    def __init__(self, *, movielens_directory=None, **kwargs):
        super().__init__(**kwargs)
        self.movielens_directory = movielens_directory

    @property
    def movielens(self):
        return MovieLens(self.movielens_directory)

    @property
    def user_features(self):
        user_features = self.movielens["u.user"]
        user_features.index -= 1
        return user_features

    @property
    def item_features(self):
        item_features = self.movielens["u.item"]
        item_features.index -= 1
        item_features = item_features.drop(
            ["video release date", "IMDb URL"], axis="columns"
        )
        item_features["release date"] = pd.to_datetime(item_features["release date"])
        return item_features

    @property
    def cat_features(self):
        user_cat_features = ["gender", "occupation", "zip code"]
        item_cat_features = list(
            self.item_features.columns.drop(["release date", "movie title"])
        )
        return ["user_ids", "item_ids"] + user_cat_features + item_cat_features

    @property
    def text_features(self):
        return ["movie title"]

    def features_dataframe(self, dataframe):
        dataframe = pd.merge(
            dataframe,
            self.user_features,
            how="left",
            left_on="user_ids",
            right_index=True,
        )
        dataframe = pd.merge(
            dataframe,
            self.item_features,
            how="left",
            left_on="item_ids",
            right_index=True,
        )
        return dataframe

    def train_dataframe(self, explicit):
        dataframe = self.explicit_dataframe(explicit)
        dataframe = self.features_dataframe(dataframe)
        return dataframe

    def predict_dataframe(self, user_ids, item_ids):
        dataframe = self.dense_dataframe(user_ids, item_ids)
        dataframe = self.features_dataframe(dataframe)
        return dataframe

    def get_extra_state(self):
        binary_bytes = super().get_extra_state()
        state = dict(
            binary_bytes=binary_bytes, movielens_directory=self.movielens_directory
        )
        return state

    def set_extra_state(self, state):
        super().set_extra_state(state["binary_bytes"])
        self.movielens_directory = state["movielens_directory"]


# class CatboostModule(CatboostBase):
#     def forward(self, user_ids=None, item_ids=None):
#         if not torch.is_tensor(user_ids):
#             user_ids = torch.from_numpy(user_ids)
#         explicit_feedback = torch_sparse_to_scipy_coo(self.explicit_feedback)
#         batch = self.topk_data(user_ids.to(torch.int64))
#         pool = self.pool(**batch)
#         grouped_ratings = self.model.predict(pool)
#         grouped_ratings = (
#             torch.from_numpy(grouped_ratings)
#             .reshape(len(user_ids), -1)
#             .to(torch.float32)
#         )
#         grouped_item_ids = torch.from_numpy(
#             batch["dataframe"]["item_id"]
#             .values.reshape(len(user_ids), -1)
#             .astype("int64")
#         )
#         ratings = torch.full(
#             (len(user_ids), self.n_items),
#             torch.finfo(torch.float32).min,
#             dtype=torch.float32,
#         )
#         ratings = ratings.scatter(1, grouped_item_ids, grouped_ratings)
#         return ratings
#
#
# class CatboostRecommenderAggregatorModule(CatboostBase, RecommenderModuleBase):
#     def __init__(
#         self,
#         n_users,
#         n_items,
#         models,
#         model_names=None,
#         n_recommendations=10,
#         **cb_params,
#     ):
#         if model_names is not None and len(models) != len(model_names):
#             raise ValueError("Models and their names must be of equal length.")
#         super().__init__(n_users=n_users, n_items=n_items)
#         self.models = models
#         self.model_names = (
#             self.generate_model_names(models) if model_names is None else model_names
#         )
#         self.n_recommendations = n_recommendations
#         self.model = catboost.CatBoostRanker(**cb_params)
#
#     @staticmethod
#     def generate_model_names(models):
#         names = []
#         counter = Counter()
#         for model in models:
#             name = model.__class__.__name__
#             names.append(f"{name}_{counter[name]}")
#             counter.update([name])
#         return names
#
#     def skeleton_dataframe(self):
#         return pd.DataFrame(
#             columns=["user_id", "item_id", "explicit"] + self.model_names
#         )
#
#     def topk_data(self, user_ids, users_explicit_feedback=None):
#         ratings = []
#         topk_item_ids = []
#         for model in self.models:
#             with torch.inference_mode():
#                 model_ratings = model(user_ids=user_ids).to_dense()
#             ratings.append(model_ratings)
#             values, item_ids = torch.topk(model_ratings, k=self.n_recommendations)
#             topk_item_ids.append(item_ids)
#
#         per_user_topk_item_ids = []
#         for user_id, item_ids in zip(user_ids, torch.cat(topk_item_ids, dim=1)):
#             unique_item_ids, counts = item_ids.unique(return_counts=True)
#             _, indices = torch.topk(counts, k=self.n_recommendations)
#             per_user_topk_item_ids.append(unique_item_ids[indices])
#         per_user_topk_item_ids = torch.stack(per_user_topk_item_ids)
#
#         dataframe = self.skeleton_dataframe()
#         for i, (user_id, item_ids) in enumerate(zip(user_ids, per_user_topk_item_ids)):
#             df = self.skeleton_dataframe()
#             df["item_id"] = item_ids.numpy()
#             df["user_id"] = user_id.numpy()
#             if users_explicit_feedback is not None:
#                 df["explicit"] = (
#                     users_explicit_feedback[i, item_ids].toarray().squeeze()
#                 )
#             item_ratings = [rating[i, item_ids] for rating in ratings]
#             df.iloc[:, 3:] = torch.stack(item_ratings, dim=0).T.numpy()
#             dataframe = pd.concat([dataframe, df])
#
#         return dict(
#             dataframe=dataframe,
#             cat_features=["user_id", "item_id"],
#             text_features=[],
#         )
#
#     def batched_topk_data(self, user_ids=None, explicit_feedback=None, batch_size=100):
#         if user_ids is None:
#             user_ids = torch.arange(self.n_users)
#         dataframe = self.skeleton_dataframe()
#         for batch_user_ids in user_ids.split(batch_size):
#             batch_explicit = (
#                 None if explicit_feedback is None else explicit_feedback[user_ids]
#             )
#             batch = self.topk_data(batch_user_ids, batch_explicit)
#             dataframe = pd.concat([dataframe, batch["dataframe"]])
#         return dict(
#             dataframe=dataframe,
#             cat_features=batch["cat_features"],
#             text_features=batch["text_features"],
#         )
#
#     def fit(self, explicit_feedback):
#         data = self.batched_topk_data(explicit_feedback=explicit_feedback)
#         pool = self.pool(**data)
#         self.model.fit(pool, verbose=50)
#
#     def forward(self, user_ids=None, item_ids=None):
#         if not torch.is_tensor(user_ids):
#             user_ids = torch.from_numpy(user_ids)
#         batch = self.topk_data(user_ids.to(torch.int64))
#         pool = self.pool(**batch)
#         grouped_ratings = self.model.predict(pool)
#         grouped_ratings = (
#             torch.from_numpy(grouped_ratings)
#             .reshape(len(user_ids), -1)
#             .to(torch.float32)
#         )
#         grouped_item_ids = torch.from_numpy(
#             batch["dataframe"]["item_id"]
#             .values.reshape(len(user_ids), -1)
#             .astype("int64")
#         )
#         ratings = torch.full(
#             (len(user_ids), self.n_items),
#             torch.finfo(torch.float32).min,
#             dtype=torch.float32,
#         )
#         ratings = ratings.scatter(1, grouped_item_ids, grouped_ratings)
#         return ratings
#


# class CatboostRecommenderModule(CatboostMixin, RecommenderModuleBase):
#     def __init__(
#         self,
#         n_users,
#         n_items,
#         models,
#         model_names=None,
#         n_recommendations=10,
#         **cb_params,
#     ):
#         if model_names is not None and len(models) != len(model_names):
#             raise ValueError("Models and their names must be of equal length.")
#         super().__init__(n_users=n_users, n_items=n_items)
#         self.models = models
#         self.model_names = (
#             self.generate_model_names(models) if model_names is None else model_names
#         )
#         self.n_recommendations = n_recommendations
#         self.model = catboost.CatBoostRanker(**cb_params)
#
#     @staticmethod
#     def generate_model_names(models):
#         names = []
#         counter = Counter()
#         for model in models:
#             name = model.__class__.__name__
#             names.append(f"{name}_{counter[name]}")
#             counter.update([name])
#         return names
#
#     def skeleton_dataframe(self):
#         return pd.DataFrame(
#             columns=["user_id", "item_id", "explicit"] + self.model_names
#         )
#
#     def topk_data(self, user_ids, users_explicit_feedback=None):
#         ratings = []
#         topk_item_ids = []
#         for model in self.models:
#             with torch.inference_mode():
#                 model_ratings = model(user_ids=user_ids).to_dense()
#             ratings.append(model_ratings)
#             values, item_ids = torch.topk(model_ratings, k=self.n_recommendations)
#             topk_item_ids.append(item_ids)
#
#         per_user_topk_item_ids = []
#         for user_id, item_ids in zip(user_ids, torch.cat(topk_item_ids, dim=1)):
#             unique_item_ids, counts = item_ids.unique(return_counts=True)
#             _, indices = torch.topk(counts, k=self.n_recommendations)
#             per_user_topk_item_ids.append(unique_item_ids[indices])
#         per_user_topk_item_ids = torch.stack(per_user_topk_item_ids)
#
#         dataframe = self.skeleton_dataframe()
#         for i, (user_id, item_ids) in enumerate(zip(user_ids, per_user_topk_item_ids)):
#             df = self.skeleton_dataframe()
#             df["item_id"] = item_ids.numpy()
#             df["user_id"] = user_id.numpy()
#             if users_explicit_feedback is not None:
#                 df["explicit"] = (
#                     users_explicit_feedback[i, item_ids].toarray().squeeze()
#                 )
#             item_ratings = [rating[i, item_ids] for rating in ratings]
#             df.iloc[:, 3:] = torch.stack(item_ratings, dim=0).T.numpy()
#             dataframe = pd.concat([dataframe, df])
#
#         return dict(
#             dataframe=dataframe,
#             cat_features=["user_id", "item_id"],
#             text_features=[],
#         )
#
#     def batched_topk_data(self, user_ids=None, explicit_feedback=None, batch_size=100):
#         if user_ids is None:
#             user_ids = torch.arange(self.n_users)
#         dataframe = self.skeleton_dataframe()
#         for batch_user_ids in user_ids.split(batch_size):
#             batch_explicit = (
#                 None if explicit_feedback is None else explicit_feedback[user_ids]
#             )
#             batch = self.topk_data(batch_user_ids, batch_explicit)
#             dataframe = pd.concat([dataframe, batch["dataframe"]])
#         return dict(
#             dataframe=dataframe,
#             cat_features=batch["cat_features"],
#             text_features=batch["text_features"],
#         )
#
#     def fit(self, explicit_feedback):
#         data = self.batched_topk_data(explicit_feedback=explicit_feedback)
#         pool = self.pool(**data)
#         self.model.fit(pool, verbose=50)
#
#     def forward(self, user_ids=None, item_ids=None):
#         if not torch.is_tensor(user_ids):
#             user_ids = torch.from_numpy(user_ids)
#         batch = self.topk_data(user_ids.to(torch.int64))
#         pool = self.pool(**batch)
#         grouped_ratings = self.model.predict(pool)
#         grouped_ratings = (
#             torch.from_numpy(grouped_ratings)
#             .reshape(len(user_ids), -1)
#             .to(torch.float32)
#         )
#         grouped_item_ids = torch.from_numpy(
#             batch["dataframe"]["item_id"]
#             .values.reshape(len(user_ids), -1)
#             .astype("int64")
#         )
#         ratings = torch.full(
#             (len(user_ids), self.n_items),
#             torch.finfo(torch.float32).min,
#             dtype=torch.float32,
#         )
#         ratings = ratings.scatter(1, grouped_item_ids, grouped_ratings)
#         return ratings
#


# class CatboostRecommenderModuleFromArtifacts(CatboostMovieLensRecommenderModule):
#     def __init__(self, *args, model_artifact_names, **kwargs):
#         models = []
#         for artifact_name in model_artifact_names:
#             models.append(
#                 pl_module_from_checkpoint_artifact(
#                     artifact_name=artifact_name,
#                     class_candidates=[
#                         MovieLensNonGradientRecommender,
#                         MovieLensPMFRecommender,
#                         MovieLensSLIMRecommender,
#                     ],
#                 )
#             )
#         super().__init__(
#             *args, models=models, model_names=model_artifact_names, **kwargs
#         )
