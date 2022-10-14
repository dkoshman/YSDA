import io
import pickle
from typing import Literal

import implicit
import numpy as np
import scipy
import torch
from sklearn.decomposition import TruncatedSVD

from my_tools.utils import build_class

from .interface import (
    RecommenderModuleInterface,
    FitExplicitInterfaceMixin,
    InMemoryRecommender,
)
from .utils import torch_sparse_to_scipy_coo


class RandomRecommender(RecommenderModuleInterface, FitExplicitInterfaceMixin):
    def fit(self, explicit_feedback):
        pass

    def forward(self, user_ids):
        ratings = torch.randn(len(user_ids), self.n_items)
        return ratings

    def _new_users(self, n_new_users) -> None:
        pass

    def _new_items(self, n_new_items) -> None:
        pass


class PopularRecommender(RecommenderModuleInterface, FitExplicitInterfaceMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer(name="items_count", tensor=torch.zeros(self.n_items))

    def fit(self, explicit_feedback):
        implicit_feedback = explicit_feedback > 0
        self.items_count = torch.from_numpy(implicit_feedback.sum(axis=0).A.squeeze())

    def forward(self, user_ids):
        ratings = self.items_count.repeat(len(user_ids), 1)
        return ratings

    def _new_users(self, n_new_users) -> None:
        pass

    def _new_items(self, n_new_items) -> None:
        self.items_count = torch.cat([self.items_count, torch.zeros(n_new_items)])


class SVDRecommender(InMemoryRecommender, FitExplicitInterfaceMixin):
    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self.model = TruncatedSVD(n_components=n_components)

    def fit(self, explicit_feedback):
        self.save_explicit_feedback(explicit_feedback)
        self.model.fit(explicit_feedback)

    def forward(self, user_ids):
        explicit_feedback = self.explicit_feedback_scipy_coo.tocsr()
        embedding = self.model.transform(explicit_feedback[user_ids].A)
        ratings = self.model.inverse_transform(embedding)
        return torch.from_numpy(ratings)

    def _new_users(self, n_new_users) -> None:
        self.fit(self.explicit_feedback_scipy_coo)

    def _new_items(self, n_new_items) -> None:
        self.fit(self.explicit_feedback_scipy_coo)

    def save(self):
        bytes = pickle.dumps(self.model)
        return bytes

    def load(self, bytes):
        self.model = pickle.load(io.BytesIO(bytes))


class ImplicitRecommenderBase(InMemoryRecommender, FitExplicitInterfaceMixin):
    def __init__(self, *, implicit_model, implicit_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.model = build_class(
            module_candidates=[
                implicit.nearest_neighbours,
                implicit.als,
                implicit.bpr,
                implicit.lmf,
            ],
            name=implicit_model,
            **(implicit_kwargs or {}),
        )

    def fit(self, explicit_feedback):
        self.save_explicit_feedback(explicit_feedback.astype(np.float32))
        self.model.fit(explicit_feedback)

    def forward(self, user_ids):
        explicit_feedback = self.explicit_feedback_scipy_coo.tocsr()
        indices, item_ratings = self.model.recommend(
            userid=user_ids.numpy(),
            user_items=explicit_feedback[user_ids],
            N=self.n_items,
            filter_already_liked_items=False,
        )
        ratings = np.full(
            shape=(len(user_ids), self.n_items + 1),
            fill_value=np.finfo(np.float32).min,
        )
        np.put_along_axis(arr=ratings, indices=indices, values=item_ratings, axis=1)
        ratings = torch.from_numpy(ratings[:, :-1])
        return ratings

    def _new_users(self, n_new_users) -> None:
        explicit_feedback = self.explicit_feedback_scipy_coo
        new_users_feedback = explicit_feedback.tocsr()[-n_new_users:]

        if hasattr(self.model, "partial_fit_users"):
            new_userids = np.arange(self.n_users, self.n_users + n_new_users)
            # Partial fit fails if user_items is completely empty.
            nonzero_user = explicit_feedback.row[0]
            new_userids = np.concatenate([new_userids, [nonzero_user]])
            new_users_feedback = scipy.sparse.vstack(
                [new_users_feedback, explicit_feedback.tocsr()[nonzero_user]]
            )
            self.model.partial_fit_users(
                userids=new_userids,
                user_items=new_users_feedback.tocsr(),
            )
        else:
            self.model.fit(explicit_feedback)

    def _new_items(self, n_new_items) -> None:
        explicit_feedback = self.explicit_feedback_scipy_coo
        new_items_feedback = explicit_feedback.tocsr()[:, -n_new_items:]

        if hasattr(self.model, "partial_fit_items"):
            new_itemids = np.arange(self.n_items, self.n_items + n_new_items)
            # Partial fit fails if item_users is completely empty.
            nonzero_item = explicit_feedback.col[0]
            new_itemids = np.concatenate([new_itemids, [nonzero_item]])
            new_items_feedback = scipy.sparse.hstack(
                [new_items_feedback, explicit_feedback.tocsr()[:, nonzero_item]]
            )
            self.model.partial_fit_items(
                itemids=new_itemids,
                item_users=new_items_feedback.T,
            )
        else:
            self.model.fit(explicit_feedback)

    def save(self):
        bytesio = io.BytesIO()
        self.model.save(bytesio)
        return bytesio

    def load(self, bytesio):
        bytesio.seek(0)
        self.model = self.model.load(bytesio)


class ImplicitMatrixFactorizationRecommender(ImplicitRecommenderBase):
    def __init__(
        self,
        implicit_model: Literal[
            "AlternatingLeastSquares",
            "LogisticMatrixFactorization",
            "BayesianPersonalizedRanking",
        ] = "AlternatingLeastSquares",
        factors=100,
        learning_rate=1e-2,
        regularization=1e-2,
        num_threads=0,
        use_gpu=True,
        **kwargs,
    ):
        implicit_kwargs = dict(
            factors=factors,
            learning_rate=learning_rate,
            regularization=regularization,
            num_threads=num_threads,
            use_gpu=use_gpu,
        )
        if implicit_model == "AlternatingLeastSquares":
            implicit_kwargs.pop("learning_rate")
        else:
            implicit_kwargs["use_gpu"] = False
        super().__init__(
            implicit_model=implicit_model, implicit_kwargs=implicit_kwargs, **kwargs
        )


class ImplicitNearestNeighborsRecommender(ImplicitRecommenderBase):
    def __init__(
        self,
        implicit_model: Literal[
            "BM25Recommender", "CosineRecommender", "TFIDFRecommender"
        ] = "BM25Recommender",
        num_neighbors=20,
        num_threads=0,
        **kwargs,
    ):
        super().__init__(
            implicit_model=implicit_model,
            implicit_kwargs=dict(K=num_neighbors, num_threads=num_threads),
            **kwargs,
        )

    def similar_users(self, users_feedback=None, user_ids=None):
        explicit_feedback = torch_sparse_to_scipy_coo(self.explicit_feedback)
        if users_feedback is not None:
            n_users = explicit_feedback.shape[0]
            explicit_feedback = scipy.sparse.vstack(
                [
                    explicit_feedback,
                    users_feedback,
                ]
            )
            user_ids = np.arange(n_users, explicit_feedback.shape[0])
            filter_users = user_ids
        else:
            filter_users = None

        self.model.fit(explicit_feedback.T)
        similar_users_ids, similarity = self.model.similar_items(
            user_ids, filter_items=filter_users
        )
        self.model.fit(explicit_feedback)
        if users_feedback is None:
            similar_users_ids = similar_users_ids[:, 1:]
            similarity = similarity[:, 1:]
        return dict(
            similar_users=torch.from_numpy(similar_users_ids).to(torch.int64),
            similarity=torch.from_numpy(similarity),
        )


# # I didn't use this class much so I didn't bother to keep it up to date.
# class NearestNeighbours(InMemoryRecommender, FitExplicitInterfaceMixin):
#     def __init__(self, num_neighbors=10, **kwargs):
#         super().__init__(**kwargs)
#         self.num_neighbors = num_neighbors
#
#     @staticmethod
#     def norm(sparse_matrix: scipy.sparse.coo_matrix):
#         return np.maximum(sparse_matrix.multiply(sparse_matrix).sum(axis=1), 1e-8)
#
#     def sparse_cosine_similarity(self, left, right):
#         if left.shape[-1] != right.shape[-1]:
#             raise ValueError(
#                 "Cannot compute similarity between tensors with non matching"
#                 f"last dimensions: {left.shape}, {right.shape}"
#             )
#         similarity = left @ right.T
#         similarity = similarity / self.norm(left) / self.norm(right).T
#         return np.asarray(similarity)
#
#     def similar_users(self, users_feedback=None, user_ids=None, k=None):
#         if k is None:
#             k = self.num_neighbors
#         explicit_feedback = torch_sparse_to_scipy_coo(self.explicit_feedback).tocsr()
#
#         if torch.is_tensor(users_feedback):
#             users_feedback = torch_sparse_to_scipy_coo(users_feedback)
#         elif users_feedback is None:
#             if user_ids is None:
#                 raise ValueError("One of users_feedback, user_ids must be non None.")
#             users_feedback = explicit_feedback[user_ids]
#
#         similarity = self.sparse_cosine_similarity(users_feedback, explicit_feedback)
#         similarity = torch.from_numpy(similarity)
#         neighbors_similarity, indices = torch.topk(similarity, k=k)
#         neighbors_similarity /= neighbors_similarity.sum(axis=1)[:, None]
#
#         ratings = np.empty(users_feedback.shape)
#         for i, (similarity, ind) in enumerate(
#             zip(neighbors_similarity.numpy(), indices.numpy())
#         ):
#             ratings[i] = explicit_feedback[ind].T @ similarity
#
#         return dict(
#             ratings=torch.from_numpy(ratings),
#             similar_users=indices,
#             similarity=neighbors_similarity.to(torch.float32),
#         )
#
#     def forward(self, users_feedback=None, user_ids=None, item_ids=None):
#         similarity_dict = self.similar_users(
#             users_feedback=users_feedback, user_ids=user_ids
#         )
#         if item_ids is None:
#             item_ids = slice(None)
#         return similarity_dict["ratings"][item_ids]
