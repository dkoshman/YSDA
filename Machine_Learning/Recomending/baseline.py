import abc
import io
import pickle
from typing import Literal

import implicit
import numpy as np
import scipy
import torch
from sklearn.decomposition import TruncatedSVD

from my_tools.utils import build_class

from utils import torch_sparse_to_scipy_coo, scipy_coo_to_torch_sparse


class BaselineRecommender(torch.nn.Module, abc.ABC):
    def __init__(self, *, n_users, n_items):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

    def fit(self, explicit_feedback):
        pass

    @abc.abstractmethod
    def forward(self, user_ids=None, item_ids=None):
        return torch.Tensor()


class RandomRecommender(BaselineRecommender):
    def forward(self, user_ids=None, item_ids=None):
        n_users = self.n_users if user_ids is None else len(user_ids)
        n_items = self.n_items if item_ids is None else len(item_ids)
        ratings = torch.randn(n_users, n_items)
        return ratings


class PopularRecommender(BaselineRecommender):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.items_count = torch.nn.Parameter(
            torch.zeros(self.n_items), requires_grad=False
        )

    def fit(self, explicit_feedback):
        implicit_feedback = explicit_feedback > 0
        self.items_count[:] = torch.from_numpy(
            implicit_feedback.sum(axis=0).A.squeeze()
        )

    def forward(self, user_ids=None, item_ids=None):
        n_users = self.n_users if user_ids is None else len(user_ids)
        if item_ids is None:
            item_ids = slice(None)

        ratings = self.items_count[item_ids]
        ratings = ratings.repeat(n_users, 1)
        return ratings


class InMemoryRecommender(BaselineRecommender, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.explicit_feedback = torch.nn.Parameter(
            torch.sparse_coo_tensor(size=(self.n_users, self.n_items)),
            requires_grad=False,
        )

    def fit(self, explicit_feedback):
        if explicit_feedback.shape != self.explicit_feedback.shape:
            raise ValueError("Explicit feedback shape mismatch.")
        self.explicit_feedback = torch.nn.Parameter(
            scipy_coo_to_torch_sparse(explicit_feedback.tocoo()),
            requires_grad=False,
        )


class ImplicitRecommender(InMemoryRecommender):
    def __init__(
        self,
        *,
        implicit_model,
        implicit_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.implicit_model = build_class(
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
        self.implicit_model.fit(explicit_feedback)
        super().fit(explicit_feedback)

    def forward(self, user_ids=None, item_ids=None):
        if user_ids is None:
            user_ids = np.arange(self.n_users)
        elif torch.is_tensor(user_ids):
            user_ids = user_ids.numpy()

        if item_ids is None:
            item_ids = slice(None)

        explicit_feedback = torch_sparse_to_scipy_coo(self.explicit_feedback)
        explicit_feedback = explicit_feedback.tocsr().astype(np.float32)
        indices, item_ratings = self.implicit_model.recommend(
            user_ids,
            explicit_feedback[user_ids],
            N=self.n_items,
            filter_already_liked_items=False,
        )
        ratings = np.full(
            [indices.shape[0], indices.shape[1] + 1], np.finfo(np.float32).min
        )
        np.put_along_axis(arr=ratings, indices=indices, values=item_ratings, axis=1)
        ratings = ratings[:, :-1]
        ratings = torch.from_numpy(ratings)[:, item_ids]
        return ratings

    def save(self):
        bytes = io.BytesIO()
        self.implicit_model.save(bytes)
        return bytes

    def load(self, bytes):
        bytes.seek(0)
        print(self.implicit_model, "\nLoaded\n")
        self.implicit_model.load(bytes)


class ImplicitMatrixFactorizationRecommender(ImplicitRecommender):
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


class ImplicitNearestNeighborsRecommender(ImplicitRecommender):
    def __init__(
        self,
        implicit_model: Literal[
            "BM25Recommender", "CosineRecommender", "TFIDFRecommender"
        ] = "BM25Recommender",
        num_neighbors=20,
        num_threads=0,
        **kwargs,
    ):
        implicit_kwargs = dict(K=num_neighbors, num_threads=num_threads)
        super().__init__(
            implicit_model=implicit_model, implicit_kwargs=implicit_kwargs, **kwargs
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

        self.implicit_model.fit(explicit_feedback.T)
        similar_users_ids, similarity = self.implicit_model.similar_items(
            user_ids, filter_items=user_ids
        )
        return dict(
            similar_users=torch.from_numpy(similar_users_ids).to(torch.int64),
            similarity=torch.from_numpy(similarity),
        )


class SVDRecommender(InMemoryRecommender):
    def __init__(self, n_components=10, **kwargs):
        super().__init__(**kwargs)
        self.model = TruncatedSVD(n_components=n_components)

    def fit(self, explicit_feedback):
        self.model.fit(explicit_feedback)
        super().fit(explicit_feedback)

    def forward(self, user_ids=None, item_ids=None):
        if user_ids is None:
            user_ids = slice(None)

        if item_ids is None:
            item_ids = slice(None)

        explicit_feedback = torch_sparse_to_scipy_coo(self.explicit_feedback).tocsr()
        embedding = self.model.transform(explicit_feedback[user_ids].A)
        ratings = self.model.inverse_transform(embedding)[:, item_ids]
        return torch.from_numpy(ratings)

    def save(self):
        return pickle.dumps(self.model)

    def load(self, bytes):
        self.model = pickle.load(io.BytesIO(bytes))


class NearestNeighbours(InMemoryRecommender):
    def __init__(self, num_neighbors=10, **kwargs):
        super().__init__(**kwargs)
        self.num_neighbors = num_neighbors

    @staticmethod
    def norm(sparse_matrix: scipy.sparse.coo_matrix):
        return np.maximum(sparse_matrix.multiply(sparse_matrix).sum(axis=1), 1e-8)

    def sparse_cosine_similarity(self, left, right):
        if left.shape[-1] != right.shape[-1]:
            raise ValueError(
                "Cannot compute similarity between tensors with non matching"
                f"last dimensions: {left.shape}, {right.shape}"
            )
        similarity = left @ right.T
        similarity = similarity / self.norm(left) / self.norm(right).T
        return np.asarray(similarity)

    def similar_users(self, users_feedback=None, user_ids=None, k=None):
        if k is None:
            k = self.num_neighbors
        explicit_feedback = torch_sparse_to_scipy_coo(self.explicit_feedback).tocsr()

        if torch.is_tensor(users_feedback):
            users_feedback = torch_sparse_to_scipy_coo(users_feedback)
        elif users_feedback is None:
            if user_ids is None:
                raise ValueError("One of users_feedback, user_ids must be non None.")
            users_feedback = explicit_feedback[user_ids]

        similarity = self.sparse_cosine_similarity(users_feedback, explicit_feedback)
        similarity = torch.from_numpy(similarity)
        neighbors_similarity, indices = torch.topk(similarity, k=k)
        neighbors_similarity /= neighbors_similarity.sum(axis=1)[:, None]

        ratings = np.empty(users_feedback.shape)
        for i, (similarity, ind) in enumerate(
            zip(neighbors_similarity.numpy(), indices.numpy())
        ):
            ratings[i] = explicit_feedback[ind].T @ similarity

        return dict(
            ratings=torch.from_numpy(ratings),
            similar_users=indices,
            similarity=neighbors_similarity.to(torch.float32),
        )

    def forward(self, users_feedback=None, user_ids=None, item_ids=None):
        similarity_dict = self.similar_users(
            users_feedback=users_feedback, user_ids=user_ids
        )
        if item_ids is None:
            item_ids = slice(None)
        return similarity_dict["ratings"][item_ids]
