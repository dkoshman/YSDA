import abc
from typing import Literal

import implicit
import numpy as np
import scipy
import torch

from sklearn.decomposition import TruncatedSVD

from my_tools.utils import build_class

from utils import torch_sparse_to_scipy_coo


class BaselineRecommender:
    def __init__(self, explicit_feedback):
        self.explicit_feedback = explicit_feedback
        self.n_users, self.n_items = explicit_feedback.shape
        self.model = None
        self.setup()

    def setup(self):
        ...

    def fit(self):
        self.model.fit(self.explicit_feedback)

    @abc.abstractmethod
    def __call__(self, user_ids=None, item_ids=None):
        ratings = self.model.recommend(user_ids, item_ids)
        return ratings


class RandomRecommender(BaselineRecommender):
    def fit(self):
        pass

    def __call__(self, user_ids=None, item_ids=None):
        n_users = self.n_users if user_ids is None else len(user_ids)
        n_items = self.n_items if item_ids is None else len(item_ids)
        ratings = torch.randn(n_users, n_items)
        return ratings


class PopularRecommender(BaselineRecommender):
    def fit(self):
        implicit_feedback = self.explicit_feedback > 0
        self.items_count = implicit_feedback.sum(axis=0).A.squeeze()

    def __call__(self, user_ids=None, item_ids=None):
        n_users = self.n_users if user_ids is None else len(user_ids)
        if item_ids is None:
            item_ids = slice(None)

        ratings = torch.from_numpy(self.items_count[item_ids])
        ratings = ratings.repeat(n_users, 1)
        return ratings


class ImplicitBase(BaselineRecommender):
    def __init__(self, explicit_feedback, name, **kwargs):
        kwargs.update(name=name)
        self.kwargs = kwargs
        super().__init__(explicit_feedback)

    def setup(self):
        self.explicit_feedback = self.explicit_feedback.astype(np.float64)
        self.model = build_class(
            modules=[
                implicit.nearest_neighbours,
                implicit.als,
                implicit.bpr,
                implicit.lmf,
            ],
            **self.kwargs,
        )

    def __call__(self, user_ids=None, item_ids=None):
        if user_ids is None:
            user_ids = np.arange(self.n_users)
        elif torch.is_tensor(user_ids):
            user_ids = user_ids.numpy()

        if item_ids is None:
            item_ids = slice(None)

        ids, ratings = self.model.recommend(
            user_ids, self.explicit_feedback[user_ids], N=self.n_items
        )
        return ratings[item_ids]


class ImplicitNearestNeighbors(ImplicitBase):
    def __init__(
        self,
        explicit_feedback,
        name: Literal[
            "BM25Recommender",
            "CosineRecommender",
            "TFIDFRecommender",
        ] = "BM25Recommender",
        num_neighbors=20,
        num_threads=1,
    ):
        self.k = num_neighbors
        super().__init__(
            explicit_feedback, name=name, K=num_neighbors, num_threads=num_threads
        )

    def similar_users(self, users_feedback=None, user_ids=None):
        if users_feedback is not None:
            n_users = self.explicit_feedback.shape[0]
            explicit_feedback = scipy.sparse.vstack(
                [
                    self.explicit_feedback,
                    users_feedback,
                ]
            )
            user_ids = np.arange(n_users, explicit_feedback.shape[0])
        else:
            explicit_feedback = self.explicit_feedback

        self.model.fit(explicit_feedback.T)
        similar_users, similarity = self.model.similar_items(
            user_ids, filter_items=user_ids
        )
        self.fit()
        return dict(similar_users=similar_users, similarity=similarity)


class ImplicitMatrixFactorization(ImplicitBase):
    def __init__(
        self,
        explicit_feedback,
        matrix_factorization_model: Literal[
            "AlternatingLeastSquares",
            "LogisticMatrixFactorization",
            "BayesianPersonalizedRanking",
        ] = "AlternatingLeastSquares",
        **kwargs,
    ):
        """
        The following keywords are allowed:

        factors
        learning_rate
        regularization
        iterations
        use_gpu
        num_threads
        """
        super().__init__(explicit_feedback, name=matrix_factorization_model, **kwargs)


class SVDRecommender(BaselineRecommender):
    def __init__(self, explicit_feedback, n_components=10):
        self.n_components = n_components
        super().__init__(explicit_feedback)

    def setup(self):
        self.model = TruncatedSVD(n_components=self.n_components)

    def __call__(self, user_ids=None, item_ids=None):
        if user_ids is None:
            user_ids = slice(None)

        if item_ids is None:
            item_ids = slice(None)

        embedding = self.model.transform(self.explicit_feedback[user_ids].A)
        ratings = self.model.inverse_transform(embedding)[:, item_ids]
        return ratings


class NearestNeighbours(BaselineRecommender):
    def __init__(self, explicit_feedback, num_neighbors=10):
        self.num_neighbors = num_neighbors
        super().__init__(explicit_feedback)

    def fit(self):
        pass

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

        if torch.is_tensor(users_feedback):
            users_feedback = torch_sparse_to_scipy_coo(users_feedback)
        elif users_feedback is None:
            if user_ids is None:
                raise ValueError("One of users_feedback, user_ids must be non None.")
            users_feedback = self.explicit_feedback[user_ids]

        similarity = self.sparse_cosine_similarity(
            users_feedback, self.explicit_feedback
        )
        similarity = torch.from_numpy(similarity)
        neighbors_similarity, indices = torch.topk(similarity, k=k)
        neighbors_similarity /= neighbors_similarity.sum(axis=1)[:, None]

        ratings = np.empty(users_feedback.shape)
        for i, (similarity, ind) in enumerate(
            zip(neighbors_similarity.numpy(), indices.numpy())
        ):
            ratings[i] = self.explicit_feedback[ind].T @ similarity

        return dict(
            ratings=ratings,
            similar_users=indices,
            similarity=neighbors_similarity.to(torch.float32),
        )

    def __call__(self, users_feedback=None, user_ids=None, item_ids=None):
        similarity_dict = self.similar_users(
            users_feedback=users_feedback, user_ids=user_ids
        )
        if item_ids is None:
            item_ids = slice(None)
        return similarity_dict["ratings"][item_ids]
