import io
import pickle
from typing import Literal

import implicit
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

from my_tools.utils import build_class

from ..interface import RecommenderModuleBase, FitExplicitInterfaceMixin


class RandomRecommender(RecommenderModuleBase, FitExplicitInterfaceMixin):
    def fit(self):
        pass

    def forward(self, user_ids, item_ids):
        ratings = torch.randn(len(user_ids), len(item_ids))
        return ratings


class PopularRecommender(RecommenderModuleBase, FitExplicitInterfaceMixin):
    def __init__(self, explicit=None):
        super().__init__(explicit=explicit)
        self.register_buffer(name="items_count", tensor=torch.zeros(self.n_items))

    def fit(self):
        implicit_feedback = self.to_scipy_coo(self.explicit) > 0
        self.items_count = torch.from_numpy(implicit_feedback.sum(axis=0).A.squeeze())

    def forward(self, user_ids, item_ids):
        ratings = self.items_count[item_ids].repeat(len(user_ids), 1)
        return ratings.to(torch.float32)


class SVDRecommender(RecommenderModuleBase, FitExplicitInterfaceMixin):
    def __init__(self, explicit=None, n_components=10):
        super().__init__(explicit=explicit)
        self.model = TruncatedSVD(n_components=n_components)

    def fit(self):
        self.model.fit(self.to_scipy_coo(self.explicit))

    def forward(self, user_ids, item_ids):
        explicit_feedback = self.to_scipy_coo(self.explicit).tocsr()
        embedding = self.model.transform(explicit_feedback[user_ids].A)
        ratings = self.model.inverse_transform(embedding)[:, item_ids]
        return torch.from_numpy(ratings).to(torch.float32)

    def get_extra_state(self):
        pickled_bytes = pickle.dumps(self.model)
        return pickled_bytes

    def set_extra_state(self, pickled_bytes):
        self.model = pickle.load(io.BytesIO(pickled_bytes))


class ImplicitRecommenderBase(RecommenderModuleBase, FitExplicitInterfaceMixin):
    def __init__(self, *, explicit=None, implicit_model, implicit_kwargs=None):
        super().__init__(explicit=explicit)
        self.model = build_class(
            module_candidates=[
                implicit.nearest_neighbours,
                implicit.als,
                implicit.bpr,
                implicit.lmf,
            ],
            class_name=implicit_model,
            **(implicit_kwargs or {}),
        )

    def fit(self):
        self.explicit = self.explicit.to(torch.float32)
        self.model.fit(self.to_scipy_coo(self.explicit))

    def forward(self, user_ids, item_ids):
        explicit_feedback = self.to_scipy_coo(self.explicit).tocsr()
        recommended_item_ids, item_ratings = self.model.recommend(
            userid=user_ids.numpy(),
            user_items=explicit_feedback[user_ids],
            N=self.n_items,
            filter_already_liked_items=False,
        )
        ratings = np.full(
            shape=(len(user_ids), self.n_items + 1),
            fill_value=np.finfo(np.float32).min,
        )
        np.put_along_axis(
            arr=ratings, indices=recommended_item_ids, values=item_ratings, axis=1
        )
        ratings = torch.from_numpy(ratings[:, :-1])
        return ratings[:, item_ids]

    @torch.inference_mode()
    def recommend(
        self, user_ids, n_recommendations=10, filter_already_liked_items=True
    ):
        item_ids, item_ratings = self.model.recommend(
            userid=user_ids.numpy(),
            user_items=self.to_scipy_coo(self.explicit).tocsr()[user_ids],
            N=n_recommendations,
            filter_already_liked_items=filter_already_liked_items,
        )
        return torch.from_numpy(item_ids).to(torch.int64)

    def get_extra_state(self):
        bytesio = io.BytesIO()
        self.model.save(bytesio)
        return bytesio

    def set_extra_state(self, bytesio):
        bytesio.seek(0)
        self.model = self.model.load(bytesio)


class ImplicitNearestNeighborsRecommender(ImplicitRecommenderBase):
    def __init__(
        self,
        explicit=None,
        implicit_model: Literal[
            "BM25Recommender", "CosineRecommender", "TFIDFRecommender"
        ] = "BM25Recommender",
        num_neighbors=20,
        num_threads=0,
    ):
        super().__init__(
            explicit=explicit,
            implicit_model=implicit_model,
            implicit_kwargs=dict(K=num_neighbors, num_threads=num_threads),
        )

    def online_recommend(
        self,
        users_explicit,
        n_recommendations=None,
    ) -> torch.IntTensor:
        users_explicit = self.to_torch_coo(users_explicit)
        item_ids, item_ratings = self.model.recommend(
            userid=np.arange(users_explicit.shape[0]),
            user_items=self.to_scipy_coo(users_explicit.to(torch.float32)).tocsr(),
            N=n_recommendations or self.n_items,
        )
        return torch.from_numpy(item_ids).to(torch.int64)


class ImplicitMatrixFactorizationRecommender(ImplicitRecommenderBase):
    def __init__(
        self,
        explicit=None,
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
        **implicit_kwargs,
    ):
        implicit_kwargs.update(
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
            explicit=explicit,
            implicit_model=implicit_model,
            implicit_kwargs=implicit_kwargs,
        )
