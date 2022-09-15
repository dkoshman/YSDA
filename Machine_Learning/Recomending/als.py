import numba
import numpy as np
import pandas as pd
import scipy
import torch

from tqdm.auto import tqdm
from typing_extensions import Literal


class ALS:
    def __init__(self, epochs=10, latent_dimension_size=10, regularization_lambda=1e-2):
        self.epochs = epochs
        self.latent_dimension_size = latent_dimension_size
        self.regularization_lambda = regularization_lambda

    @staticmethod
    def parameter_init(*dimensions):
        return np.random.randn(*dimensions).astype(np.float32)

    def init(self, n_users, n_items):
        self.user_factors = self.parameter_init(n_users, self.latent_dimension_size)
        self.item_factors = self.parameter_init(n_items, self.latent_dimension_size)

    def check_feedback(self, feedback):
        if (feedback < 0).sum():
            raise ValueError(
                "This als implementation works only with non negative feedback"
            )

    def from_implicit_feedback(self, feedback, confidence_alpha=40):
        preference = feedback > 0
        self.preference = preference
        self.confidence_minus_1 = feedback * confidence_alpha
        self.confidence_minus_1.eliminate_zeros()
        self.confidence_x_preference = (
            self.confidence_minus_1.multiply(preference) + preference
        )

    def from_explicit_feedback(self, feedback):
        raise NotImplementedError(
            "Confidence and confidence - 1 might not be sparse,"
            "need to provide a way to efficiently compute for"
            " concrete instance of feedback."
        )
        preference = feedback > 0
        confidence_minus_1 = feedback - 1

    def sparse_iterator(self, transpose=False):
        confidence_minus_1 = self.confidence_minus_1
        confidence_x_preference = self.confidence_x_preference

        assert confidence_minus_1.indices.shape == confidence_x_preference.indices.shape
        assert (confidence_minus_1.indices == confidence_x_preference.indices).all()
        assert confidence_minus_1.indptr.shape == confidence_x_preference.indptr.shape
        assert (confidence_minus_1.indptr == confidence_x_preference.indptr).all()

        if transpose:
            confidence_minus_1 = confidence_minus_1.tocsc()
            confidence_x_preference = confidence_x_preference.tocsc()

        cm1_data = confidence_minus_1.data
        cp_data = confidence_x_preference.data
        indices = confidence_minus_1.indices
        indptr = confidence_minus_1.indptr
        for ptr_id, (ind_begin, ind_end) in enumerate(zip(indptr, indptr[1:])):
            ind_slice = slice(ind_begin, ind_end)
            yield ptr_id, indices[ind_slice], cm1_data[ind_slice], cp_data[ind_slice]

    def least_squares_optimization_with_fixed_factors(
        self,
        fixed=Literal["users", "items"],
    ):
        kwargs = self.preprocess_optimization_args(fixed=fixed)
        self.analytic_optimum_dispatcher(**kwargs)

    def preprocess_optimization_args(self, *, fixed):
        if fixed == "items":
            X = self.user_factors
            Y = self.item_factors
            sparse_iterator = self.sparse_iterator()

        elif fixed == "users":
            X = self.item_factors
            Y = self.user_factors
            sparse_iterator = self.sparse_iterator(transpose=True)

        else:
            raise ValueError

        return dict(X=X, Y=Y, sparse_iterator=sparse_iterator, fixed=fixed)

    def analytic_optimum_dispatcher(self, X, Y, sparse_iterator):
        lambda_I = self.regularization_lambda * np.eye(Y.shape[1])
        YtY_plus_lambdaI = Y.T @ Y + lambda_I

        for row_id, col_indices, cm1, cp in sparse_iterator:
            if col_indices.size == 0:
                X[row_id] = 0
                continue

            y = Y[col_indices]
            YtCY_plus_lambdaI = (y.T * cm1) @ y + YtY_plus_lambdaI
            X[row_id] = np.linalg.inv(YtCY_plus_lambdaI) @ (y.T @ cp)

    def fit(
        self,
        feedback: scipy.sparse.csr.csr_matrix,
        kind: Literal["implicit", "explicit"] = "implicit",
    ):
        self.check_feedback(feedback)

        if kind == "implicit":
            self.from_implicit_feedback(feedback)
        elif kind == "explicit":
            self.from_explicit_feedback(feedback)
        else:
            raise ValueError

        self.init(*feedback.shape)

        for epoch in tqdm(range(self.epochs), "Alternating"):
            self.least_squares_optimization_with_fixed_factors(fixed="items")
            self.least_squares_optimization_with_fixed_factors(fixed="users")

    def topk_recommendations(self, user_ids, topk):
        user_factors = torch.tensor(self.user_factors[user_ids], device="cuda")
        item_factors = torch.tensor(self.item_factors, device="cuda")

        ratings = user_factors @ item_factors.T
        ratings_of_recommended_items, ids_of_recommended_items = torch.topk(
            ratings, topk
        )

        ratings_of_recommended_items = ratings_of_recommended_items.cpu().numpy()
        ids_of_recommended_items = ids_of_recommended_items.cpu().numpy()

        del user_factors, item_factors, ratings

        return ratings_of_recommended_items, ids_of_recommended_items

    def recommend(self, user_ids, topk=20, explain=False):
        (
            ratings_of_recommended_items,
            ids_of_recommended_items,
        ) = self.topk_recommendations(user_ids, topk)

        if explain:
            return self.explain_recommendations(
                user_ids, ids_of_recommended_items, ratings_of_recommended_items
            )

        return ids_of_recommended_items

    def explain_recommendations(self, user_ids, item_ids, ratings):
        recommendations = pd.DataFrame(
            item_ids, index=user_ids, columns=range(item_ids.shape[1])
        )
        YtY_plus_lambdaI = item_factors_regularization_term = (
            self.item_factors.T @ self.item_factors
            + self.regularization_lambda * np.eye(self.item_factors.shape[1])
        )
        explanations = {}

        for (user_id, recommended_item_ids, user_ratings) in zip(
            user_ids, item_ids, ratings
        ):
            relative_row_ids, liked_item_ids = self.preference[user_id].nonzero()
            if liked_item_ids.size == 0:
                explanations[user_id] = None
                continue

            confidence_minus_1 = (
                self.confidence_minus_1[user_id, liked_item_ids].toarray().squeeze()
            )

            Y = liked_items_factors = self.item_factors[liked_item_ids]
            Y_recommended = recommended_items_factors = self.item_factors[
                recommended_item_ids
            ]
            user_latent_weight = YtCY = np.linalg.inv(
                (Y.T * confidence_minus_1) @ Y + YtY_plus_lambdaI
            )
            similarity = Y_recommended @ user_latent_weight @ Y.T

            explanations[user_id] = {
                "ratings": pd.Series(
                    user_ratings,
                    recommended_item_ids,
                    name=f"predicted relevance",
                ).rename_axis(index="recommended items for user"),
                "similarity": pd.DataFrame(
                    similarity, index=recommended_item_ids, columns=liked_item_ids
                ).rename_axis(
                    index=f"recommended items for user",
                    columns=f"liked items by user",
                ),
                "confidence": pd.Series(
                    confidence_minus_1 + 1,
                    liked_item_ids,
                    name=f"estimated confidence that user likes items",
                ).rename_axis(index="liked items by user"),
            }

        return recommendations, explanations


class ALS_jit(ALS):
    def __init__(self, *args, num_threads=8, **kwargs):
        super().__init__(*args, **kwargs)
        numba.config.THREADING_LAYER = "threadsafe"
        numba.set_num_threads(num_threads)

    def preprocess_optimization_args(self, **kwargs):
        kwargs = super().preprocess_optimization_args(**kwargs)

        X = kwargs["X"]
        Y = kwargs["Y"]
        fixed = kwargs["fixed"]

        if fixed == "items":
            confidence_minus_1 = self.confidence_minus_1
            confidence_x_preference = self.confidence_x_preference

        elif fixed == "users":
            confidence_minus_1 = self.confidence_minus_1.tocsc()
            confidence_x_preference = self.confidence_x_preference.tocsc()

        else:
            raise ValueError("Unknown fixed value")

        cm1_data = confidence_minus_1.data.astype(np.float32)
        cp_data = confidence_x_preference.data.astype(np.float32)
        indices = confidence_minus_1.indices
        indptr = confidence_minus_1.indptr

        lambda_I = self.regularization_lambda * np.eye(Y.shape[1], dtype=np.float32)
        YtY_plus_lambdaI = Y.T @ Y + lambda_I

        return dict(
            X=X,
            Y=Y,
            cm1_data=cm1_data,
            cp_data=cp_data,
            indices=indices,
            indptr=indptr,
            YtY_plus_lambdaI=YtY_plus_lambdaI,
        )

    @staticmethod
    @numba.njit(parallel=True)
    def analytic_optimum_dispatcher(
        X, Y, cm1_data, cp_data, indices, indptr, YtY_plus_lambdaI
    ):
        for row_id in numba.prange(X.shape[0]):
            ind_slice = slice(indptr[row_id], indptr[row_id + 1])
            if ind_slice.start == ind_slice.stop:
                X[row_id] = 0
                continue

            cm1 = cm1_data[ind_slice]
            cp = cp_data[ind_slice]
            col_indices = indices[ind_slice]
            y = Y[col_indices]

            YtCY_plus_lambdaI = (y.T * cm1) @ y + YtY_plus_lambdaI
            X[row_id] = np.linalg.inv(YtCY_plus_lambdaI).astype(np.float32) @ (y.T @ cp)


class ALS_biased_jit(ALS_jit):
    def init(self, n_users, n_items):
        self.bias = self.parameter_init(1)
        self.user_factors = self.parameter_init(n_users, self.latent_dimension_size + 2)
        self.user_factors[:, 0] = 1
        self.item_factors = self.parameter_init(n_items, self.latent_dimension_size + 2)
        self.item_factors[:, 1] = 1

    def preprocess_optimization_args(self, **kwargs):
        fixed = kwargs["fixed"]
        kwargs = super().preprocess_optimization_args(**kwargs)

        if fixed == "items":
            X_constant_latent_index = 0

        elif fixed == "users":
            X_constant_latent_index = 1

        YtY_plus_lambdaI = kwargs["YtY_plus_lambdaI"]
        YtY_plus_lambdaI[
            X_constant_latent_index, X_constant_latent_index
        ] -= self.regularization_lambda

        XY_sum = (self.user_factors.sum(axis=0) * self.item_factors.sum(axis=0)).sum()

        self.bias[:] = (
            self.confidence_x_preference.sum()
            - XY_sum
            - (self.user_factors.T @ self.confidence_minus_1 @ self.item_factors).sum()
        ) / (self.confidence_minus_1.sum() + np.prod(self.confidence_minus_1.shape))

        return {
            **{
                k: kwargs[k]
                for k in [
                    "X",
                    "Y",
                    "cm1_data",
                    "cp_data",
                    "indices",
                    "indptr",
                ]
            },
            **{
                "YtY_plus_lambdaI": YtY_plus_lambdaI,
                "bias": self.bias,
                "X_constant_latent_index": X_constant_latent_index,
            },
        }

    @staticmethod
    @numba.njit(parallel=True)
    def analytic_optimum_dispatcher(
        X,
        Y,
        cm1_data,
        cp_data,
        indices,
        indptr,
        YtY_plus_lambdaI,
        bias,
        X_constant_latent_index,
    ):
        Yt_bias = bias[0] * Y.sum(axis=0)
        for row_id in numba.prange(X.shape[0]):
            ind_slice = slice(indptr[row_id], indptr[row_id + 1])
            if ind_slice.start == ind_slice.stop:
                X[row_id] = 0
                continue

            cm1 = cm1_data[ind_slice]
            cp = cp_data[ind_slice]
            col_indices = indices[ind_slice]
            y = Y[col_indices]

            YtCY_plus_lambdaI = (y.T * cm1) @ y + YtY_plus_lambdaI
            YtCP_minus_biasYtC = y.T @ (cp - bias[0] * cm1) - Yt_bias
            X[row_id] = np.linalg.inv(YtCY_plus_lambdaI).astype(np.float32) @ (
                YtCP_minus_biasYtC
            )

        X[:, X_constant_latent_index] = 1

    def explain_recommendations(self, user_ids, item_ids, ratings):
        raise NotImplementedError("Need to add bias to existing implementation")
