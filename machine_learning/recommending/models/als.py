import abc
import io

import numba
import numpy as np
import pandas as pd
import torch
import wandb
from scipy.sparse import spmatrix


from pandas.io.formats.style import Styler
from tqdm.auto import tqdm
from typing_extensions import Literal

from my_tools.models import WandbLoggerMixin

from ..interface import (
    RecommenderModuleBase,
    FitExplicitInterfaceMixin,
    ExplanationMixin,
)
from ..utils import build_weight, build_bias


class ALSInterface:
    """Skeleton to expose main als logic."""

    @abc.abstractmethod
    def fit(self):
        """Where "alternating" in als comes from."""
        for epoch in range(10):
            self.least_squares_optimization_with_fixed_factors(fixed="items")
            self.least_squares_optimization_with_fixed_factors(fixed="users")

    def least_squares_optimization_with_fixed_factors(
        self, fixed: Literal["users", "items"]
    ):
        """Decoupler function to enable dispatcher independence on what features are currently fixed"""
        kwargs = self.preprocess_optimization_args(fixed=fixed)
        self.analytic_optimum_dispatcher(**kwargs)

    @abc.abstractmethod
    def preprocess_optimization_args(self, *, fixed: Literal["users", "items"]) -> dict:
        """
        Prepare arguments for dispatcher to facilitate its
        independence on what features are currently fixed.
        """

    @abc.abstractmethod
    def analytic_optimum_dispatcher(self, **kwargs) -> None:
        """
        Main optimization logic to calculate analytic
        MSE minimum with one of users or items features fixed.
        This method is agnostic about what features are fixed.
        Where "least squares" in als comes from.
        """


class ALS(
    ALSInterface,
    RecommenderModuleBase,
    FitExplicitInterfaceMixin,
    WandbLoggerMixin,
    ExplanationMixin,
):
    def __init__(
        self,
        epochs=10,
        latent_dimension_size=10,
        regularization_lambda=100,
        confidence_alpha=10,
        lambda_decay=0.75,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epochs = epochs
        self.regularization_lambda = regularization_lambda
        self.confidence_alpha = confidence_alpha
        self.lambda_decay = lambda_decay

        self.user_factors = build_weight(self.n_users, latent_dimension_size)
        self.user_factors.requires_grad = False
        self.item_factors = build_weight(self.n_items, latent_dimension_size)
        self.item_factors.requires_grad = False

        self.confidence_x_preference = None
        self.confidence_minus_1 = None
        self.preference = None

    def init_preference_confidence(self, explicit_feedback: spmatrix):
        implicit_feedback = (explicit_feedback > 0).astype(np.float32)
        self.preference = implicit_feedback
        self.confidence_minus_1 = implicit_feedback * self.confidence_alpha
        self.confidence_minus_1.eliminate_zeros()
        self.confidence_x_preference = (
            self.confidence_minus_1.multiply(self.preference) + self.preference
        )

    def fit(self):
        self.init_preference_confidence(self.to_scipy_coo(self.explicit))
        for _ in tqdm(range(self.epochs), "Alternating"):
            self.on_train_epoch_start()
            self.least_squares_optimization_with_fixed_factors(fixed="items")
            self.least_squares_optimization_with_fixed_factors(fixed="users")
            self.regularization_lambda *= self.lambda_decay

    def on_train_epoch_start(self):
        self.log(dict(regularization_lambda=self.regularization_lambda))

    def least_squares_optimization_with_fixed_factors(
        self, fixed: Literal["users", "items"]
    ):
        kwargs = self.preprocess_optimization_args(fixed=fixed)
        self.analytic_optimum_dispatcher(**kwargs)

    def preprocess_optimization_args(self, *, fixed):
        if fixed == "items":
            X = self.user_factors.numpy()
            Y = self.item_factors.numpy()
            sparse_iterator = self.sparse_iterator()
        elif fixed == "users":
            X = self.item_factors.numpy()
            Y = self.user_factors.numpy()
            sparse_iterator = self.sparse_iterator(transpose=True)
        else:
            raise ValueError
        return dict(X=X, Y=Y, sparse_iterator=sparse_iterator, fixed=fixed)

    def analytic_optimum_dispatcher(self, X, Y, sparse_iterator, fixed):
        """This implementation and the next helper method are left here mainly for
        completeness's sake, for more efficient and less convoluted implementation see ALSJIT."""
        lambda_I = self.regularization_lambda * np.eye(Y.shape[1])
        YtY_plus_lambdaI = Y.T @ Y + lambda_I

        for row_id, col_indices, cm1, cp in sparse_iterator:
            if col_indices.size == 0:
                X[row_id] = 0
                continue

            y = Y[col_indices]
            YtCY_plus_lambdaI = (y.T * cm1) @ y + YtY_plus_lambdaI
            X[row_id] = np.linalg.inv(YtCY_plus_lambdaI) @ (y.T @ cp)

    def sparse_iterator(self, transpose=False):
        confidence_minus_1 = self.confidence_minus_1
        confidence_x_preference = self.confidence_x_preference

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

    def forward(self, user_ids, item_ids):
        ratings = self.user_factors[user_ids] @ self.item_factors[item_ids].T
        return ratings

    def explain_recommendations_for_user(
        self,
        user_id=None,
        user_explicit=None,
        n_recommendations=10,
        log=False,
        logging_prefix="",
    ) -> Styler:
        if user_id is None:
            raise NotImplementedError(
                "Online explanations are not implemented for als."
            )
        recommendations = self.recommend(
            user_ids=torch.IntTensor([user_id]), n_recommendations=n_recommendations
        ).numpy()
        user_ratings = self(user_ids=torch.tensor([user_id]), item_ids=recommendations)[
            0
        ].numpy()
        YtY_plus_lambdaI = item_factors_regularization_term = (
            self.item_factors.T @ self.item_factors
            + self.regularization_lambda * np.eye(self.item_factors.shape[1])
        ).numpy()
        relative_row_ids, liked_item_ids = self.preference[user_id].nonzero()
        if liked_item_ids.size == 0:
            raise ValueError(
                "No personal recommendations generated, probably a cold user."
            )

        confidence_minus_1 = (
            self.confidence_minus_1[user_id, liked_item_ids].toarray().squeeze()
        )
        Y = liked_items_factors = self.item_factors[liked_item_ids].numpy()
        Y_recommended = recommended_items_factors = self.item_factors[
            recommendations
        ].numpy()
        user_latent_weight = YtCY = np.linalg.inv(
            (Y.T * confidence_minus_1) @ Y + YtY_plus_lambdaI
        )
        similarity = Y_recommended @ user_latent_weight @ Y.T

        confidence = confidence_minus_1 + 1

        explanations = {
            "ratings": pd.Series(
                user_ratings,
                recommendations,
                name=f"predicted relevance",
            ).rename_axis(index="recommended items for user"),
            "similarity": pd.DataFrame(
                similarity, index=recommendations, columns=liked_item_ids
            ).rename_axis(
                index=f"recommended items for user",
                columns=f"liked items by user",
            ),
            "confidence": pd.Series(
                confidence,
                liked_item_ids,
                name=f"estimated confidence that user likes items",
            ).rename_axis(index="liked items by user"),
        }

        dataframe = pd.concat(
            [explanations["similarity"], explanations["ratings"]], axis="columns"
        )
        dataframe = pd.concat(
            [dataframe.T, explanations["confidence"]], axis="columns"
        ).T.rename_axis(index="recommended items", columns="liked items")

        style = dataframe.style
        style.set_caption(
            f"Recommended items for user {user_id} based on personal similarity measure "
            "between items and estimated confidence in these similarities."
        ).set_table_styles(
            [
                dict(
                    selector="caption",
                    props=[
                        ("text-align", "center"),
                        ("font-size", "125%"),
                        ("color", "black"),
                    ],
                )
            ]
        )
        common_gradient_kwargs = dict(low=0.5, high=0.5)
        style = style.background_gradient(
            subset=(dataframe.index[-1], dataframe.columns[:-1]),
            axis="columns",
            **common_gradient_kwargs,
            cmap="YlOrRd",
        )
        style = style.background_gradient(
            subset=(dataframe.index[:-1], dataframe.columns[:-1]),
            **common_gradient_kwargs,
            cmap="coolwarm",
        )
        style = style.background_gradient(
            subset=(dataframe.index[:-1], dataframe.columns[-1]),
            **common_gradient_kwargs,
            cmap="YlOrRd",
        )
        if log:
            textio = io.TextIOWrapper(io.BytesIO())
            style.to_html(textio)
            textio.seek(0)
            wandb.log(
                {
                    logging_prefix
                    + " Recommendation explaining dataframe": wandb.Html(textio)
                }
            )
        return style


class ALSjit(ALS):
    def __init__(self, *args, num_threads=8, **kwargs):
        super().__init__(*args, **kwargs)
        numba.config.NUMBA_NUM_THREADS = 32
        numba.config.THREADING_LAYER = "threadsafe"
        numba.set_num_threads(num_threads)

    def preprocess_optimization_args(self, **kwargs):
        kwargs = super().preprocess_optimization_args(**kwargs)

        if kwargs["fixed"] == "items":
            confidence_minus_1 = self.confidence_minus_1
            confidence_x_preference = self.confidence_x_preference
        elif kwargs["fixed"] == "users":
            confidence_minus_1 = self.confidence_minus_1.tocsc()
            confidence_x_preference = self.confidence_x_preference.tocsc()
        else:
            raise ValueError("Unknown fixed value")

        cm1_data = confidence_minus_1.data.astype(np.float32)
        cp_data = confidence_x_preference.data.astype(np.float32)
        indices = confidence_minus_1.indices
        indptr = confidence_minus_1.indptr

        X = kwargs["X"]
        Y = kwargs["Y"]
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


class ALSjitBiased(ALSjit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bias = build_bias(1)
        self.bias.requires_grad = False

        self.user_factors = build_weight(
            self.user_factors.shape[0], self.user_factors.shape[1] + 2
        )
        self.user_factors.requires_grad = False
        self.user_factors[:, 0] = 1

        self.item_factors = build_weight(
            self.item_factors.shape[0], self.item_factors.shape[1] + 2
        )
        self.item_factors.requires_grad = False
        self.item_factors[:, 1] = 1

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.log(
            dict(
                bias=self.bias.item(),
                mean_user_bias=self.user_factors[:, 1].mean(),
                mean_item_bias=self.item_factors[:, 0].mean(),
            )
        )

    def preprocess_optimization_args(self, **kwargs):
        fixed = kwargs["fixed"]
        kwargs = super().preprocess_optimization_args(**kwargs)

        if fixed == "items":
            X_constant_latent_index = 0
        elif fixed == "users":
            X_constant_latent_index = 1
        else:
            raise ValueError("Unknown fixed value")

        YtY_plus_lambdaI = kwargs["YtY_plus_lambdaI"]
        YtY_plus_lambdaI[
            X_constant_latent_index, X_constant_latent_index
        ] -= self.regularization_lambda

        XY_sum = (
            (self.user_factors.sum(axis=0) * self.item_factors.sum(axis=0)).sum().item()
        )

        self.bias[:] = (
            self.confidence_x_preference.sum()
            - XY_sum
            - (
                self.user_factors.T.numpy()
                @ self.confidence_minus_1
                @ self.item_factors.numpy()
            ).sum()
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
                "bias": self.bias.numpy(),
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
            YtCP_minus_biasYtC = y.T @ (cp - bias[0] * cm1 - bias[0])
            X[row_id] = np.linalg.inv(YtCY_plus_lambdaI).astype(np.float32) @ (
                YtCP_minus_biasYtC
            )

        X[:, X_constant_latent_index] = 1

    def explain_recommendations_for_user(self, *args, **kwargs):
        raise NotImplementedError("Need to add bias to existing implementation")
