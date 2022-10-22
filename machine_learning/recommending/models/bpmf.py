raise NotImplementedError("There's something wrong with this implementation")

from typing import Literal

import einops
import numba
import numpy as np
import scipy
import torch

from tqdm.auto import tqdm

from my_tools.models import WandbLoggerMixin

from ..interface import RecommenderModuleBase, FitExplicitInterfaceMixin


class BayesianPMF(RecommenderModuleBase, FitExplicitInterfaceMixin, WandbLoggerMixin):
    def __init__(
        self,
        n_feature_dimensions=10,
        burn_in_steps=500,
        keeper_steps=100,
        predictive_explicit_precision=1,
        features_hyper_precision_coefficient=1,
        **kwargs,
    ):
        """
        This model approximates not just a single point estimate in parameter space,
        but a whole distribution, therefore optimizing over hyperparameter space.
        It accomplishes this by MCMC sampling to estimate intractable integral
        that represents the probability of observing the train data:
            P(X) = \\int P(X|Z)P(Z)dZ \\approx \\sum_i^n P(X|Z_i) / n
        where Z_i are sampled from some prior distribution, and uses it to maximize P(Z|X):
            P(Z|X) = P(X|Z)P(Z) / P(X)
        In the end though we still need a single estimate point, which
        is usually the mean or the mode of distribution P(Z|X).

        See https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf
        https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
        https://en.wikipedia.org/wiki/Normal-Wishart_distribution

        :param n_feature_dimensions: size of latent dimension
        :param burn_in_steps: first samples are usually pretty biased,
        so we skip them in hopes that after that model has substantially converged
        :param keeper_steps: the steps after burn in to run for
        :param predictive_explicit_precision: the shared precision of normal distributions
        (whose means will be saved in step_explicit_normal_distribution_means),
        which comprise the uniformly weighted mixture that is the estimation of
        P(Y|X)=f(Z)P(Z|X), where Y=f(z) are the predicted ratings
        :param features_hyper_precision_coefficient: hyper-hyperparameters precision
        distribution, the higher it is, the more regularized the model will be;
        it corresponds to lambda in the last wiki link
        """

        super().__init__(**kwargs)
        self.n_feature_dimensions = n_feature_dimensions
        self.burn_in_steps = burn_in_steps
        self.keeper_steps = keeper_steps
        self.alpha = predictive_explicit_precision

        # Coefficients for distribution of hyperparameters
        self.beta = features_hyper_precision_coefficient
        self.hyper_mean = self.init_hyper_mean()
        self.degrees_of_freedom = self.init_degrees_of_freedom()
        self.scale_matrix = self.init_scale_matrix()
        assert self.scale_matrix.shape == (
            self.n_feature_dimensions,
            self.n_feature_dimensions,
        )
        self.hyper_variance = np.linalg.inv(self.scale_matrix)

        self.user_features = self.init_user_features()
        self.item_features = self.init_item_features()

        # The output of the model – means of normal distributions
        # with precision predictive_explicit_precision, the mixture
        # of which is the estimated distribution of explicit feedback.
        self.step_explicit_normal_distribution_means = torch.nn.Parameter(
            torch.zeros(keeper_steps, self.n_users, self.n_items), requires_grad=False
        )

        self.implicit = None
        self.explicit = None
        self.alpha_x_implicit_x_explicit = None

    def init_hyper_mean(self):
        return 0

    def init_degrees_of_freedom(self):
        return self.n_feature_dimensions

    def init_scale_matrix(self):
        return np.eye(self.n_feature_dimensions)

    def init_features(self, size):
        return np.random.multivariate_normal(
            mean=np.full(self.n_feature_dimensions, self.hyper_mean),
            cov=self.hyper_variance / self.n_feature_dimensions,
            size=size,
        )

    def init_user_features(self):
        """Override this method if you want to finetune a fitted MF model."""
        return self.init_features(self.n_users)

    def init_item_features(self):
        """Override this method if you want to finetune a fitted MF model."""
        return self.init_features(self.n_items)

    def fit(self, explicit):
        self.explicit = explicit
        self.implicit = explicit > 0
        self.alpha_x_implicit_x_explicit = self.alpha * self.implicit.multiply(explicit)

        for _ in tqdm(range(self.burn_in_steps), "Sampling burn in steps"):
            self.step()

        for step in tqdm(range(self.keeper_steps), "Sampling keeper steps"):
            step_means = torch.from_numpy(self.step())
            self.step_explicit_normal_distribution_means[step] = step_means

    def step(self):
        user_hyperparameters = self.sample_hyperparameters(self.user_features)
        item_hyperparameters = self.sample_hyperparameters(self.item_features)

        user_features = self.sample_features(
            self.item_features,
            self.implicit,
            self.alpha_x_implicit_x_explicit,
            **user_hyperparameters,
        )
        item_features = self.sample_features(
            self.user_features,
            self.implicit.T.tocsr(),
            self.alpha_x_implicit_x_explicit.T.tocsr(),
            **item_hyperparameters,
        )

        self.user_features = user_features
        self.item_features = item_features

        step_explicit_mean = user_features @ item_features.T

        self.log(
            dict(
                mse=((step_explicit_mean - self.explicit.toarray()) ** 2).mean(),
                mse_filtered=(
                    (
                        self.implicit.multiply(step_explicit_mean).toarray()
                        - self.explicit.toarray()
                    )
                    ** 2
                ).mean(),
            )
        )
        return step_explicit_mean

    def sample_hyperparameters(self, features):
        n = features.shape[0]
        features_mean = np.einsum("i d -> d", features) / n
        sample_variance = np.einsum(
            "i d, i k -> d k", features - features_mean, features - features_mean
        )

        beta_estimation = self.beta + n
        degrees_of_freedom_estimation = self.degrees_of_freedom + n
        hyper_mean_estimation = (self.beta * self.hyper_mean + n * features_mean) / (
            self.beta + n
        )
        biased_variance_estimation = np.einsum(
            "d, k -> d k",
            features_mean - self.hyper_mean,
            features_mean - self.hyper_mean,
        )
        scale_matrix_estimation = np.linalg.inv(
            self.hyper_variance
            + sample_variance
            + (self.beta * n) / (self.beta + n) * biased_variance_estimation
        )

        hyper_variance_inverse = scipy.stats.wishart.rvs(
            df=degrees_of_freedom_estimation, scale=scale_matrix_estimation
        )
        hyper_mean = np.random.multivariate_normal(
            hyper_mean_estimation,
            np.linalg.inv(beta_estimation * hyper_variance_inverse),
        )
        return dict(
            hyper_variance_inverse=hyper_variance_inverse, hyper_mean=hyper_mean
        )

    def sample_features(
        self,
        fixed_features,
        implicit,
        alpha_x_implicit_x_ratings,
        hyper_mean,
        hyper_variance_inverse,
    ):
        n = implicit.shape[0]
        means = np.empty((n, self.n_feature_dimensions))
        variances = np.empty((n, self.n_feature_dimensions, self.n_feature_dimensions))
        alpha_x_features_x_features_t = self.alpha * np.einsum(
            "j d, j k -> j d k", fixed_features, fixed_features
        )

        precalculated = (
            hyper_variance_inverse @ hyper_mean
            + alpha_x_implicit_x_ratings @ fixed_features
        )

        self.jit_sample_features(
            precalculated,
            alpha_x_features_x_features_t,
            hyper_variance_inverse,
            implicit.indices,
            implicit.indptr,
            means,
            variances,
        )

        distribution = torch.distributions.MultivariateNormal(
            loc=torch.from_numpy(means), covariance_matrix=torch.from_numpy(variances)
        )
        features = distribution.sample().numpy()
        return features

    @staticmethod
    @numba.njit(parallel=True)
    def jit_sample_features(
        precalculated,
        alpha_x_features_x_features_t,
        hyper_variance_inverse,
        indices,
        indptr,
        means,
        variances,
    ):
        hyper_variance = np.linalg.inv(hyper_variance_inverse)
        for i in numba.prange(len(means)):
            col_indices = indices[indptr[i] : indptr[i + 1]]
            if col_indices.size:
                var = np.linalg.inv(
                    hyper_variance_inverse
                    + alpha_x_features_x_features_t[col_indices].sum(axis=0)
                )
            else:
                var = hyper_variance
            mean = var @ precalculated[i]
            means[i] = mean
            variances[i] = var

    def aggregate_prediction_mean(self, user_ids):
        return einops.reduce(
            self.step_explicit_normal_distribution_means[:, user_ids, :],
            "step user item -> user item",
            "mean",
        )

    def aggregate_prediction_mode_approximation(self, user_ids):
        # Caution: very memory hungry.
        means = self.step_explicit_normal_distribution_means[:, user_ids, :]

        normal = torch.distributions.Normal(loc=means, scale=self.alpha**-0.5)
        pdfs_at_means = normal.log_prob(
            einops.repeat(means, f"step user item -> step {means.shape[0]} user item")
        )
        step_ids_with_highest_mean_pdf = torch.logsumexp(pdfs_at_means, dim=0).argmax(0)
        mode_means = torch.gather(means, 0, step_ids_with_highest_mean_pdf[None])[0]
        return mode_means

    def forward(self, user_ids, aggregation_method: Literal["mean", "mode"] = "mean"):
        match aggregation_method:
            case "mean":
                ratings = self.aggregate_prediction_mean(user_ids)
            case "mode":
                ratings = self.aggregate_prediction_mode_approximation(user_ids)
            case _:
                raise ValueError(f"Unknown aggregation method {aggregation_method}")
        return ratings
