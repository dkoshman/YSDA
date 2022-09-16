from typing import Literal

import einops
import numba
import numpy as np
import scipy
import torch
from tqdm.auto import tqdm


class BayesianPMF:
    def __init__(
        self,
        explicit: scipy.sparse.csr_matrix,
        n_feature_dimensions,
        burn_in_steps,
        keeper_steps,
        predictive_explicit_precision,
        features_hyper_precision_coefficient,
    ):
        """See https://www.cs.toronto.edu/~amnih/papers/bpmf.pdf
        https://en.wikipedia.org/wiki/Normal-Wishart_distribution"""

        self.explicit = explicit
        self.implicit = explicit != 0
        self.n_users, self.n_items = explicit.shape
        self.n_feature_dimensions = n_feature_dimensions
        self.burn_in_steps = burn_in_steps
        self.keeper_steps = keeper_steps
        self.alpha = predictive_explicit_precision
        self.alpha_x_implicit_x_explicit = self.alpha * self.implicit.multiply(explicit)

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
        # with precision $predictive_explicit_precision$, the mixture
        # of which is the estimated distribution of explicit feedback.
        self.step_explicit_normal_distribution_means = []

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
        return self.init_features(self.n_users)

    def init_item_features(self):
        return self.init_features(self.n_items)

    def fit(self):
        for step in tqdm(range(self.burn_in_steps + self.keeper_steps), "Sampling"):
            step_mean = self.step()
            if step > self.burn_in_steps:
                self.step_explicit_normal_distribution_means.append(step_mean)

        self.finalize()

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

    def finalize(self):
        self.step_explicit_normal_distribution_means = np.array(
            self.step_explicit_normal_distribution_means
        )

    def aggregate_prediction_mean(self, user_ids):
        return einops.reduce(
            self.step_explicit_normal_distribution_means[:, user_ids, :],
            "step user item -> user item",
            "mean",
        )

    def aggregate_prediction_mode_approximation(self, user_ids):
        # Caution: very memory hungry
        means = torch.from_numpy(
            self.step_explicit_normal_distribution_means[:, user_ids, :]
        )
        normal = torch.distributions.Normal(loc=means, scale=self.alpha**-0.5)
        pdfs_at_means = normal.log_prob(
            einops.repeat(means, f"step user item -> step {means.shape[0]} user item")
        )
        step_ids_with_highest_mean_pdf = torch.logsumexp(pdfs_at_means, dim=0).argmax(0)
        mode_means = torch.gather(means, 0, step_ids_with_highest_mean_pdf[None])[0]
        return mode_means

    def recommend(self, user_ids, aggregation_method: Literal["mean", "mode"] = "mode"):
        # TODO: plot the gaussian mixture, run sweep
        if aggregation_method == "mean":
            return self.aggregate_prediction_mean(user_ids)
        elif aggregation_method == "mode":
            return self.aggregate_prediction_mode_approximation(user_ids)
        else:
            raise ValueError(f"Unknown aggregation method {aggregation_method}")
