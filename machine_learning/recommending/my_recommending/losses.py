import einops
import numpy as np
import torch
import wandb

from .interface import RecommendingLossInterface
from .maths import kl_divergence, pairwise_difference


class MSELoss(RecommendingLossInterface):
    def __call__(
        self, explicit, model_ratings=None, model=None, user_ids=None, item_ids=None
    ):
        if model_ratings is None:
            model_ratings = model(user_ids=user_ids, item_ids=item_ids)
        explicit = explicit.to_dense()
        loss = ((model_ratings - explicit) ** 2).mean()
        return loss


class MSEL1Loss(RecommendingLossInterface):
    def __init__(self, explicit, ratings_deviation=1, mean_unobserved_rating=0):
        super().__init__(explicit)
        self.ratings_variance = ratings_deviation**2
        self.mean_unobserved_rating = mean_unobserved_rating

    def __call__(
        self, explicit, model_ratings=None, model=None, user_ids=None, item_ids=None
    ):
        if model_ratings is None:
            model_ratings = model(user_ids=user_ids, item_ids=item_ids)
        explicit = explicit.to_dense()
        observed_mask = explicit > 0
        loss = (
            observed_mask
            * (model_ratings - explicit) ** 2
            / (2 * self.ratings_variance)
        )
        loss += ~observed_mask * (model_ratings - self.mean_unobserved_rating).abs()
        return loss.sum()


class MSEProbabilityLoss(RecommendingLossInterface):
    """
    Recommender based on the following data generation model:
    1. Each user-item pair has a true, unknown, rating
    generated from normal distribution N_{r_ui, var}.
    2. Each rating has a probability to be observed p_ui = f(r_ui),
    where f is a non-decreasing function.
    That way higher quality items are more likely to be observed.

    Then the likelihood of given data sample is:
    likelihood(sample) =
    prod_observed_ui(p_ui * N_{r_ui, var}(rating_ui))) * prod_unobserved_ui(1 - p_ui)

    argmax(likelihood(sample)) =
    argmax(prod_observed_ui(f(r_ui) * N_{r_ui, var}(rating_ui))) * prod_unobserved_ui(1 - f(r_ui))) =
    argmax(sum_observed_ui(log f(r_ui) - (r_ui - rating_ui) ** 2 / (2 * var)) + sum_unobserved_ui(log(1 - f(r_ui))))

    The default for f is f(x) = c * x, where c is estimated as density / mean_rating
    """

    def __init__(self, explicit, ratings_deviation=1.0):
        super().__init__(explicit)
        self.ratings_variance = ratings_deviation**2
        mean_observed_rating = explicit.data.mean()
        density = len(explicit.data) / np.prod(explicit.shape)
        self.probability_to_be_observed_coef = density / mean_observed_rating

    def __call__(self, model, explicit, user_ids, item_ids):
        explicit = explicit.to_dense()
        observed_mask = explicit > 0
        ratings = model(user_ids=user_ids, item_ids=item_ids)
        loss = observed_mask * (ratings - explicit) ** 2 / (2 * self.ratings_variance)
        prob_to_be_observed = self.probability_to_be_observed_coef * ratings
        prob_to_be_observed = torch.clip(prob_to_be_observed, 1e-7, 1 - 1e-7)
        loss -= observed_mask * torch.log(prob_to_be_observed)
        loss -= ~observed_mask * torch.log(1 - prob_to_be_observed)
        return loss.sum()


def probability_of_user_preferences(
    ratings_of_supposedly_better_items, ratings_of_supposedly_worse_items
):
    """
    Given two rating matrices A, B of shape (n_users, n_items) corresponding
    to the same users, returns matrix C of shape (n_users, n_items, n_items)
    such that C[u, i, j] = Probability({user u prefers item A[u, i] over B[u, j]})
    """
    # Now P(item with rating 1 is more relevant than unrated item) =
    # = scaling_coefficient * sigmoid(1 - 0) = confidence_in_rating_quality
    probability = torch.sigmoid(
        pairwise_difference(
            ratings_of_supposedly_better_items, ratings_of_supposedly_worse_items
        )
    )
    return probability


class PersonalizedRankingLoss(RecommendingLossInterface):
    """Ranking based loss inspired by Bayesian Personalized Ranking paper."""

    def __init__(self, confidence=40):
        self.confidence = confidence

    def __call__(self, explicit, model_ratings):
        explicit = (explicit * (1 + self.confidence)).to_dense()
        model_ratings = model_ratings.to_dense()
        estimated_probs = torch.sigmoid(pairwise_difference(explicit, explicit))
        predicted_probs = torch.sigmoid(
            pairwise_difference(model_ratings, model_ratings)
        )
        # maybe in other order:
        # loss = kl_divergence(predicted_probs, estimated_probs)
        raise NotImplementedError
        # This is not a prob distribution!!!
        loss = kl_divergence(estimated_probs, predicted_probs)
        return loss / explicit.numel()

    # def __call__(self, *, explicit, model_ratings):
    #     explicit = explicit.coalesce()
    #     ids_of_users_who_rated_something = explicit.indices()[0].unique()
    #     if ids_of_users_who_rated_something.numel() == 0:
    #         return None
    #
    #     n_users, n_items = explicit.size()
    #     item_ids = torch.arange(n_items)
    #     items_mask = torch.full((n_items,), True)
    #     criterion = 0
    #     for i, user_id in enumerate(ids_of_users_who_rated_something):
    #         explicit_user_feedback = explicit[user_id]
    #         liked_item_ids = explicit_user_feedback.indices().squeeze()
    #         items_mask[:] = True
    #         items_mask[liked_item_ids] = False
    #         uninteracted_item_ids = item_ids[items_mask]
    #
    #         predicted_user_ratings = model_ratings[user_id]
    #
    #         user_ranking_correctness_criterion = -torch.log1p(
    #             torch.exp(
    #                 -(
    #                     predicted_user_ratings[liked_item_ids, None]
    #                     - predicted_user_ratings[uninteracted_item_ids]
    #                 )
    #             )
    #         ).mean()
    #         criterion += user_ranking_correctness_criterion
    #
    #     loss = -criterion / len(ids_of_users_who_rated_something)
    #     return loss


class PersonalizedRankingLossFast(RecommendingLossInterface):
    def __init__(self, max_rating=5, confidence_in_rating=0.9, memory_upper_bound=1e8):
        """
        :param max_rating: maximum value on the ratings scale
        :param confidence_in_rating: probability that item with best rating
        is more relevant than an unrated item
        :param memory_upper_bound: O(n) bound on size of tensor that can fit in the memory
        """
        self.scaling_coefficient = confidence_in_rating * (1 + torch.e**-max_rating)
        self.memory_upper_bound = memory_upper_bound

    def probability_of_user_preferences(
        self, ratings_of_supposedly_better_items, ratings_of_supposedly_worse_items
    ):
        """
        Given two rating matrices A, B of shape (n_users, n_items) corresponding
        to the same users, returns matrix C of shape (n_users, n_items, n_items)
        such that C[u, i, j] = Probability({user u prefers item A[u, i] over B[u, j]})
        """
        # Now P(item with rating 1 is more relevant than unrated item) =
        # = scaling_coefficient * sigmoid(1 - 0) = confidence_in_rating_quality
        probability = torch.sigmoid(
            pairwise_difference(
                ratings_of_supposedly_better_items, ratings_of_supposedly_worse_items
            )
        )
        return self.scaling_coefficient * probability

    @staticmethod
    def kl_divergence(
        probability_of_user_preferences, predicted_probability_of_user_preferences
    ):
        return -probability_of_user_preferences * torch.log(
            predicted_probability_of_user_preferences
        )

    def dense_cutout_kl_divergence(
        self, explicit_dense_cutout, model_ratings_dense_cutout
    ):
        probability_of_user_preferences = self.probability_of_user_preferences(
            explicit_dense_cutout, explicit_dense_cutout
        )
        predicted_probability_of_user_preferences = (
            self.probability_of_user_preferences(
                model_ratings_dense_cutout, model_ratings_dense_cutout
            )
        )
        kl_divergence = self.kl_divergence(
            probability_of_user_preferences,
            predicted_probability_of_user_preferences,
        )
        return kl_divergence

    def rated_unrated_kl_divergence(
        self,
        explicit_dense_cutout,
        model_ratings_dense_cutout,
        unrated_items_model_ratings,
    ):
        probability_of_user_preferences = self.probability_of_user_preferences(
            explicit_dense_cutout, torch.zeros_like(unrated_items_model_ratings)
        )
        predicted_probability_of_user_preferences = (
            self.probability_of_user_preferences(
                model_ratings_dense_cutout, unrated_items_model_ratings
            )
        )
        kl_divergence = self.kl_divergence(
            probability_of_user_preferences, predicted_probability_of_user_preferences
        )
        complementary_kl_divergence = self.kl_divergence(
            1 - probability_of_user_preferences,
            1 - predicted_probability_of_user_preferences,
        )
        return kl_divergence + complementary_kl_divergence

    def get_dense_and_unrated_splits(self, explicit):
        users_who_rated_something = explicit.indices()[0].unique()
        rated_items, counts = explicit.indices()[1].unique(return_counts=True)

        n_users, n_items = explicit.size()
        discriminant = n_items**2 - 4 * self.memory_upper_bound / n_users
        if discriminant > 0:
            n_rated_items_upper_bound = (n_items - discriminant**0.5) // 2
            if len(rated_items) > n_rated_items_upper_bound:
                n_clipped_items = len(rated_items) - n_rated_items_upper_bound
                wandb.log(
                    {"number of clipped items because of memory bound": n_clipped_items}
                )
                rated_items = rated_items[
                    counts.argsort(descending=True)[:n_rated_items_upper_bound]
                ]

        item_ids = torch.arange(n_items)
        items_mask = torch.full((n_items,), True)
        items_mask[rated_items] = False
        unrated_items = item_ids[items_mask]

        return users_who_rated_something, rated_items, unrated_items

    def split_explicit_feedback_and_ratings_into_dense_and_unrated(
        self, explicit, model_ratings
    ):
        (
            users_who_rated_something,
            rated_items,
            unrated_items,
        ) = self.get_dense_and_unrated_splits(explicit)

        explicit_dense_cutout = explicit.to_dense()[users_who_rated_something][
            :, rated_items
        ]
        model_ratings_dense_cutout = model_ratings[users_who_rated_something][
            :, rated_items
        ]
        unrated_items_model_ratings = model_ratings[users_who_rated_something][
            :, unrated_items
        ]
        return (
            explicit_dense_cutout,
            model_ratings_dense_cutout,
            unrated_items_model_ratings,
        )

    def __call__(self, *, explicit, model_ratings, implicit=None):
        explicit = explicit.to_sparse_coo().coalesce()
        if explicit.values().numel() == 0:
            return 0

        (
            explicit_dense_cutout,
            model_ratings_dense_cutout,
            unrated_items_model_ratings,
        ) = self.split_explicit_feedback_and_ratings_into_dense_and_unrated(
            explicit, model_ratings
        )

        dense_cutout_kl_divergence = self.dense_cutout_kl_divergence(
            explicit_dense_cutout, model_ratings_dense_cutout
        )

        rated_unrated_kl_divergence = self.rated_unrated_kl_divergence(
            explicit_dense_cutout,
            model_ratings_dense_cutout,
            unrated_items_model_ratings,
        )

        unrated_items_in_dense_cutout = explicit_dense_cutout != 0
        dense_cutout_mask = (
            unrated_items_in_dense_cutout[:, :, None]
            | unrated_items_in_dense_cutout[:, None, :]
        )
        dense_cutout_mask[
            :, (arange := torch.arange(dense_cutout_mask.shape[-1])), arange
        ] = False
        assert dense_cutout_mask.shape == dense_cutout_kl_divergence.shape
        loss = (dense_cutout_mask * dense_cutout_kl_divergence).sum()
        n_items_contributing_to_loss = dense_cutout_mask.sum()

        if n_unrated_items := unrated_items_model_ratings.shape[1]:
            rated_unrated_mask = einops.repeat(
                unrated_items_in_dense_cutout,
                f"user item -> user item {n_unrated_items}",
            )
            assert rated_unrated_mask.shape == rated_unrated_kl_divergence.shape
            loss += (rated_unrated_mask * rated_unrated_kl_divergence).sum()
            n_items_contributing_to_loss += rated_unrated_mask.sum()

        loss /= n_items_contributing_to_loss
        return loss
