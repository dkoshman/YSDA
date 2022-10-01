import einops
import torch
import wandb

from my_tools.utils import sparse_dense_multiply


class MSELoss:
    def __call__(self, explicit, model_ratings, implicit=None):
        loss = ((explicit.to_dense() - model_ratings) ** 2).mean()
        return loss


class ImplicitAwareSparseLoss:
    def __call__(self, explicit, implicit, model_ratings):
        """Loss = 1/|Explicit| * \\sum_{ij}Implicit_{ij} * (Explicit_{ij} - ModelRating_{ij})^2"""
        error = model_ratings - explicit
        error = sparse_dense_multiply(sparse_dense_multiply(implicit, error), error)
        loss = torch.sparse.sum(error) / error.values().numel()
        return loss


class PersonalizedRankingLoss:
    def __init__(
        self, max_rating, confidence_in_rating_quality=0.9, memory_upper_bound=1e8
    ):
        """
        Ranking based loss inspired by Bayesian Personalized Ranking paper.

        :param max_rating: maximum value on the ratings scale
        :param confidence_in_rating_quality: probability that item with best rating
        is more relevant than an unrated item
        :param memory_upper_bound: O(n) bound on size of tensor that can fit in the memory
        """
        self.scaling_coefficient = confidence_in_rating_quality * (
            1 + torch.e**-max_rating
        )
        self.memory_upper_bound = memory_upper_bound

    @staticmethod
    def pairwise_difference(left, right):
        """
        Given two rating matrices A, B of shape (n_users, n_items),
        returns matrix C of shape (n_users, n_items, n_items) such that
        C[u, i, j] = A[u, i] - B[u, j]
        """
        return left[:, :, None] - right[:, None, :]

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
            self.pairwise_difference(
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


class PersonalizedRankingLossMemoryEfficient(PersonalizedRankingLoss):
    """
    Loss from Bayesian Personalized Ranking paper.
    Caution: this implementation is pretty slow,
    but straightforward and uses less memory.
    """

    def __call__(self, *, explicit, model_ratings, implicit=None):
        implicit = implicit.coalesce()
        ids_of_users_who_rated_something = implicit.indices()[0].unique()
        if ids_of_users_who_rated_something.numel() == 0:
            return None

        n_users, n_items = implicit.size()
        item_ids = torch.arange(n_items)
        items_mask = torch.full((n_items,), True)
        criterion = 0
        for i, user_id in enumerate(ids_of_users_who_rated_something):
            implicit_user_feedback = implicit[user_id]
            liked_item_ids = implicit_user_feedback.indices().squeeze()
            items_mask[:] = True
            items_mask[liked_item_ids] = False
            uninteracted_item_ids = item_ids[items_mask]

            predicted_user_ratings = model_ratings[user_id]

            user_ranking_correctness_criterion = -torch.log1p(
                torch.exp(
                    -(
                        predicted_user_ratings[liked_item_ids, None]
                        - predicted_user_ratings[uninteracted_item_ids]
                    )
                )
            ).mean()
            criterion += user_ranking_correctness_criterion

        loss = -criterion / len(ids_of_users_who_rated_something)
        return loss
