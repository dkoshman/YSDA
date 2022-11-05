import einops
import numpy as np
import pandas as pd
import torch

from numba import njit
from scipy.sparse import csr_matrix


def relevant_pairs_to_frame(relevant_pairs):
    relevant_pairs = pd.DataFrame(relevant_pairs)
    relevant_pairs.columns = ["user_id", "item_id"]
    return relevant_pairs


@njit
def mark_duplicate_recommended_items_as_invalid(recommendations, invalid_mark=-1):
    """Mark all but first duplicated items as invalid for each user.
    :param recommendations: array of shape [n_users, n_items] with recommended item_ids
    :param invalid_mark: integer to replace duplicates with
    :return: recommendations array with duplicated values in each row marked as invalid,
    keeping the first occurrence.
    """
    for user, recommended_items in enumerate(recommendations):
        occurred_item_ids = []
        for item_rank, item in enumerate(recommended_items):
            if item in occurred_item_ids:
                recommendations[user, item_rank] = invalid_mark
            else:
                occurred_item_ids.append(item)
    return recommendations


def binary_relevance(explicit_feedback, recommendee_user_ids, recommendations):
    """
    Determine whether recommended items are relevant to users based on provided relevant pairs.
    :param explicit_feedback: csr_matrix with explicit feedback
    :param recommendee_user_ids: numpy array of shape [n_users] indicating users for whom the
    recommendations were generated.
    :param recommendations: numpy array of shape [n_users, n_items] with recommended item ids,
    with each row corresponding to user_id from recommendee_user_ids, with items sorted in
    decreasing predicted relevance.
    :return: numpy bool array of shape [n_users, n_items] indicating whether the recommended
    [user_id, item_id] pair is in relevant_pairs.
    """
    explicit = explicit_feedback[recommendee_user_ids]
    relevance = njit_binary_relevance(
        explicit.indptr, explicit.indices, recommendations
    )
    return relevance


@njit
def njit_binary_relevance(indptr, indices, recommendations):
    relevance = np.zeros_like(recommendations)
    for i, (begin, end, item_ids) in enumerate(
        zip(indptr, indptr[1:], recommendations)
    ):
        for j, item_id in enumerate(item_ids):
            relevance[i, j] = item_id in indices[begin:end]
    return relevance


def hitrate(relevance):
    """
    Returns proportion of users who were hit (were recommended at least one relevant item).
    This metric is pretty basic and represents an upper bound on other metrics
    and the model performance in general.

    :param relevance: boolean array of shape [n_users, n_items], representing
    relevant recommendations per user.
    """
    hits_per_user = einops.reduce(relevance, "user relevance -> user", np.any)
    return np.mean(hits_per_user)


def accuracy(relevance):
    """Average proportion of relevant recommendations."""
    return np.mean(relevance)


def mean_reciprocal_rank(relevance):
    """
    Returns average reciprocal rank, i.e. 1 / position, of first relevant item per user.
    Represents how far users have to explore their recommendations before encountering
    something relevant.
    """
    reciprocal_ranks = 1 / np.arange(1, relevance.shape[1] + 1)
    relevant_ranks = np.einsum("i j, j -> i j", relevance, reciprocal_ranks)
    first_relevant_rank = einops.reduce(relevant_ranks, "user rank -> user", np.max)
    return np.mean(first_relevant_rank)


def number_of_all_relevant_items_per_user(relevant_pairs, recommendee_user_ids):
    return (
        relevant_pairs_to_frame(relevant_pairs)
        .groupby("user_id")
        .size()
        .loc[recommendee_user_ids]
        .fillna(0)
        .astype(np.int32)
        .to_numpy()
    )


def recall(relevance, n_relevant_items_per_user):
    """
    Proportion of retrieved relevant items among all
    known to be relevant for each user.
    Very sensitive to size of recommendations.
    """
    return np.mean(
        einops.reduce(relevance, "user item -> user", "sum")
        / np.where(n_relevant_items_per_user == 0, 1, n_relevant_items_per_user)
    )


def mean_average_precision(relevance, n_relevant_items_per_user):
    """
    This metric represents average proportion of relevant items at every possible cutoff.
    While metrics generally try to keep their image equal to the [0, 1] interval, value of 1 is
    unreachable if user has no relevant items. This metric suffers from this downside the
    most, so it needs to know maximum possible number of relevant items per user to
    normalize its image to the [0, 1] interval.
    """

    reciprocal_ranks = 1 / np.arange(1, relevance.shape[1] + 1)
    n_relevant_items_at_k = np.cumsum(relevance, axis=1)
    precision_at_k = np.einsum("i j, j -> i j", n_relevant_items_at_k, reciprocal_ranks)
    average_precision = np.einsum(
        "i j, i j, i -> i",
        precision_at_k,
        relevance,
        1 / np.maximum(n_relevant_items_per_user, 1e-8),
    )
    return np.mean(average_precision)


def coverage(recommendations, all_possible_item_ids):
    """
    Coverage represents the proportion of items that ever get recommended.
    This metric is very sensitive to the structure of recommendations, so it is
    more of a debug kind of metric and useful only when comparing models
    with same queried users and same number of recommendations.

    :param recommendations: array of shape [n_users, n_items] with recommended item ids
    :param all_possible_item_ids: array or set with all item_ids from training dataset
    """
    all_items = set(all_possible_item_ids)
    return len(set(recommendations.flat) & all_items) / len(all_items)


def normalized_items_self_information(explicit_feedback):
    """
    Computes self information of each item, representing how much information
    an item occurrence brings. The higher the information, the more niche the
    item is.
    :return: pandas Series with item ids in index and self information in values
    """
    explicit_feedback = explicit_feedback.tocoo()
    relevant_pairs = np.stack([explicit_feedback.row, explicit_feedback.col]).T
    relevant_pairs = relevant_pairs_to_frame(relevant_pairs)
    n_item_interactions = relevant_pairs.groupby("item_id").size()
    information = np.log2(len(relevant_pairs)) - np.log2(n_item_interactions)
    assert (
        information.notna().all()
    ), f"Debug: Self information shouldn't be na, {information}"
    return information / np.log2(len(relevant_pairs))


@njit
def jit_surpisal(item_ids, self_information_per_item, recommendations):
    self_information = {i: j for i, j in zip(item_ids, self_information_per_item)}
    recommendations_information = np.zeros_like(recommendations, dtype=np.float32)
    for i, row in enumerate(recommendations):
        for j, item_id in enumerate(row):
            if item_id in self_information:
                recommendations_information[i, j] = self_information[item_id]
    return np.mean(recommendations_information)


def surprisal(recommendations, explicit_feedback=None, items_information=None):
    """
    Computes average normalized self information for each recommended item.
    The higher the surprisal, the more specific recommendations the model gives.

    :param recommendations: recommended item ids per user
    :param explicit_feedback: explicit feedback csr_matrix
    :param items_information: precomputed items self information
    """
    if items_information is None:
        if explicit_feedback is None:
            raise ValueError(
                "Either explicit_feedback or items_information must be passed."
            )
        items_information = normalized_items_self_information(explicit_feedback)
    return jit_surpisal(
        item_ids=items_information.index.to_numpy(),
        self_information_per_item=items_information.values,
        recommendations=recommendations,
    )


def discounted_cumulative_gain(relevance):
    """This DCG implementation works with float relevance as well"""
    relevance = 2**relevance - 1
    discount = 1 / np.log2(np.arange(2, relevance.shape[1] + 2))
    return np.einsum("i j, j -> i", relevance, discount)


def normalized_discounted_cumulative_gain(relevance, n_relevant_items_per_user):
    """General front-heavy measure of ranking effectiveness."""
    dcg = discounted_cumulative_gain(relevance)
    ideal_relevance = np.arange(relevance.shape[1]) < n_relevant_items_per_user[:, None]
    ideal_dcg = discounted_cumulative_gain(ideal_relevance)
    return np.mean(dcg / (ideal_dcg + 1e-8))


class RecommendingMetrics:
    def __init__(self, explicit_feedback: csr_matrix):
        self.explicit_feedback = explicit_feedback
        self.items_information = normalized_items_self_information(explicit_feedback)
        self.all_items = self.items_information.index.to_numpy()
        self.unique_recommended_items = np.empty(0, dtype=np.int32)
        self.n_relevant_items_per_user = (explicit_feedback > 0).sum(axis=1).A1
        self.k = None

    @staticmethod
    def atk_suffix(dictionary, k):
        return {key + "@" + str(k): v for key, v in dictionary.items()}

    def batch_metrics(
        self, user_ids: torch.IntTensor, recommendations: torch.IntTensor
    ):
        user_ids = user_ids.to("cpu", torch.int32).numpy()
        recommendations = recommendations.to("cpu", torch.int32).numpy()
        relevance = binary_relevance(self.explicit_feedback, user_ids, recommendations)
        n_relevant_items_per_user = self.n_relevant_items_per_user[user_ids]
        self.update_coverage(recommendations)
        metrics = dict(
            hitrate=hitrate(
                relevance=relevance,
            ),
            accuracy=accuracy(
                relevance=relevance,
            ),
            recall=recall(
                relevance=relevance,
                n_relevant_items_per_user=n_relevant_items_per_user,
            ),
            ndcg=normalized_discounted_cumulative_gain(
                relevance=relevance,
                n_relevant_items_per_user=n_relevant_items_per_user,
            ),
            map=mean_average_precision(
                relevance=relevance,
                n_relevant_items_per_user=n_relevant_items_per_user,
            ),
            mrr=mean_reciprocal_rank(
                relevance=relevance,
            ),
            surprisal=surprisal(
                recommendations=recommendations,
                items_information=self.items_information,
            ),
        )
        self.k = recommendations.shape[1]
        return self.atk_suffix(metrics, k=self.k)

    def update_coverage(self, recommendations):
        self.unique_recommended_items = np.union1d(
            self.unique_recommended_items, recommendations
        )

    def finalize_coverage(self):
        coverage = len(
            np.intersect1d(self.unique_recommended_items, self.all_items)
        ) / len(self.all_items)
        coverage_atk = self.atk_suffix(dict(coverage=coverage), k=self.k)
        self.unique_recommended_items = np.empty(0, dtype=np.int32)
        self.k = None
        return coverage_atk

    def metrics(self, user_ids: torch.IntTensor, recommendations: torch.IntTensor):
        metrics = self.batch_metrics(user_ids, recommendations)
        self.update_coverage(recommendations)
        coverage = self.finalize_coverage()
        metrics.update(coverage)
        return metrics
