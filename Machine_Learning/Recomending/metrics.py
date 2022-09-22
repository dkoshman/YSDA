import einops
import numpy as np
import pandas as pd

from numba import njit


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


@njit
def binary_relevance(relevant_pairs, recommendee_user_ids, recommendations):
    """
    Determine whether recommended items are relevant to users based on provided relevant pairs.
    :param relevant_pairs: numpy array of shape [n_pairs, 2] containing [user_id, item_id] pairs
    indicating that user meaningfully interacted with item; implicit feedback in format of pairs.
    :param recommendee_user_ids: numpy array of shape [n_users] indicating users for whom the
    recommendations were generated.
    :param recommendations: numpy array of shape [n_users, n_items] with recommended item ids,
    with each row corresponding to user_id from recommendee_user_ids, with items sorted in
    decreasing predicted relevance.
    :return: numpy bool array of shape [n_users, n_items] indicating whether the recommended
    [user_id, item_id] pair is in relevant_pairs.
    """
    relevant_pairs = set([(i, j) for (i, j) in relevant_pairs])
    binary_relevance = np.empty_like(recommendations, dtype=bool)
    for i, (user, items) in enumerate(zip(recommendee_user_ids, recommendations)):
        for j, item in enumerate(items):
            binary_relevance[i, j] = (user, item) in relevant_pairs
    return binary_relevance


@njit
def binary_relevance_optimized(relevant_pairs, recommendee_user_ids, recommendations):
    """A more optimized, but convoluted and bug prone version of binary relevance."""
    user_ids_sorted, item_ids = relevant_pairs[np.argsort(relevant_pairs[:, 0])].T
    splits = np.flatnonzero(user_ids_sorted[1:] != user_ids_sorted[:-1]) + 1
    starts = np.concatenate((np.array([0]), splits))

    start_indices = np.zeros(user_ids_sorted.max() + 1)
    end_indices = np.zeros_like(start_indices)
    unique_users = user_ids_sorted[starts]

    start_indices[unique_users] = starts
    end_indices[unique_users] = np.concatenate(
        (splits, np.array([len(user_ids_sorted)]))
    )

    relevance = np.empty_like(recommendations)
    for i, (user_id, items_for_user) in enumerate(
        zip(recommendee_user_ids, recommendations)
    ):
        relevant_items = item_ids[start_indices[user_id] : end_indices[user_id]]
        for j, item_id in enumerate(items_for_user):
            relevance[i, j] = item_id in relevant_items

    return relevance


def hitrate(relevance):
    """
    Returns proportion of users who were hit (were recommended at least one relevant item).
    This metric is pretty basic and represents an upper bound on other metrics
    and the model performance in general.

    :param relevance: boolean array of shape [n_users, n_items], representing
    relevant recommendations per user.
    """
    hits_per_user = einops.reduce(relevance, "user relevance -> relevance", np.any)
    return np.mean(hits_per_user)


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
        "i j, i j, i -> i", precision_at_k, relevance, 1 / n_relevant_items_per_user
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
    return len(set(recommendations.flat()) & all_items) / len(all_items)


def normalized_items_self_information(relevant_pairs):
    """
    Computes self information of each item, representing how much information
    an item occurrence brings. The higher the information, the more niche the
    item is.
    :param relevant_pairs: dataframe or array of user-item pairs from implicit feedback
    :return: pandas Series with item ids in index and self information in values
    """
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
    recommendations_information = np.empty_like(recommendations, dtype=np.float32)
    for i, row in enumerate(recommendations):
        for j, item_id in enumerate(row):
            recommendations_information[i, j] = self_information.get(item_id, 0)
    return np.mean(recommendations_information)


def surprisal(relevant_pairs, recommendations):
    """
    Computes average normalized self information for each recommended item.
    The higher the surprisal, the more specific recommendations the model gives.
    """
    items_information = normalized_items_self_information(relevant_pairs)
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
    ideal_relevance = np.arange(relevance.shape[0]) < n_relevant_items_per_user[:, None]
    ideal_dcg = discounted_cumulative_gain(ideal_relevance)
    return np.mean(dcg / (ideal_dcg + 1e-8))


class RecommendingMetrics:
    def __init__(self, relevant_pairs, k=10, invalid_item_mark=-1):
        """
        Class for recommendation metrics
        :param relevant_pairs: pandas.DataFrame with columns "user_id", "item_id"
        corresponding to relevant_pairs
        :param k: metrics will be calculated based on top k recommendations
        """
        self.relevant_pairs = relevant_pairs
        self.k = k
        self.invalid_item_mark = invalid_item_mark

    def calculate_metrics(self, recommendations):
        """
        Return all metrics for the given recommendations
        :param recommendations: pandas.DataFrame with user ids in index and
        recommended item ids in rows in order corresponding to relevance; null
        item ids should be encoded as -1
        :return: pandas.Series with metric names in index and metric values
        """
        recommendations = recommendations.iloc[:, : self.k].astype(np.int32)

        user_recomendees = recommendations.index.to_numpy()

        recommendations = self.mark_duplicate_recommended_items_as_invalid_except_first(
            recommendations.values, self.invalid_item_mark
        )

        relevance = self.binary_relevance(
            self.relevant_pairs.values,
            user_recomendees,
            recommendations,
        )

        results = pd.Series(
            {
                "mnap": self.normalized_average_precision(
                    user_recomendees, relevance
                ).mean(),
                "hitrate": self.hitrate(relevance).mean(),
                "mrr": self.reciprocal_rank(relevance).mean(),
                "coverage": self.coverage(recommendations),
                "surprisal": self.surprisal(recommendations).mean(),
            }
        )
        return results
