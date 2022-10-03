import string
from typing import Literal

import einops
import numpy as np
import pandas as pd
import pytorch_lightning
import scipy
import torch
import wandb

from numba import njit
from scipy.sparse import csr_matrix

from data import MovieLens
from utils import scipy_coo_to_torch_sparse


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
    relevance = np.zeros_like(recommendations)
    for i, (user, items) in enumerate(zip(recommendee_user_ids, recommendations)):
        for j, item in enumerate(items):
            relevance[i, j] = (user, item) in relevant_pairs
    return relevance


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
        einops.reduce(relevance, "user item -> user", "sum") / n_relevant_items_per_user
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
    recommendations_information = np.zeros_like(recommendations, dtype=np.float32)
    for i, row in enumerate(recommendations):
        for j, item_id in enumerate(row):
            if item_id in self_information:
                recommendations_information[i, j] = self_information[item_id]
    return np.mean(recommendations_information)


def surprisal(recommendations, relevant_pairs=None, items_information=None):
    """
    Computes average normalized self information for each recommended item.
    The higher the surprisal, the more specific recommendations the model gives.

    :param recommendations: recommended item ids per user
    :param relevant_pairs: if passed, it should contain ALL the relevant pairs
    containing recommended items
    :param items_information: precomputed items self information
    """
    if items_information is None:
        if relevant_pairs is None:
            raise ValueError(
                "Either relevant_pairs or items_information must be passed."
            )
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
    ideal_relevance = np.arange(relevance.shape[1]) < n_relevant_items_per_user[:, None]
    ideal_dcg = discounted_cumulative_gain(ideal_relevance)
    return np.mean(dcg / (ideal_dcg + 1e-8))


class RecommendingMetrics:
    def __init__(self, explicit_feedback: csr_matrix, k=10):
        self.explicit_feedback = explicit_feedback
        self.k = k

        explicit_feedback = explicit_feedback.tocoo()
        self.relevant_pairs = np.stack([explicit_feedback.row, explicit_feedback.col]).T
        self.items_information = normalized_items_self_information(self.relevant_pairs)
        self.all_items = self.items_information.index.to_numpy()
        self.unique_recommended_items = np.empty(0, dtype=np.int32)
        self.n_relevant_items_per_user = (explicit_feedback > 0).sum(axis=1).A1

    def atk_suffix(self, dictionary):
        if self.k is None:
            return dictionary
        else:
            return {key + "@" + str(self.k): v for key, v in dictionary.items()}

    def batch_metrics_from_ratings(self, user_ids, ratings):
        if isinstance(ratings, np.ndarray):
            ratings = torch.from_numpy(ratings)
        ratings = ratings.to_dense()
        if isinstance(user_ids, np.ndarray):
            user_ids = torch.from_numpy(user_ids)

        if self.k is None:
            recommendations = torch.argsort(ratings, descending=True)
        else:
            values, recommendations = torch.topk(input=ratings, k=self.k)

        metrics = self.batch_metrics(
            *[i.to("cpu", torch.int32).numpy() for i in [user_ids, recommendations]]
        )
        return self.atk_suffix(metrics)

    def batch_metrics(self, user_ids, recommendations):
        relevance = binary_relevance(self.relevant_pairs, user_ids, recommendations)
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
        return metrics

    def update_coverage(self, recommendations):
        self.unique_recommended_items = np.union1d(
            self.unique_recommended_items, recommendations
        )

    def finalize_coverage(self):
        coverage = len(
            np.intersect1d(self.unique_recommended_items, self.all_items)
        ) / len(self.all_items)
        self.unique_recommended_items = np.empty(0, dtype=np.int32)
        return self.atk_suffix(dict(coverage=coverage))


class RecommendingMetricsCallback(pytorch_lightning.callbacks.Callback):
    def __init__(self, directory, k=10, aggregate_test_metrics=False):
        self.aggregate_test_metrics = aggregate_test_metrics
        movielens = MovieLens(directory)
        explicit_feedback = movielens.explicit_feedback_scipy_csr("u.data")
        if isinstance(k, int):
            k = [k]
        self.metrics = [RecommendingMetrics(explicit_feedback, k1) for k1 in k]

    @staticmethod
    def test_prefix(metrics_dict):
        return {"test_" + k: v for k, v in metrics_dict.items()}

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_batch(
            user_ids=batch["user_ids"], ratings=pl_module(**batch), kind="test"
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self.log_batch(
            user_ids=batch["user_ids"], ratings=pl_module(**batch), kind="val"
        )

    def log_batch(self, user_ids, ratings, kind: Literal["val", "test"]):
        for atk_metric in self.metrics:
            metrics = atk_metric.batch_metrics_from_ratings(
                user_ids=user_ids, ratings=ratings
            )
            if kind == "test" and self.aggregate_test_metrics:
                self.define_test_metrics(metrics)
            wandb.log(metrics)

    def define_test_metrics(self, metrics):
        metrics = self.test_prefix(metrics)
        for metric_name in metrics:
            wandb.define_metric(metric_name, summary="mean")

    def on_test_epoch_end(self, trainer=None, pl_module=None):
        self.epoch_end(kind="test")

    def on_validation_epoch_end(self, trainer=None, pl_module=None):
        self.epoch_end(kind="val")

    def epoch_end(self, kind: Literal["val", "test"]):
        for atk_metric in self.metrics:
            if len(atk_metric.unique_recommended_items):
                metrics = atk_metric.finalize_coverage()
                if kind == "test" and self.aggregate_test_metrics:
                    self.define_test_metrics(metrics)
                wandb.log(metrics)


class RecommendingIMDBCallback(pytorch_lightning.callbacks.Callback):
    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        path_to_movielens_folder="local/ml-100k",
        n_recommendations=10,
    ):
        imdb = ImdbRatings(path_to_imdb_ratings_csv, path_to_movielens_folder)
        self.explicit_feedback = imdb.explicit_feedback_scipy()
        self.item_df = imdb.movielens["u.item"]
        self.n_recommendations = n_recommendations

    def on_test_epoch_end(self, trainer=None, pl_module=None):
        recommendations = pl_module.recommend(
            users_explicit_feedback=self.explicit_feedback,
            n_recommendations=self.n_recommendations,
        )
        self.log_recommendation(recommendations.cpu().numpy())

    def on_validation_epoch_end(self, trainer=None, pl_module=None):
        recommendations = pl_module.recommend(
            users_explicit_feedback=self.explicit_feedback,
            n_recommendations=self.n_recommendations,
        )
        self.log_recommendation(recommendations.cpu().numpy())

    def log_recommendation(self, recommendations):
        recommendations += 1
        for i, recs in enumerate(recommendations):
            items_description = self.item_df.loc[recs]
            imdb_urls = "https://www.imdb.com/find?q=" + items_description[
                "movie title"
            ].str.split("(").str[0].str.replace(r"\s+", "+", regex=True)
            items_description = pd.concat(
                [items_description["movie title"], imdb_urls], axis="columns"
            )
            wandb.log(
                {
                    f"recommended_items_user_{i}": wandb.Table(
                        dataframe=items_description.T
                    )
                }
            )


class ImdbRatings:
    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        path_to_movielens_folder="local/ml-100k",
    ):
        self.path_to_imdb_ratings_csv = path_to_imdb_ratings_csv
        self.movielens = MovieLens(path_to_movielens_folder)

    @property
    def imdb_ratings(self):
        return pd.read_csv(self.path_to_imdb_ratings_csv)

    @staticmethod
    def normalize_titles(titles):
        titles = (
            titles.str.lower()
            .str.normalize("NFC")
            .str.replace(f"[{string.punctuation}]", "", regex=True)
            .str.replace("the", "")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return titles

    def explicit_feedback_scipy(self):
        imdb_ratings = self.imdb_ratings
        imdb_ratings["Title"] = self.normalize_titles(imdb_ratings["Title"])
        imdb_ratings["Your Rating"] = (imdb_ratings["Your Rating"] + 1) // 2

        movielens_titles = self.movielens["u.item"]["movie title"]
        movielens_titles = self.normalize_titles(movielens_titles.str.split("(").str[0])

        liked_items_ids = []
        liked_items_ratings = []
        for title, rating in zip(imdb_ratings["Title"], imdb_ratings["Your Rating"]):
            for movie_id, ml_title in movielens_titles.items():
                if title == ml_title:
                    liked_items_ids.append(movie_id - 1)
                    liked_items_ratings.append(rating)

        data = liked_items_ratings
        row = np.zeros(len(liked_items_ids))
        col = np.array(liked_items_ids)
        shape = (1, self.movielens.shape[1])
        explicit_feedback = scipy.sparse.coo_matrix((data, (row, col)), shape=shape)
        return explicit_feedback

    def explicit_feedback_torch(self):
        return scipy_coo_to_torch_sparse(self.explicit_feedback_scipy())
