import functools

import einops
import numpy as np
import pandas as pd

from numba import njit


class Metrics:
    def __init__(self, relevant_pairs, k=20):
        """
        Class for recommendation metrics
        :param relevant_pairs: pandas.DataFrame with columns "user_id", "org_id"
        corresponding to relevant_pairs
        :param k: metrics will be calculated based on top k recommendations
        """
        self.relevant_pairs = relevant_pairs
        self.k = k
        self._invalid_org_mark = -1

    @staticmethod
    @njit
    def mark_duplicates_as_invalid(recommended_orgs, invalid_mark):
        result = recommended_orgs.copy()
        for i, orgs in enumerate(recommended_orgs):
            occurences = []
            for j, org in enumerate(orgs):
                if org in occurences:
                    result[i, j] = invalid_mark
                else:
                    occurences.append(org)

        return result

    @staticmethod
    @njit
    def binary_relevance(relevant_pairs, recommendee_user_ids, recommended_org_ids):
        user_ids, org_ids = relevant_pairs[np.argsort(relevant_pairs[:, 0])].T
        splits = np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1
        starts = np.concatenate((np.array([0]), splits))

        start_indices = np.zeros(user_ids.max() + 1)
        end_indices = np.zeros_like(start_indices)
        unique_users = user_ids[starts]

        start_indices[unique_users] = starts
        end_indices[unique_users] = np.concatenate((splits, np.array([len(user_ids)])))

        relevance = np.empty_like(recommended_org_ids)
        for i, (user_id, orgs_for_user) in enumerate(
            zip(recommendee_user_ids, recommended_org_ids)
        ):
            relevant_orgs = org_ids[start_indices[user_id] : end_indices[user_id]]
            for j, org_id in enumerate(orgs_for_user):
                relevance[i, j] = org_id in relevant_orgs

        return relevance

    @property
    @functools.lru_cache()
    def _org_ids(self):
        return set(self.relevant_pairs["org_id"])

    @property
    @functools.lru_cache()
    def _n_reviews_per_user(self):
        return self.relevant_pairs.groupby("user_id").size()

    @staticmethod
    def precision(relevance):
        return np.cumsum(relevance, axis=1) / np.arange(1, relevance.shape[1] + 1)

    @property
    @functools.lru_cache()
    def _normalized_self_information(self):
        n_org_interactions = self.relevant_pairs.groupby("org_id").size()
        self_information = 1 - np.log2(n_org_interactions) / np.log2(
            len(self.relevant_pairs)
        )
        return self_information.fillna(0)

    def ideal_average_precision(self, user_recomendees):
        n_relevant_orgs = self._n_reviews_per_user.reindex(user_recomendees).fillna(1)
        positions = np.arange(1, self.k + 1)
        ideal_relevant_count_for_position = np.minimum(
            positions, einops.rearrange(n_relevant_orgs.values, "x -> x ()")
        )
        ideal_ap = (ideal_relevant_count_for_position / positions).mean(axis=1)
        return ideal_ap

    def normalized_average_precision(self, user_recomendees, relevance):
        average_precision = self.precision(relevance).mean(axis=1)
        ideal_ap = self.ideal_average_precision(user_recomendees)
        return average_precision / ideal_ap

    @staticmethod
    def hitrate(relevance):
        return np.any(relevance, axis=1)

    @staticmethod
    def reciprocal_rank(relevance):
        return (relevance / np.arange(1, relevance.shape[1] + 1)).max(axis=1)

    def coverage(self, recommended_orgs):
        unique_recommended_orgs = set(recommended_orgs.flatten())
        unique_recommended_orgs.discard(self._invalid_org_mark)
        org_ids = self._org_ids
        return len(unique_recommended_orgs & org_ids) / len(org_ids)

    @staticmethod
    @njit
    def jit_surpisal(org_ids, self_information_per_org, recomendee_org_ids):
        self_information_lookup = np.zeros(org_ids.max() + 1)
        self_information_lookup[org_ids] = self_information_per_org

        self_information = self_information_lookup[recomendee_org_ids.flatten()]
        self_information = self_information.reshape(recomendee_org_ids.shape)

        surprisal = np.empty(recomendee_org_ids.shape[0])
        for i, row in enumerate(self_information):
            surprisal[i] = row.mean()

        return surprisal

    def surprisal(self, recommended_orgs):
        return self.jit_surpisal(
            self._normalized_self_information.index.to_numpy(),
            self._normalized_self_information.values,
            recommended_orgs,
        )

    def calculate_metrics(self, recommendations):
        """
        Return all metrics for the given recommendations
        :param recommendations: pandas.DataFrame with user ids in index and
        recommended org ids in rows in order corresponding to relevance; null
        org ids should be encoded as -1
        :return: pandas.Series with metric names in index and metric values
        """
        recommendations = recommendations.iloc[:, : self.k].astype(np.int32)

        user_recomendees = recommendations.index.to_numpy()
        recommended_orgs = recommendations.values

        recommended_orgs = self.mark_duplicates_as_invalid(
            recommended_orgs, self._invalid_org_mark
        )

        relevance = self.binary_relevance(
            self.relevant_pairs.values,
            user_recomendees,
            recommended_orgs,
        )

        results = pd.Series(
            {
                "mnap": self.normalized_average_precision(
                    user_recomendees, relevance
                ).mean(),
                "hitrate": self.hitrate(relevance).mean(),
                "mrr": self.reciprocal_rank(relevance).mean(),
                "coverage": self.coverage(recommended_orgs),
                "surprisal": self.surprisal(recommended_orgs).mean(),
            }
        )
        return results
