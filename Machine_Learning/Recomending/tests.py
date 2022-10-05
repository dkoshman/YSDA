import numpy as np
import scipy
import torch
import torchmetrics as tm
import wandb

import als
import entrypoints
import bpmf
import metrics
import pmf
import slim

from callbacks import RecommendingDataOverviewCallback


def get_config_base():
    config_base = dict(
        config_path="configs/config_for_testing.yaml",
        logger=dict(name="WandbLogger", offline=True, anonymous=True, save_dir="local"),
        datamodule=dict(
            name="MovieLensDataModule", directory="local/ml-100k", batch_size=100
        ),
    )
    return config_base


def test_mark_duplicate_recommended_items_as_invalid():
    marked = metrics.mark_duplicate_recommended_items_as_invalid(
        recommendations=np.array([[1, 2, 1, 2]]), invalid_mark=-1
    )
    assert (marked == np.array([[1, 2, -1, -1]])).all()

    marked = metrics.mark_duplicate_recommended_items_as_invalid(
        recommendations=np.array([[1, 2, 1], [1, 2, 2]]), invalid_mark=-1
    )
    assert (marked == np.array([[1, 2, -1], [1, 2, -1]])).all()


def test_binary_relevance():
    relevance = metrics.binary_relevance(
        relevant_pairs=np.array([[1, 2], [3, 4]]),
        recommendee_user_ids=np.array([1, 3]),
        recommendations=np.array([[0, 2, 1], [3, 2, 4]]),
    )
    assert (relevance == np.array([[0, 1, 0], [0, 0, 1]])).all()


def test_hitrate():
    hitrate = metrics.hitrate(
        relevance=np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0], [0, 1, 0], [1, 1, 1]])
    )
    assert hitrate == 3 / 5


def test_mean_reciprocal_rank():
    mrr = metrics.mean_reciprocal_rank(relevance=np.array([[0, 1], [1, 0]]))
    assert mrr == np.mean([1 / 1, 1 / 2])

    mrr = metrics.mean_reciprocal_rank(
        relevance=np.array([[0, 0, 0], [1, 0, 1], [0, 0, 1], [0, 1, 1]])
    )
    assert mrr == np.mean([0, 1 / 1, 1 / 3, 1 / 2])

    mrr = metrics.mean_reciprocal_rank(
        relevance=np.array([[0, 0, 0, 1, 1], [0, 0, 1, 0, 1]])
    )
    assert mrr == np.mean([1 / 4, 1 / 3])


def test_number_of_all_relevant_items_per_user():
    relevant_items_per_user = metrics.number_of_all_relevant_items_per_user(
        relevant_pairs=np.array([[3, 1], [0, 2], [0, 1]]),
        recommendee_user_ids=np.array([0, 3]),
    )
    assert (relevant_items_per_user == np.array([2, 1])).all()


def test_mean_average_precision():
    map = metrics.mean_average_precision(
        relevance=np.array([[0, 1], [1, 1], [0, 0], [1, 0]]),
        n_relevant_items_per_user=np.array([1, 2, 0, 1]),
    )
    assert np.isclose(map, np.mean([1 / 2, 1, 0, 1]))


def test_coverage():
    coverage = metrics.coverage(
        recommendations=np.array([[0, 3, -1], [2, 1, 0], [1, -1, -1]]),
        all_possible_item_ids=np.arange(6),
    )
    assert coverage == 4 / 6


def test_surprisal():
    surprisal = metrics.surprisal(
        recommendations=np.array([[0, 2, 3], [1, 2, 0]]),
        relevant_pairs=np.array([[0, 0], [2, 1], [1, 1]]),
    )

    assert np.isclose(
        surprisal,
        -np.mean(
            np.array([[np.log2(1 / 3), 0, 0], [np.log2(2 / 3), 0, np.log2(1 / 3)]])
            / np.log2(3)
        ),
    )


def test_normalized_discounted_cumulative_gain():
    ndcg = metrics.normalized_discounted_cumulative_gain(
        relevance=np.array([[0, 1], [1, 0], [0, 0]]),
        n_relevant_items_per_user=np.array([2, 3, 0]),
    )
    assert np.isclose(
        ndcg,
        np.mean([(1 / np.log2(3)) / (1 + 1 / np.log2(3)), 1 / (1 + 1 / np.log2(3)), 0]),
    )


def test_slim():
    config = get_config_base()
    config["datamodule"]["name"] = "MovielensSlimDatamodule"
    slim.main(**config, model=dict(name="SLIM"), lightning_module=dict(name="LitSLIM"))


def test_pmf():
    config = get_config_base()
    config["datamodule"]["name"] = "MovielensPMFDataModule"
    pmf.main(
        **config,
        model=dict(name="ProbabilityMatrixFactorization"),
        lightning_module=dict(name="LitProbabilityMatrixFactorization"),
    )


def test_als():
    for als_name in ["ALS", "ALSjit", "ALSjitBiased"]:
        als.main(
            **get_config_base(),
            model=dict(name=als_name),
            lightning_module=dict(name="ALSRecommender"),
        )


def test_baseline():
    for name in [
        "RandomRecommender",
        "PopularRecommender",
        "ImplicitNearestNeighbors",
        "SVDRecommender",
        "NearestNeighbours",
    ]:
        entrypoints.main(
            **get_config_base(),
            model=dict(name=name),
            lightning_module=dict(name="LitBaselineRecommender"),
        )


def test_implicit_matrix_factorization():
    for matrix_factorization_model in [
        "AlternatingLeastSquares",
        "LogisticMatrixFactorization",
        "BayesianPersonalizedRanking",
    ]:
        entrypoints.main(
            **get_config_base(),
            model=dict(
                name="ImplicitMatrixFactorization",
                matrix_factorization_model=matrix_factorization_model,
            ),
            lightning_module=dict(name="LitBaselineRecommender"),
        )


def test_RecommendingMetrics():
    k = 10
    for _ in range(10):
        explicit = np.random.choice(
            a=5, size=(1000, 100), p=[0.90, 0.01, 0.02, 0.03, 0.04]
        )
        explicit = scipy.sparse.coo_matrix(explicit).tocsr()
        recommending_metrics_atk = metrics.RecommendingMetrics(explicit, k=k)
        recommending_metrics = metrics.RecommendingMetrics(explicit, k=None)

        for _ in range(10):
            n_users = 100
            user_ids = torch.tensor(
                np.random.choice(explicit.shape[0], size=n_users, replace=False)
            )
            ratings = torch.tensor(np.random.randn(n_users, explicit.shape[1]))
            metrics_dict_atk = recommending_metrics_atk.batch_metrics_from_ratings(
                user_ids, ratings
            )
            metrics_dict_atk = {k.split("@")[0]: v for k, v in metrics_dict_atk.items()}
            metrics_dict = recommending_metrics.batch_metrics_from_ratings(
                user_ids, ratings
            )

            target = torch.tensor((explicit[user_ids] > 0).toarray())
            assert np.isclose(
                metrics_dict_atk["hitrate"],
                np.mean(
                    [
                        tm.functional.retrieval_hit_rate(preds=r, target=t, k=k)
                        for r, t in zip(ratings, target)
                    ]
                ),
            )
            assert np.isclose(
                metrics_dict["map"],
                np.mean(
                    [
                        tm.functional.retrieval_average_precision(preds=r, target=t)
                        for r, t in zip(ratings, target)
                    ]
                ),
            )
            assert np.isclose(
                metrics_dict["mrr"],
                np.mean(
                    [
                        tm.functional.retrieval_reciprocal_rank(preds=r, target=t)
                        for r, t in zip(ratings, target)
                    ]
                ),
            )
            assert np.isclose(
                metrics_dict_atk["ndcg"],
                np.mean(
                    [
                        tm.functional.retrieval_normalized_dcg(preds=r, target=t, k=k)
                        for r, t in zip(ratings, target)
                    ]
                ),
            )


def test_bpmf():
    bpmf.main(
        **get_config_base(),
        model=dict(name="BayesianPMF"),
        lightning_module=dict(name="BPMFRecommender"),
    )

def test_RecommendingDataOverviewCallback():
    explicit_feedback = scipy.sparse.csr_matrix(
    np.random.choice(
        np.arange(6),
        size=(1300, 800),
        replace=True,
        p=[0.90, 0, 0.01, 0.02, 0.03, 0.04],
    )
)
    callback = RecommendingDataOverviewCallback(explicit_feedback=explicit_feedback)
    with wandb.init(dir="local", mode="offline"):
        callback.log_data_overview()
