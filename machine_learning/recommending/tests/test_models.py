import torch

from .utils import random_explicit_feedback, get_config_base
from .. import main
from ..interface import FitExplicitInterfaceMixin
from ..models import als, baseline, mf, slim


recommender_modules = [
    als.ALS,
    als.ALSjit,
    als.ALSjitBiased,
    baseline.RandomRecommender,
    baseline.PopularRecommender,
    baseline.SVDRecommender,
    baseline.ImplicitNearestNeighborsRecommender,
    baseline.ImplicitMatrixFactorizationRecommender,
    mf.MatrixFactorizationBase,
    mf.ConstrainedProbabilityMatrixFactorization,
    slim.SLIM,
]


def test_slim():
    main.main(
        **get_config_base(),
        model=dict(name="SLIM"),
        lightning_module=dict(name="MovieLensSLIMRecommender"),
    )


def test_mf():
    for name in [
        "MatrixFactorizationBase",
        "ConstrainedProbabilityMatrixFactorization",
    ]:
        main.main(
            **get_config_base(),
            model=dict(name=name),
            lightning_module=dict(name="MovieLensPMFRecommender"),
        )


def test_als():
    for name in ["ALS", "ALSjit", "ALSjitBiased"]:
        main.main(
            **get_config_base(),
            model=dict(name=name),
            lightning_module=dict(name="MovieLensNonGradientRecommender"),
        )


def test_baseline():
    for name in [
        "RandomRecommender",
        "PopularRecommender",
        "SVDRecommender",
        # "NearestNeighbours",
    ]:
        main.main(
            **get_config_base(),
            model=dict(name=name),
            lightning_module=dict(name="MovieLensNonGradientRecommender"),
        )


def test_implicit_nearest_neighbors():
    for implicit_model in [
        "BM25Recommender",
        "CosineRecommender",
        "TFIDFRecommender",
    ]:
        main.main(
            **get_config_base(),
            model=dict(
                name="ImplicitNearestNeighborsRecommender",
                implicit_model=implicit_model,
            ),
            lightning_module=dict(name="MovieLensNonGradientRecommender"),
        )


def test_implicit_matrix_factorization():
    for implicit_model in [
        "AlternatingLeastSquares",
        "LogisticMatrixFactorization",
        "BayesianPersonalizedRanking",
    ]:
        main.main(
            **get_config_base(),
            model=dict(
                name="ImplicitMatrixFactorizationRecommender",
                implicit_model=implicit_model,
            ),
            lightning_module=dict(name="MovieLensNonGradientRecommender"),
        )


# def test_bpmf():
#     main.main(
#         **get_config_base(),
#         model=dict(name="BayesianPMF"),
#         lightning_module=dict(name="MovieLensNonGradientRecommender"),
#     )


def test_new_users_items():
    explicit = random_explicit_feedback(size=(1007, 113))
    for Module in recommender_modules:
        module = Module(n_users=explicit.shape[0], n_items=explicit.shape[1])
        if isinstance(module, FitExplicitInterfaceMixin):
            try:
                module.fit(explicit)
            except RuntimeError as e:
                if str(e).startswith("CURAND error"):
                    continue
                else:
                    raise e
        module.new_users(10)
        module.new_items(10)
        new_users = module.new_users(100)
        module.new_items(100)
        ratings = module(user_ids=new_users, item_ids=torch.arange(module.n_items))
        assert ratings.shape == (100, explicit.shape[1] + 110)
        assert not ratings.isnan().any()


def test_save_load():
    explicit = random_explicit_feedback(size=(1007, 113))
    for Module in recommender_modules:
        module = Module(n_users=explicit.shape[0], n_items=explicit.shape[1])
        if isinstance(module, FitExplicitInterfaceMixin):
            module.fit(explicit)
        pickleable = module.save()
        module = Module(n_users=explicit.shape[0], n_items=explicit.shape[1])
        module.load(pickleable)
        ratings = module(
            user_ids=torch.tensor([5, 3, 2, 9, 1, 7]), item_ids=torch.arange(10, 100)
        )
        assert not ratings.isnan().any()
