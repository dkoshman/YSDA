import os
from typing import Literal

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from recommending.data import SparseDataset
from scipy.sparse import coo_matrix

from my_tools.utils import to_torch_coo
from ..data import build_recommending_dataloader
from ..interface import ExplanationMixin
from ..lit import NonGradientRecommenderMixin

from ..models import als, baseline, cat, mf, slim
from ..models.slim import SLIMRecommender
from ..movielens import cat as movielens_cat, MovieLens100k
from .conftest import (
    get_available_devices,
    random_explicit_feedback,
    MockLitRecommender,
    MockLightningModuleInterface,
    seed_everything,
)
from ..movielens.cat import CatboostMovieLensAggregatorFromArtifacts

seed_everything()


def _test_recommender_module(module, test_reload=True):
    for device in get_available_devices():
        module = module.to(device)

        n_users = module.n_users
        n_items = module.n_items
        assert np.isscalar(n_users)
        assert np.isscalar(n_items)
        user_ids = torch.from_numpy(
            np.random.choice(
                range(n_users), size=np.random.randint(1, n_users), replace=False
            )
        )
        item_ids = torch.from_numpy(
            np.random.choice(
                range(n_items), size=np.random.randint(1, n_items), replace=False
            )
        )
        n_recommendations = np.random.randint(1, n_items)
        filter_already_liked_items = np.random.choice([True, False])

        ratings = module(user_ids=user_ids, item_ids=item_ids)
        assert torch.is_tensor(ratings)
        assert not ratings.is_sparse and not ratings.is_sparse_csr
        assert ratings.dtype == torch.float32
        recommendations = module.recommend(
            user_ids=user_ids,
            n_recommendations=n_recommendations,
            filter_already_liked_items=filter_already_liked_items,
        )
        assert torch.is_tensor(recommendations)
        assert recommendations.dtype == torch.int64
        explicit = random_explicit_feedback(n_items=module.n_items)
        explicit = to_torch_coo(explicit)
        online_ratings = module.online_ratings(users_explicit=explicit)
        assert torch.is_tensor(online_ratings)
        online_recommendations = module.online_recommend(
            users_explicit=explicit, n_recommendations=n_recommendations
        )
        assert torch.is_tensor(online_recommendations)
        assert online_recommendations.dtype == torch.int64
        assert isinstance(module.device, torch.device)
        assert isinstance(module.to_scipy_coo(module.explicit), coo_matrix)

        if test_reload:
            state_dict = module.state_dict()
            loaded_module = module.__class__()
            loaded_module.load_state_dict(state_dict)
            loaded_ratings = loaded_module(user_ids=user_ids, item_ids=item_ids)
            assert torch.isclose(ratings, loaded_ratings, atol=1e-5).all()
            loaded_recommendations = loaded_module.recommend(
                user_ids=user_ids,
                n_recommendations=n_recommendations,
                filter_already_liked_items=filter_already_liked_items,
            )
            assert (recommendations != loaded_recommendations).to(
                torch.float32
            ).mean() < recommendations.shape[0] + 0.01 * recommendations.numel()
            loaded_online_ratings = loaded_module.online_ratings(
                users_explicit=explicit
            )
            assert torch.isclose(online_ratings, loaded_online_ratings, atol=1e-5).all()
            loaded_online_recommendations = loaded_module.online_recommend(
                users_explicit=explicit, n_recommendations=n_recommendations
            )
            assert (online_recommendations != loaded_online_recommendations).to(
                torch.float32
            ).sum() < online_recommendations.shape[
                0
            ] + 0.01 * online_recommendations.numel()

        if isinstance(module, ExplanationMixin):
            user_id = np.random.randint(module.n_users)
            user_explicit = random_explicit_feedback(n_users=1, n_items=module.n_items)

            with wandb.init(project="Testing"):
                module.explain_recommendations(
                    user_id=user_id,
                    n_recommendations=np.random.randint(1, 3),
                    log=True,
                    logging_prefix="testing",
                )
                try:
                    module.explain_recommendations(
                        user_explicit=user_explicit,
                        n_recommendations=np.random.randint(1, 3),
                        log=True,
                        logging_prefix="testing online",
                    )
                except NotImplementedError:
                    pass


class MockNonGradientRecommender(
    NonGradientRecommenderMixin, MockLightningModuleInterface
):
    def dataloader(self, stage: Literal["train", "val", "test", "predict"]):
        return build_recommending_dataloader(
            dataset=SparseDataset(explicit=self.model.explicit)
        )

    def forward(self, *args, **kwargs):
        pass

    def __init__(self, model):
        super().__init__()
        self.model = model

    def step(self, batch, stage):
        pass


def test_catboost_aggregator():
    movielens = MovieLens100k()
    explicit = movielens.explicit_feedback_scipy_csr("u1.base")
    recommender_artifact_names = ["als"]
    module = CatboostMovieLensAggregatorFromArtifacts(
        entity="dkoshman",
        project="Recommending",
        recommender_artifact_names=recommender_artifact_names,
        explicit=explicit,
    )
    module.fit()
    _test_recommender_module(module)


def test_als():
    for cls in als.ALS, als.ALSjit, als.ALSjitBiased:
        explicit = random_explicit_feedback()
        module = cls(
            explicit=explicit,
            epochs=np.random.randint(1, 3),
            latent_dimension_size=np.random.randint(1, 5),
            regularization_lambda=np.random.uniform(0, 100),
            confidence_alpha=np.random.uniform(0, 1),
            lambda_decay=np.random.uniform(0, 1),
        )
        lightning_module = MockNonGradientRecommender(model=module)
        trainer = pl.Trainer()
        trainer.fit(lightning_module)
        _test_recommender_module(module)
        if cls != als.ALSjitBiased:
            user_id = np.random.randint(0, module.n_users)
            recommended_item_ids = module.recommend(
                user_ids=torch.tensor([user_id]),
                n_recommendations=np.random.randint(1, 10),
            )[0]
            explanations = module.explain_recommendations(
                user_id=user_id, recommendations=recommended_item_ids
            )
            assert isinstance(explanations["dataframe"], pd.DataFrame)
            assert isinstance(explanations["style"], pd.io.formats.style.Styler)


def test_baseline():
    for cls in [baseline.RandomRecommender, baseline.PopularRecommender]:
        explicit = random_explicit_feedback()
        module = cls(explicit=explicit)
        lightning_module = MockNonGradientRecommender(model=module)
        trainer = pl.Trainer()
        trainer.fit(lightning_module)
        _test_recommender_module(module, test_reload=cls != baseline.RandomRecommender)


def test_catboost():
    for cls in [
        cat.CatboostExplicitRecommender,
        movielens_cat.CatboostMovieLensFeatureRecommender,
    ]:
        explicit = random_explicit_feedback(max_rating=np.random.randint(2, 11))
        kwargs = dict(explicit=explicit, iterations=np.random.randint(1, 10))
        if cls == movielens_cat.CatboostMovieLensFeatureRecommender:
            kwargs["movielens_directory"] = "local/ml-100k"
        module = cls(**kwargs)
        lightning_module = MockNonGradientRecommender(model=module)
        trainer = pl.Trainer()
        trainer.fit(lightning_module)
        _test_recommender_module(module)
        feature_importance = module.feature_importance()
        assert isinstance(feature_importance, pd.DataFrame)


def test_implicit_nearest_neighbors():
    for implicit_model in [
        "BM25Recommender",
        "CosineRecommender",
        "TFIDFRecommender",
    ]:
        explicit = random_explicit_feedback()
        module = baseline.ImplicitNearestNeighborsRecommender(
            explicit=explicit,
            implicit_model=implicit_model,
            num_neighbors=np.random.choice(range(1, 100)),
            num_threads=np.random.choice(range(4)),
        )
        lightning_module = MockNonGradientRecommender(model=module)
        trainer = pl.Trainer()
        trainer.fit(lightning_module)
        _test_recommender_module(module)


def test_implicit_matrix_factorization():
    for implicit_model in [
        "LogisticMatrixFactorization",
        "BayesianPersonalizedRanking",
        "AlternatingLeastSquares",
    ]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        explicit = random_explicit_feedback()
        module = baseline.ImplicitMatrixFactorizationRecommender(
            explicit=explicit,
            implicit_model=implicit_model,
            iterations=np.random.randint(1, 10),
            factors=np.random.choice(range(1, 100)),
            learning_rate=10 ** np.random.uniform(-2, -1),
            regularization=10 ** np.random.uniform(-2, -1),
            num_threads=np.random.choice(range(4)),
            use_gpu=False,
        )
        lightning_module = MockNonGradientRecommender(model=module)
        trainer = pl.Trainer()
        trainer.fit(lightning_module)
        _test_recommender_module(module)


class MockMFRecommender(MockLitRecommender):
    def train_dataloader(self):
        return build_recommending_dataloader(
            dataset=SparseDataset(explicit=self.model.explicit),
            sampler_type="grid",
            batch_size=np.random.randint(10, 100),
            num_workers=np.random.randint(0, 4),
            shuffle=True,
        )


def test_mf():
    for cls in [mf.MatrixFactorization, mf.ConstrainedProbabilityMatrixFactorization]:
        explicit = random_explicit_feedback()
        module = cls(
            explicit=explicit,
            latent_dimension=np.random.choice(range(1, 10)),
            weight_decay=10 ** np.random.uniform(-4, -2),
        )
        lightning_module = MockMFRecommender(model=module)
        trainer = pl.Trainer(max_epochs=10)
        if torch.cuda.is_available():
            torch.cuda.init()
        trainer.fit(lightning_module)
        _test_recommender_module(module)


class MockSLIMRecommender(SLIMRecommender):
    def train_explicit(self):
        n_users = self.hparams["datamodule_config"]["n_users"]
        n_items = self.hparams["datamodule_config"]["n_items"]
        return random_explicit_feedback(n_users=n_users, n_items=n_items)

    def val_explicit(self):
        n_users = self.hparams["datamodule_config"]["n_users"]
        n_items = self.hparams["datamodule_config"]["n_items"]
        return random_explicit_feedback(n_users=n_users, n_items=n_items)

    @property
    def class_candidates(self):
        return super().class_candidates + [slim.SLIM]


def train_mock_slim():
    model_config = dict(
        class_name="SLIM",
        l1_coefficient=10 ** np.random.uniform(-5, -2),
        l2_coefficient=10 ** np.random.uniform(-5, -2),
    )
    datamodule_config = dict(
        n_users=np.random.randint(10, 20),
        n_items=np.random.randint(10, 20),
    )
    lightning_module = MockSLIMRecommender(
        patience=np.random.randint(0, 3),
        min_delta=np.random.uniform(0.01, 1),
        check_val_every_n_epoch=np.random.randint(1, 5),
        datamodule_config=datamodule_config,
        model_config=model_config,
    )
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=1,
        limit_val_batches=0,
        max_epochs=-1,
    )
    if torch.cuda.is_available():
        torch.cuda.init()
    trainer.fit(lightning_module)
    return lightning_module


def test_slim():
    lightning_module = train_mock_slim()
    _test_recommender_module(lightning_module.model)
