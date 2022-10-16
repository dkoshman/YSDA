import time

import numpy as np
import scipy
import torch
import wandb
from utils import scipy_to_torch_sparse

from .utils import random_explicit_feedback, get_config_base
from .. import main, losses
from ..callbacks import RecommendingDataOverviewCallback
from ..movielens.lit import MovieLensNonGradientRecommender
from ..utils import WandbAPI


def test_losses():
    size = (1000, 100)
    for i in range(100):
        mse_loss_0_confidence = losses.MSEConfidenceLoss(confidence=0)
        mse_loss_non_0_confidence = losses.MSEConfidenceLoss(confidence=1 + i)
        explicit = scipy_to_torch_sparse(
            random_explicit_feedback(size=size, max_rating=10)
        )
        ratings = torch.randn(*size)

        mse_loss = ((explicit.to_dense() - ratings) ** 2).mean()
        assert torch.isclose(mse_loss, mse_loss_0_confidence(explicit, ratings))
        assert mse_loss < mse_loss_non_0_confidence(explicit, ratings)

        # pr_loss = losses.PersonalizedRankingLoss(confidence=i)
        # assert not torch.isnan(pr_loss(explicit, ratings))


def test_RecommendingDataOverviewCallback():
    explicit_feedback = scipy.sparse.csr_matrix(
        np.random.choice(
            np.arange(6),
            size=(1300, 800),
            replace=True,
            p=[0.90, 0, 0.01, 0.02, 0.03, 0.04],
        )
    )
    callback = RecommendingDataOverviewCallback()
    callback.explicit = explicit_feedback
    with wandb.init(dir="local", mode="offline"):
        callback.log_data_overview()


def test_wandb_artifact_checkpointing():
    config = get_config_base()
    config["logger"] = dict(name="WandbLogger", save_dir="local")
    config["trainer"] = dict(fast_dev_run=False)
    artifact_name = "PopularRecommenderTesting"

    with wandb.init():
        main.main(
            **config,
            model=dict(name="PopularRecommender"),
            lightning_module=dict(name="MovieLensNonGradientRecommender"),
            callbacks=dict(WandbCheckpointCallback=dict(artifact_name=artifact_name)),
        )
        wandb_api = WandbAPI()
        artifact = wandb_api.artifact(artifact_name=artifact_name)

    time.sleep(1)  # Give time to upload.
    model = wandb_api.build_from_checkpoint_artifact(
        artifact, class_candidates=[MovieLensNonGradientRecommender]
    )
    model.recommend(user_ids=[0, 1])
