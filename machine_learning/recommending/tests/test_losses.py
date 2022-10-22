import torch
from utils import scipy_to_torch_sparse

from .. import losses
from .conftest import random_explicit_feedback, seed_everything

seed_everything()


def test_losses():
    for i in range(100):
        mse_loss_1_confidence = losses.MSEConfidenceLoss(confidence=1)
        mse_loss_non_1_confidence = losses.MSEConfidenceLoss(confidence=1 / (1 + i))
        explicit = scipy_to_torch_sparse(random_explicit_feedback())
        ratings = torch.randn(*explicit.shape)
        mse_loss = ((explicit.to_dense() - ratings) ** 2).mean()
        assert torch.isclose(mse_loss, mse_loss_1_confidence(explicit, ratings))
        assert mse_loss >= mse_loss_non_1_confidence(explicit, ratings)
