import numpy as np
import torch
from machine_learning.recommending.interface import RecommendingLossInterface

from my_tools.utils import to_torch_coo

from .. import losses
from .conftest import (
    random_explicit_feedback,
    seed_everything,
    MockRecommender,
    cartesian_products_of_dict_values,
)

seed_everything()


def _test_loss(explicit, loss: RecommendingLossInterface):
    for i in range(100):
        model = MockRecommender(
            n_users=explicit.shape[0], n_items=explicit.shape[1], explicit=explicit
        )
        scalar_loss = loss(
            model=model,
            explicit=explicit,
            user_ids=torch.arange(explicit.shape[0]),
            item_ids=torch.arange(explicit.shape[1]),
        )
        assert torch.is_tensor(scalar_loss)
        assert scalar_loss.numel() == 1


def test_mse_loss():
    explicit = to_torch_coo(random_explicit_feedback())
    _test_loss(explicit=explicit, loss=losses.MSELoss(explicit=explicit))


def test_mse_l1_loss():
    explicit = to_torch_coo(random_explicit_feedback())
    grid_kwargs = dict(
        ratings_deviation=np.logspace(-3, 3, num=10),
        mean_unobserved_rating=np.linspace(0, 10, num=10),
    )
    for kwargs in cartesian_products_of_dict_values(grid_kwargs):
        _test_loss(
            explicit=explicit,
            loss=losses.MSEL1Loss(
                explicit=explicit,
                ratings_deviation=kwargs["ratings_deviation"],
                mean_unobserved_rating=kwargs["mean_unobserved_rating"],
            ),
        )
