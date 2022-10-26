import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt

from machine_learning.recommending.tests.conftest import (
    random_explicit_feedback,
    get_available_devices,
)
from machine_learning.recommending.tests.test_models import train_mock_slim


def test_slim_explain_recommendation():
    lightning_module = train_mock_slim()
    slim = lightning_module.model
    feature_names = [f"feature {i}" for i in range(slim.n_items)]
    user_id = torch.randint(slim.n_users, size=(1,))
    with wandb.init(project="Testing"):
        for device in get_available_devices():
            slim = slim.to(device)
            figures = slim.explain_recommendations(
                user_id=user_id[0],
                n_recommendations=np.random.randint(1, 4),
                feature_names=np.random.choice([None, feature_names]),
                log=True,
            )
            for figure in figures:
                assert isinstance(figure, plt.Figure)

            user_explicit = random_explicit_feedback(n_users=1, n_items=slim.n_items)

            figures = slim.explain_recommendations(
                user_explicit=user_explicit,
                n_recommendations=np.random.randint(1, 4),
                feature_names=np.random.choice([None, feature_names]),
                log=True,
            )
            for figure in figures:
                assert isinstance(figure, plt.Figure)
