import numpy as np
import torch
import wandb
import yaml

from machine_learning.recommending.entrypoint import fit
from matplotlib import pyplot as plt

from machine_learning.recommending.tests.conftest import (
    random_explicit_feedback,
    get_available_devices,
)

slim_config = """
name: "SLIM test run"
datamodule:
  movielens:
    class_name: MovieLens100k
    directory: local/ml-100k
  train_explicit_file: u1.base
  val_explicit_file: u1.test
  test_explicit_file: u1.test
loss:
  class_name: MSELoss
optimizer:
  class_name: Adam
logger:
  class_name: WandbLogger
  save_dir: local
model:
  class_name: SLIM
  l2_coefficient: 1.0e-5
  l1_coefficient: 1.0e-6
lightning_module:
  class_name: MovieLensSLIMRecommender
  patience: 5
  min_delta: 1.0e-4
trainer:
  reload_dataloaders_every_n_epochs: 1
  limit_val_batches: 0
  log_every_n_steps: 1
  default_root_dir: local
  num_sanity_val_steps: 0
  max_epochs: -1
"""


def test_slim_explain_recommendation():
    config = yaml.safe_load(slim_config)
    lightning_module = fit(config)
    slim = lightning_module.model
    feature_names = [f"feature {i}" for i in range(slim.n_items)]
    user_id = torch.randint(slim.n_users, size=(1,))
    with wandb.init(project="Testing"):
        for device in get_available_devices():
            slim = slim.to(device)
            figures = slim.explain_recommendations_for_user(
                user_id=user_id[0],
                n_recommendations=np.random.randint(1, 4),
                feature_names=np.random.choice([None, feature_names]),
                log=True,
            )
            for figure in figures:
                assert isinstance(figure, plt.Figure)

            user_explicit = random_explicit_feedback(n_users=1, n_items=slim.n_items)

            figures = slim.explain_recommendations_for_user(
                user_explicit=user_explicit,
                n_recommendations=np.random.randint(1, 4),
                feature_names=np.random.choice([None, feature_names]),
                log=True,
            )
            for figure in figures:
                assert isinstance(figure, plt.Figure)
