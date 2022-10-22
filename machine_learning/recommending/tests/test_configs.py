import os

import wandb
import yaml

from .conftest import seed_everything
from ..main import fit
from ..utils import update_from_base_config

seed_everything()


def _test_config(config_path):
    config = yaml.safe_load(open(config_path))
    config = update_from_base_config(config)
    config["project"] = "Testing"
    with wandb.init(project=config["project"]):
        fit(config)


def test_baseline_config():
    _test_config("configs/baseline.yaml")


def test_svd_config():
    _test_config("configs/svd.yaml")


def test_implicit_nn_config():
    _test_config("configs/implicit_nn.yaml")


def test_implicit_mf_config():
    _test_config("configs/implicit_mf.yaml")


def test_als_config():
    _test_config("configs/als.yaml")


def test_cat_config():
    _test_config("configs/cat.yaml")


def test_movielens_cat_config():
    _test_config("configs/movielens_cat.yaml")


def test_slim_config():
    _test_config("configs/slim.yaml")


def test_mf_config():
    _test_config("configs/mf.yaml")
