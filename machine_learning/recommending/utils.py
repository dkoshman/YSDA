import contextlib
import os

import torch
import wandb
import yaml
from matplotlib import pyplot as plt

from torch.utils.data import random_split


def build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


def split_dataset(dataset, fraction):
    len_dataset = len(dataset)
    right_size = int(fraction * len_dataset)
    left_size = len_dataset - right_size
    return random_split(dataset, [left_size, right_size])


def fetch_artifact(
    *, entity=None, project=None, artifact_name, alias="latest", api_key=None
):
    wandb_api = wandb.Api(api_key=api_key)
    entity = entity or wandb.run.entity
    project = project or wandb.run.project
    artifact = wandb_api.artifact(f"{entity}/{project}/{artifact_name}:{alias}")
    return artifact


def load_path_from_artifact(artifact, path_inside_artifact="checkpoint"):
    artifact_dir = artifact.download()
    checkpoint_path = os.path.join(
        artifact_dir, artifact.get_path(path_inside_artifact).path
    )
    return checkpoint_path


def update_from_base_config(config, base_config_file):
    """Keeps everything from config, and updates it with entries from base config up to 1 depth in."""
    base_config = yaml.safe_load(open(base_config_file))
    for k, v in config.items():
        if k in base_config and isinstance(v, dict):
            base_config[k].update(v)
        else:
            base_config[k] = v
    return base_config


def wandb_context_manager(config):
    if wandb.run is None and config.get("logger") is not None:
        return wandb.init(project=config.get("project"), config=config)
    return contextlib.nullcontext()


@contextlib.contextmanager
def plt_figure(*args, **kwargs):
    figure = plt.figure(*args, **kwargs)
    try:
        yield figure
    finally:
        plt.close(figure)


@contextlib.contextmanager
def wandb_plt_figure(title, *args, **kwargs):
    with plt_figure(*args, **kwargs) as figure:
        plt.title(title)
        try:
            yield figure
        finally:
            wandb.log({title: wandb.Image(figure)})
