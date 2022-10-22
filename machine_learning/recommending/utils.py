import os
from typing import TypeVar

import torch
import wandb
import yaml

from torch.utils.data import random_split


SparseTensor = TypeVar("SparseTensor", bound=torch.Tensor)
Pickleable = TypeVar("Pickleable")


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


def save_checkpoint_artifact(
    artifact_name,
    checkpoint_path,
    pl_module_class=None,
    metadata=None,
    description=None,
):
    if wandb.run is None:
        raise ValueError("Wandb run not initialized, can't save artifact.")
    metadata = metadata or {}
    metadata.update(pl_module_class=pl_module_class)
    artifact = wandb.Artifact(
        name=artifact_name,
        type="checkpoint",
        metadata=metadata,
        description=description,
    )
    artifact.add_file(local_path=checkpoint_path, name="checkpoint")
    wandb.run.log_artifact(artifact)


def fetch_artifact(
    *, entity=None, project=None, artifact_name, alias="latest", api_key=None
):
    wandb_api = wandb.Api(api_key=api_key)
    entity = entity or wandb.run.entity
    project = project or wandb.run.project
    artifact = wandb_api.artifact(f"{entity}/{project}/{artifact_name}:{alias}")
    return artifact


def load_checkpoint_artifact(artifact):
    artifact_dir = artifact.download()
    checkpoint_path = os.path.join(artifact_dir, artifact.get_path("checkpoint").path)
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
