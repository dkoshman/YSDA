import os

import numpy as np
import torch
import wandb

from torch.utils.data import random_split
from scipy.sparse import coo_matrix

from my_tools.utils import get_class


def build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


def to_sparse_coo(tensor):
    try:
        return tensor.to_sparse_coo()
    except NotImplementedError:
        # Torch errors if coo tensor is cast to coo again.
        return tensor


def torch_sparse_to_scipy_coo(sparse_tensor):
    sparse_tensor = to_sparse_coo(sparse_tensor).coalesce().cpu()
    sparse_tensor = coo_matrix(
        (sparse_tensor.values().numpy(), sparse_tensor.indices().numpy()),
        shape=sparse_tensor.shape,
    )
    return sparse_tensor


def scipy_coo_to_torch_sparse(sparse_matrix, device=None):
    return torch.sparse_coo_tensor(
        indices=torch.from_numpy(np.stack([sparse_matrix.row, sparse_matrix.col])),
        values=sparse_matrix.data,
        size=sparse_matrix.shape,
        device=device,
    )


def torch_sparse_slice(sparse_matrix, row_ids=None, col_ids=None, device=None):
    if torch.is_tensor(sparse_matrix):
        sparse_matrix = torch_sparse_to_scipy_coo(sparse_matrix)

    if row_ids is None:
        row_ids = slice(None)
    elif torch.is_tensor(row_ids):
        row_ids = row_ids.cpu().numpy()

    if col_ids is None:
        col_ids = slice(None)
    elif torch.is_tensor(col_ids):
        col_ids = col_ids.cpu().numpy()

    sparse_matrix = sparse_matrix.tocsr()[row_ids][:, col_ids].tocoo()
    torch_sparse_coo_tensor = scipy_coo_to_torch_sparse(sparse_matrix, device)
    return torch_sparse_coo_tensor


def split_dataset(dataset, fraction):
    len_dataset = len(dataset)
    right_size = int(fraction * len_dataset)
    left_size = len_dataset - right_size
    return random_split(dataset, [left_size, right_size])


class WandbAPI:
    CHECKPOINT_TYPE = "checkpoint"
    CHECKPOINT_PATH = "checkpoint_path"
    PL_MODULE_CLASS = "pl_module_class"
    CHECKPOINT_IN_ARTIFACT_HIERARCHY = "checkpoint"

    def __init__(self, api_key=None, entity=None):
        self.api = wandb.Api(api_key=api_key)
        self.entity = entity or wandb.run.entity

    def save_artifact(
        self, name, artifact_type, metadata=None, description=None, local_paths=None
    ):
        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata,
            description=description,
        )
        if local_paths is not None:
            for name, path in local_paths.items():
                artifact.add_file(local_path=path, name=name)
        wandb.run.log_artifact(artifact)

    def save_checkpoint(
        self,
        checkpoint_path,
        artifact_name,
        pl_module_class,
        description=None,
        metadata=None,
        add_run_summary=True,
    ):
        if metadata is None:
            metadata = {}
        metadata.update(
            {
                self.PL_MODULE_CLASS: pl_module_class,
                self.CHECKPOINT_PATH: os.path.abspath(checkpoint_path),
            }
        )
        if add_run_summary:
            metadata.update(
                {
                    k: v
                    for k, v in wandb.run._summary_get_current_summary_callback().items()
                    if not k.startswith("_")
                }
            )
        self.save_artifact(
            name=artifact_name,
            artifact_type=self.CHECKPOINT_TYPE,
            metadata=metadata,
            description=description,
            local_paths={self.CHECKPOINT_IN_ARTIFACT_HIERARCHY: checkpoint_path},
        )

    def verify_checkpoint_artifact(self, artifact):
        if artifact.type != self.CHECKPOINT_TYPE:
            raise ValueError(
                f"Expected artifact type to be {self.CHECKPOINT_TYPE}, "
                f"but found type {artifact.type}."
            )
        if (
            self.CHECKPOINT_PATH not in artifact.metadata
            or self.PL_MODULE_CLASS not in artifact.metadata
        ):
            raise ValueError(
                f"Malformed checkpoint artifact: '{self.CHECKPOINT_PATH}' "
                f"or '{self.PL_MODULE_CLASS}' "
                f"not in metadata keys {list(artifact.metadata)}"
            )

    def artifact(self, *, entity=None, project=None, artifact_name, alias="latest"):
        entity = entity or self.entity or wandb.run.entity
        project = project or wandb.run.project
        artifact = self.api.artifact(f"{entity}/{project}/{artifact_name}:{alias}")
        return artifact

    def fetch_checkpoint_dict(self, artifact):
        self.verify_checkpoint_artifact(artifact)
        checkpoint_path = artifact.metadata[self.CHECKPOINT_PATH]
        if not os.path.exists(checkpoint_path):
            artifact_dir = artifact.download()
            checkpoint_path = os.path.join(
                artifact_dir,
                artifact.get_path(self.CHECKPOINT_IN_ARTIFACT_HIERARCHY).path,
            )
        return {
            "checkpoint_path": checkpoint_path,
            self.PL_MODULE_CLASS: artifact.metadata[self.PL_MODULE_CLASS],
        }

    def build_from_checkpoint_artifact(
        self, artifact, class_candidates=(), module_candidates=()
    ):
        checkpoint_dict = self.fetch_checkpoint_dict(artifact)
        Model = get_class(
            class_name=checkpoint_dict["pl_module_class"],
            class_candidates=class_candidates,
            module_candidates=module_candidates,
        )
        model = Model.load_from_checkpoint(
            checkpoint_path=checkpoint_dict["checkpoint_path"]
        )
        return model


def pl_module_from_checkpoint_artifact(
    artifact_name, class_candidates=(), module_candidates=()
):
    if wandb.run is None:
        ValueError("Wandb run must be initialized.")
    wandb_api = WandbAPI()
    artifact = wandb_api.artifact(artifact_name=artifact_name)
    model = wandb_api.build_from_checkpoint_artifact(
        artifact=artifact,
        class_candidates=class_candidates,
        module_candidates=module_candidates,
    )
    return model
