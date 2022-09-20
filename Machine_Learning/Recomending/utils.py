import numpy as np
import scipy.sparse
import torch

from my_ml_tools.entrypoints import ConfigDispenser


def slice_sparse_matrix(sparse_matrix, row_ids, col_ids) -> dict:
    if torch.is_tensor(sparse_matrix):
        try:
            sparse_matrix = sparse_matrix.to_sparse_coo()
        except NotImplementedError:
            pass
        sparse_matrix = sparse_matrix.coalesce()
        sparse_matrix = scipy.sparse.coo_matrix(
            (sparse_matrix.values().cpu(), sparse_matrix.indices().cpu().numpy()),
            shape=sparse_matrix.shape,
        )

    sparse_matrix = sparse_matrix.tocsr()[row_ids][:, col_ids].tocoo()
    torch_sparse_tensor_init_kwargs = dict(
        indices=np.stack([sparse_matrix.row, sparse_matrix.col]),
        values=sparse_matrix.data,
        size=sparse_matrix.shape,
    )
    return torch_sparse_tensor_init_kwargs


def unpack_sparse_tensor(*, indices, values, size, device=None):
    return torch.sparse_coo_tensor(
        indices=indices, values=values, size=size, device=device
    )


def maybe_tensor_to_numpy(maybe_tensor):
    return maybe_tensor.cpu().numpy() if torch.is_tensor(maybe_tensor) else maybe_tensor


def torch_sparse_slice(sparse_matrix, row_ids=None, col_ids=None, device=None):
    row_ids = maybe_tensor_to_numpy(row_ids) if row_ids is not None else slice(None)
    col_ids = maybe_tensor_to_numpy(col_ids) if col_ids is not None else slice(None)
    return unpack_sparse_tensor(
        **slice_sparse_matrix(sparse_matrix, row_ids, col_ids), device=device
    )


class RecommendingConfigDispenser(ConfigDispenser):
    def debug_config(self, config):
        config["trainer"].update(
            dict(
                devices=None,
                accelerator=None,
            )
        )
        config["lightning_module"].update(
            dict(
                train_path="local/train_explicit_debug.npz",
                val_path="local/val_explicit_debug.npz",
            )
        )
        return config
