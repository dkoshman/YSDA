import sys
import traceback
from types import ModuleType
from typing import Any, TypeVar

import numpy as np
import torch
from scipy.sparse import coo_matrix, spmatrix

SparseTensor = TypeVar("SparseTensor", bound=torch.Tensor)
Pickleable = TypeVar("Pickleable")


def free_cuda():
    import gc as garbage_collector

    garbage_collector.collect()
    torch.cuda.empty_cache()


def to_sparse_coo(tensor):
    return tensor if tensor.is_sparse else tensor.to_sparse_coo()


def to_torch_coo(sparse_matrix: torch.Tensor or spmatrix) -> SparseTensor:
    if torch.is_tensor(sparse_matrix):
        return to_sparse_coo(sparse_matrix)
    sparse = sparse_matrix.tocoo()
    torch_sparse_tensor = torch.sparse_coo_tensor(
        indices=np.stack([sparse.row, sparse.col]),
        values=sparse.data,
        size=sparse.shape,
    )
    return torch_sparse_tensor


def to_scipy_coo(sparse_matrix: torch.Tensor or spmatrix) -> coo_matrix:
    if not torch.is_tensor(sparse_matrix):
        return sparse_matrix.tocoo()
    sparse_matrix = to_sparse_coo(sparse_matrix).coalesce().cpu()
    sparse_matrix = coo_matrix(
        (sparse_matrix.values().numpy(), sparse_matrix.indices().numpy()),
        shape=sparse_matrix.shape,
    )
    return sparse_matrix


def torch_sparse_slice(sparse_matrix, row_ids=None, col_ids=None):
    if torch.is_tensor(sparse_matrix):
        sparse_matrix = to_scipy_coo(sparse_matrix)

    if row_ids is None:
        row_ids = slice(None)
    elif torch.is_tensor(row_ids):
        row_ids = row_ids.cpu().numpy()

    if col_ids is None:
        col_ids = slice(None)
    elif torch.is_tensor(col_ids):
        col_ids = col_ids.cpu().numpy()

    sparse_matrix = sparse_matrix.tocsr()[row_ids][:, col_ids].tocoo()
    torch_sparse_coo_tensor = to_torch_coo(sparse_matrix)
    return torch_sparse_coo_tensor


def get_class(class_name, class_candidates=(), module_candidates=()):
    for cls in class_candidates:
        if cls.__name__ == class_name:
            return cls
    for module in module_candidates:
        if cls := getattr(module, class_name, False):
            return cls
    raise ValueError(
        f"Class {class_name} not found in classes {class_candidates}\n"
        f"or modules {module_candidates}"
    )


def build_class(*, class_candidates=(), module_candidates=(), class_name, **kwargs):
    cls = get_class(
        class_name=class_name,
        class_candidates=class_candidates,
        module_candidates=module_candidates,
    )
    return cls(**kwargs)


class BuilderMixin:
    @property
    def module_candidates(self) -> "list[ModuleType]":
        return []

    @property
    def class_candidates(self) -> "list[type]":
        return []

    def build_class(self, *, class_name, **kwargs) -> Any:
        return build_class(
            class_name=class_name,
            class_candidates=self.class_candidates,
            module_candidates=self.module_candidates,
            **kwargs,
        )


def full_traceback(function):
    """Enables full traceback, useful as wandb agent truncates it."""

    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exception:
            print(traceback.print_exc(), file=sys.stderr)
            raise exception

    return wrapper
