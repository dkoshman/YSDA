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
    if tensor.is_sparse:
        return tensor
    return tensor.to_sparse_coo()


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


#
# def reuse_shelved_object_or_construct(
#     hashable_attribute, object_constructor, object_name, dir_path="local"
# ):
#     import hashlib
#     import shelve
#
#     from pathlib import Path
#
#     dir_path = Path(dir_path)
#     dir_path.mkdir(exist_ok=True)
#     file_path = dir_path / Path(object_name)
#     digest = hashlib.new(name="blake2s", data=hashable_attribute).hexdigest()
#
#     with shelve.open(file_path.as_posix()) as file_dict:
#         if file_dict.get("digest") == digest:
#             print(f"Reusing {file_path}")
#             return file_dict["object"]
#
#         print(f"Constructing {file_path}")
#         obj = object_constructor()
#         file_dict["digest"] = digest
#         file_dict["object"] = obj
#         return obj

#
# def timeit(func):
#     import datetime
#
#     from functools import wraps as functools_wraps
#
#     @functools_wraps(func)
#     def _time_it(*args, **kwargs):
#         start = datetime.datetime.now()
#         try:
#             return func(*args, **kwargs)
#         finally:
#             end = datetime.datetime.now()
#             print(
#                 f'"{func.__name__}" execution time: {(end - start).total_seconds():.3f} sec'
#             )
#
#     return _time_it


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
    def module_candidates(self) -> list[ModuleType]:
        return []

    @property
    def class_candidates(self) -> list[type]:
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
