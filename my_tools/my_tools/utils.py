import sys
import traceback

import numpy as np
import torch


def free_cuda():
    import gc as garbage_collector

    garbage_collector.collect()
    torch.cuda.empty_cache()


def sparse_dense_multiply(sparse: torch.Tensor, dense: torch.Tensor):
    if not (sparse.is_sparse or sparse.is_sparse_csr) or dense.is_sparse:
        raise ValueError("Incorrect tensor layouts")
    if sparse.is_sparse_csr:
        sparse = sparse.to_sparse_coo()

    indices = sparse._indices()
    values = sparse._values() * dense[indices[0, :], indices[1, :]]
    return torch.sparse_coo_tensor(indices, values, sparse.size(), device=sparse.device)


def scipy_to_torch_sparse(scipy_sparse_csr_matrix, device="cpu"):
    sparse = scipy_sparse_csr_matrix.tocoo()
    torch_sparse_tensor = torch.sparse_coo_tensor(
        indices=np.stack([sparse.row, sparse.col]),
        values=sparse.data,
        size=sparse.shape,
        device=device,
    )
    return torch_sparse_tensor


def reuse_shelved_object_or_construct(
    hashable_attribute, object_constructor, object_name, dir_path="local"
):
    import hashlib
    import shelve

    from pathlib import Path

    dir_path = Path(dir_path)
    dir_path.mkdir(exist_ok=True)
    file_path = dir_path / Path(object_name)
    digest = hashlib.new(name="blake2s", data=hashable_attribute).hexdigest()

    with shelve.open(file_path.as_posix()) as file_dict:
        if file_dict.get("digest") == digest:
            print(f"Reusing {file_path}")
            return file_dict["object"]

        print(f"Constructing {file_path}")
        obj = object_constructor()
        file_dict["digest"] = digest
        file_dict["object"] = obj
        return obj


def timeit(func):
    import datetime

    from functools import wraps as functools_wraps

    @functools_wraps(func)
    def _time_it(*args, **kwargs):
        start = datetime.datetime.now()
        try:
            return func(*args, **kwargs)
        finally:
            end = datetime.datetime.now()
            print(
                f'"{func.__name__}" execution time: {(end - start).total_seconds():.3f} sec'
            )

    return _time_it


def get_class(class_name, class_candidates=(), modules_to_try_to_import_from=()):
    for cls in class_candidates:
        if cls.__name__ == class_name:
            return cls
    for module in modules_to_try_to_import_from:
        if cls := getattr(module, class_name, False):
            return cls
    raise ValueError(
        f"Class {class_name} not found in classes {class_candidates}\n"
        f"or modules{modules_to_try_to_import_from}"
    )


def build_class(class_candidates=(), modules=(), **kwargs):
    class_name = kwargs.pop("name")
    cls = get_class(
        class_name,
        class_candidates=class_candidates,
        modules_to_try_to_import_from=modules,
    )
    return cls(**kwargs)


class StoppingMonitor:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.impatience = 0
        self.min_delta = min_delta
        self.lowest_loss = torch.inf

    def is_time_to_stop(self, loss):
        if loss < self.lowest_loss - self.min_delta:
            self.lowest_loss = loss
            self.impatience = 0
            return False
        self.impatience += 1
        if self.impatience > self.patience:
            self.impatience = 0
            self.lowest_loss = torch.inf
            return True
        return False


def full_traceback(function):
    """Enables full traceback, useful as wandb agent truncates it."""

    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exception:
            print(traceback.print_exc(), file=sys.stderr)
            raise exception

    return wrapper
