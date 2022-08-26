import torch


def reuse_shelved_object_or_construct(
    hashable_attribute, object_constructor, object_name, dir_path="local"
):
    import hashlib
    import shelve

    from pathlib import Path

    digest = hashlib.new(name="blake2s", data=hashable_attribute).hexdigest()
    file_path = Path(dir_path) / Path(object_name + ".shelf")

    if file_path.exists():
        with shelve.open(file_path) as file_dict:
            if file_dict.get("digest") == digest:
                print(f"Reusing {file_path}")
                return file_dict["object"]

    print(f"Constructing {file_path}")
    obj = object_constructor()
    with shelve.open(file_path) as file_dict:
        file_dict["digest"] = digest
        file_dict["object"] = obj
        return obj


def free_cuda():
    import gc as garbage_collector

    garbage_collector.collect()
    torch.cuda.empty_cache()


def sparse_dense_multiply(sparse: torch.Tensor, dense: torch.Tensor):
    if not sparse.is_sparse or dense.is_sparse:
        raise ValueError("Incorrect tensor layouts")

    indices = sparse._indices()
    values = sparse._values() * dense[indices[0, :], indices[1, :]]
    return torch.sparse_coo_tensor(indices, values, sparse.size(), device=sparse.device)


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
