def reuse_pickled_object_or_construct(
        hashable_attribute, object_constructor, algorithm_name="blake2s", dirpath="local"
):
    import hashlib
    import pickle

    from pathlib import Path

    hexdigest = hashlib.new(name=algorithm_name, data=hashable_attribute).hexdigest()
    file_path = Path.cwd() / Path(dirpath) / Path(hexdigest + "." + algorithm_name)

    if file_path.exists():
        print(f"Reusing {file_path}")
        obj = pickle.load(open(file_path, "rb"))
    else:
        print(f"Constructing {file_path}")
        obj = object_constructor()
        pickle.dump(obj, open(file_path, "wb"))

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


def free_cuda():
    import gc as garbage_collector

    import torch

    garbage_collector.collect()
    torch.cuda.empty_cache()


def sparse_dense_multiply(sparse: "torch.Tensor", dense: "torch.Tensor"):
    import torch

    if not sparse.is_sparse or dense.is_sparse:
        raise ValueError("Incorrect tensor types")

    indices = sparse._indices()
    values = sparse._values() * dense[indices[0, :], indices[1, :]]
    return torch.sparse_coo_tensor(indices, values, sparse.size(), device=sparse.device)
