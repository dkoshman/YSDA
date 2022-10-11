import functools
import itertools
import os

from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch

from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import DataLoader, Dataset

from utils import torch_sparse_slice


class SparseDataset(Dataset):
    def __init__(self, explicit_feedback: csr_matrix):
        self.explicit_feedback = explicit_feedback

    def __len__(self):
        return self.explicit_feedback.shape[0]

    @property
    def shape(self):
        return self.explicit_feedback.shape

    def __getitem__(self, indices):
        """
        Here's where implicit feedback is implicitly generated from explicit feedback.
        For datasets where only implicit feedback is available, override this logic.
        """
        user_ids = indices.get("user_ids")
        item_ids = indices.get("item_ids")

        explicit = torch_sparse_slice(self.explicit_feedback, user_ids, item_ids)
        implicit = explicit.bool().int()
        return dict(
            explicit=self.pack_sparse_tensor(explicit),
            implicit=self.pack_sparse_tensor(implicit),
            user_ids=user_ids,
            item_ids=item_ids,
        )

    @staticmethod
    def pack_sparse_tensor(torch_sparse):
        torch_sparse = torch_sparse.coalesce()
        sparse_kwargs = dict(
            indices=torch_sparse.indices(),
            values=torch_sparse.values(),
            size=torch_sparse.size(),
        )
        return sparse_kwargs

    @staticmethod
    def unpack_sparse_kwargs_to_torch_sparse_coo(batch):
        for key, value in batch.items():
            match value:
                case {"indices": _, "values": _, "size": _} as kwargs:
                    batch[key] = torch.sparse_coo_tensor(**kwargs).to_sparse_csr()
        return batch


class SparseDataModuleMixin:
    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        """
        Need to manually pack and then unpack sparse torch matrices because
        they are unsupported in Dataloaders as of the moment of writing.
        """
        return SparseDataset.unpack_sparse_kwargs_to_torch_sparse_coo(batch)


class Sampler:
    def __init__(self, size, batch_size, shuffle=True):
        indices = torch.randperm(size) if shuffle else torch.arange(size)
        self.batch_indices = torch.split(indices, batch_size)

    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        yield from self.batch_indices


class UserSampler(Sampler):
    def __iter__(self):
        yield from ({"user_ids": i} for i in super().__iter__())


class ItemSampler(Sampler):
    def __iter__(self):
        yield from ({"item_ids": i} for i in super().__iter__())


class GridSampler:
    """
    Splits user ids and item ids into chunks, and uses
    cartesian product of these chunked ids to generate batches.
    """

    def __init__(self, dataset_shape, approximate_batch_size, shuffle=True):
        self.dataset_shape = dataset_shape
        self.chunks_per_dim = (
            (
                (torch.tensor(dataset_shape).prod() / approximate_batch_size)
                ** (1 / len(dataset_shape))
            )
            .round()
            .int()
        )
        self.shuffle = shuffle

    def __len__(self):
        return self.chunks_per_dim ** len(self.dataset_shape)

    def __iter__(self):
        batch_indices_per_dimension = [
            torch.randperm(dimension_size).chunk(self.chunks_per_dim)
            for dimension_size in self.dataset_shape
        ]
        numpy_batches = [[j.numpy() for j in i] for i in batch_indices_per_dimension]
        batch_indices_product = itertools.product(*numpy_batches)
        if self.shuffle:
            batch_indices_product = np.array(list(batch_indices_product), dtype=object)
            batch_indices_product = np.random.permutation(batch_indices_product)
        yield from ({"user_ids": i[0], "item_ids": i[1]} for i in batch_indices_product)


def build_recommending_sampler(
    batch_size,
    n_users,
    n_items,
    sampler_type: Literal["grid", "user", "item"],
    shuffle,
):
    match sampler_type:
        case "grid":
            grid_batch_size = int(batch_size**2 * n_items / n_users)
            sampler = GridSampler(
                dataset_shape=(n_users, n_items),
                approximate_batch_size=grid_batch_size,
                shuffle=shuffle,
            )
        case "user":
            sampler = UserSampler(
                size=n_users,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        case "item":
            sampler = ItemSampler(
                size=n_items,
                batch_size=batch_size,
                shuffle=shuffle,
            )
        case _:
            ValueError(f"Unknown sampler type {sampler_type}.")
    return sampler


def build_recommending_dataloader(
    dataset,
    sampler_type: Literal["grid", "user", "item"] = "user",
    batch_size=100,
    num_workers=0,
    persistent_workers=False,
    shuffle=False,
):
    sampler = build_recommending_sampler(
        batch_size=batch_size,
        n_users=dataset.shape[0],
        n_items=dataset.shape[1],
        sampler_type=sampler_type,
        shuffle=shuffle,
    )
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=isinstance(num_workers, int) and num_workers > 1,
        persistent_workers=persistent_workers,
    )
    return dataloader


class MovieLens:
    def __init__(self, path_to_movielens_folder="local/ml-100k"):
        self.path_to_movielens_folder = path_to_movielens_folder

    def __getitem__(self, filename):
        match filename:
            case "u.info":
                return self.read(
                    filename=filename,
                    names=["quantity", "index"],
                    index_col="index",
                    sep=" ",
                )
            case "u.genre":
                return self.read(
                    filename=filename, names=["name", "id"], index_col="id"
                )
            case "u.occupation":
                return self.read(filename=filename, names=["occupation"])
            case "u.user":
                return self.read(
                    filename=filename,
                    names=["user id", "age", "gender", "occupation", "zip code"],
                    index_col="user id",
                )
            case "u.item":
                return self.read(
                    filename=filename,
                    names=[
                        "movie id",
                        "movie title",
                        "release date",
                        "video release date",
                        "IMDb URL",
                        "unknown",
                        "Action",
                        "Adventure",
                        "Animation",
                        "Children's",
                        "Comedy",
                        "Crime",
                        "Documentary",
                        "Drama",
                        "Fantasy",
                        "Film-Noir",
                        "Horror",
                        "Musical",
                        "Mystery",
                        "Romance",
                        "Sci-Fi",
                        "Thriller",
                        "War",
                        "Western",
                    ],
                    index_col="movie id",
                    encoding_errors="backslashreplace",
                )
        if filename in ["u.data"] + [
            i + j
            for i in ["u1", "u2", "u3", "u4", "u5", "ua", "ub"]
            for j in [".base", ".test"]
        ]:
            return self.read(
                filename=filename,
                names=["user_id", "item_id", "rating", "timestamp"],
                sep="\t",
                dtype="int32",
            )
        raise ValueError(f"File {filename} not found.")

    def read(self, filename, sep="|", header=None, **kwargs):
        path = os.path.join(self.path_to_movielens_folder, filename)
        dataframe = pd.read_csv(path, sep=sep, header=header, **kwargs)
        return dataframe.squeeze()

    @property
    @functools.cache
    def shape(self):
        info = self["u.info"]
        return info["users"], info["items"]

    def explicit_feedback_scipy_csr(self, name):
        dataframe = self[name]
        data = dataframe["rating"].to_numpy()
        row_ids = dataframe["user_id"].to_numpy() - 1
        col_ids = dataframe["item_id"].to_numpy() - 1
        explicit_feedback = coo_matrix((data, (row_ids, col_ids)), shape=self.shape)
        return explicit_feedback.tocsr()
