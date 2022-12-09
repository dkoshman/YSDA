import itertools

from typing import Literal, TYPE_CHECKING

import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset, IterableDataset

from my_tools.utils import torch_sparse_slice

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from scipy.sparse import csr_matrix


class SparseDataset(Dataset):
    def __init__(self, explicit: "csr_matrix"):
        self.explicit = explicit

    def __len__(self):
        return self.explicit.shape[0]

    @property
    def shape(self):
        return self.explicit.shape

    def __getitem__(self, indices):
        user_ids = indices.get("user_ids", torch.arange(self.shape[0]))
        item_ids = indices.get("item_ids", torch.arange(self.shape[1]))

        explicit = torch_sparse_slice(self.explicit, user_ids, item_ids)
        return dict(
            explicit=self.pack_sparse_tensor(explicit),
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
    def unpack_sparse_kwargs_to_torch_sparse_csr(batch):
        for key, value in batch.items():
            if isinstance(value, dict) and set(value) == {
                "indices",
                "values",
                "size",
            }:
                batch[key] = torch.sparse_coo_tensor(**value).to_sparse_csr()
        return batch


class SparseTensorUnpacker:
    def on_after_batch_transfer(self: "pl.LightningModule", batch, dataloader_idx=0):
        """
        Need to manually pack and then unpack sparse torch matrices because
        they are unsupported in Dataloaders as of the moment of writing.
        """
        return SparseDataset.unpack_sparse_kwargs_to_torch_sparse_csr(batch)


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
        self.chunks_per_dim = max(
            1,
            (
                (torch.tensor(dataset_shape).prod() / approximate_batch_size)
                ** (1 / len(dataset_shape))
            )
            .round()
            .int(),
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


class GridIterableDataset(GridSampler, IterableDataset):
    pass


def build_recommending_sampler(
    batch_size,
    n_users,
    n_items,
    sampler_type: Literal["grid", "user", "item"],
    shuffle,
):
    if sampler_type == "grid":
        grid_batch_size = int(batch_size**2 * n_items / n_users)
        sampler = GridSampler(
            dataset_shape=(n_users, n_items),
            approximate_batch_size=grid_batch_size,
            shuffle=shuffle,
        )
    elif sampler_type == "user":
        sampler = UserSampler(
            size=n_users,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    elif sampler_type == "item":
        sampler = ItemSampler(
            size=n_items,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}.")
    return sampler


def build_recommending_dataloader(
    dataset,
    sampler_type: Literal["grid", "user", "item"] = "user",
    batch_size=100,
    num_workers=4,
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


class SparseDataModuleInterface:
    def train_explicit(self) -> "csr_matrix" or None:
        return

    def val_explicit(self) -> "csr_matrix" or None:
        return

    def test_explicit(self) -> "csr_matrix" or None:
        return


class SparseDataModuleBase(SparseDataModuleInterface, SparseTensorUnpacker):
    def build_dataloader(self, **kwargs):
        return build_recommending_dataloader(**kwargs)

    def train_dataloader(self):
        if (explicit := self.train_explicit()) is not None:
            return self.build_dataloader(
                dataset=SparseDataset(explicit),
                sampler_type="user",
                shuffle=True,
            )

    def val_dataloader(self):
        if (explicit := self.val_explicit()) is not None:
            return self.build_dataloader(
                dataset=SparseDataset(explicit),
                sampler_type="user",
            )

    def test_dataloader(self):
        if (explicit := self.test_explicit()) is not None:
            return self.build_dataloader(
                dataset=SparseDataset(explicit),
                sampler_type="user",
            )
