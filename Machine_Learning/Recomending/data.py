import itertools
from abc import ABC

import numpy as np
import pytorch_lightning as pl
import scipy.sparse
import torch

from torch.utils.data import Dataset, DataLoader

from utils import slice_sparse_matrix, unpack_sparse_tensor


class SparseDataset(Dataset, ABC):
    packed_key_suffix = "__sparse_kwargs__"

    @staticmethod
    def normalize_feedback(
        feedback: scipy.sparse.csr.csr_matrix,
        lower_outlier_quantile=0.01,
        upper_outlier_quantile=0.99,
    ):
        data = feedback.data
        lower = np.quantile(data, lower_outlier_quantile)
        upper = np.quantile(data, upper_outlier_quantile)
        data = np.clip(data, lower, upper)
        data = (data - lower) / (upper - lower)
        feedback.data = data
        return feedback

    def pack_sparse_slice_into_dict(
        self, key, sparse_matrix, user_ids=slice(None), item_ids=slice(None)
    ) -> dict[str:dict]:
        sparse_kwargs = slice_sparse_matrix(sparse_matrix, user_ids, item_ids)
        return {key + self.packed_key_suffix: sparse_kwargs}

    @staticmethod
    def maybe_unpack_sparse_kwargs(batch):
        keys = list(batch.keys())
        for key in keys:
            if key.endswith(SparseDataset.packed_key_suffix):
                batch[
                    key.removesuffix(SparseDataset.packed_key_suffix)
                ] = unpack_sparse_tensor(**batch[key])
                del batch[key]

        return batch


class SparseDataModule(pl.LightningDataModule, ABC):
    def __init__(
        self,
        batch_size,
        train_path="local/train_explicit.npz",
        val_path="local/val_explicit.npz",
        num_workers=1,
    ):
        # Call to super is needed not only for lightning,
        # but for cooperative subclassing to work.
        # So if a method is ever going to be overwritten, it is a good practice
        # to put in call to super, even if it is redundant at the moment.
        super().__init__()
        self.save_hyperparameters()

    @property
    def train_explicit(self):
        return scipy.sparse.load_npz(self.hparams["train_path"]).tocsr()

    @property
    def val_explicit(self):
        return scipy.sparse.load_npz(self.hparams["val_path"]).tocsr()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return SparseDataset.maybe_unpack_sparse_kwargs(batch)


class PMFDataModule(SparseDataModule):
    def __init__(self, *, batch_size, **kwargs):
        super().__init__(batch_size=batch_size, **kwargs)
        self.train_dataset = self.build_dataset(self.train_explicit)
        self.val_dataset = self.build_dataset(self.val_explicit)

    @staticmethod
    def build_dataset(explicit_feedback):
        implicit_feedback = explicit_feedback > 0
        return PMFDataset(explicit_feedback, implicit_feedback)

    def build_dataloader(self, dataset, shuffle):
        sampler = GridSampler(
            dataset_shape=dataset.shape,
            approximate_batch_size=self.hparams["batch_size"],
            shuffle=shuffle,
        )
        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_size=None,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["num_workers"] > 1,
        )

    def train_dataloader(self):
        return self.build_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.build_dataloader(self.val_dataset, shuffle=False)


class GridSampler:
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
        if not self.shuffle:
            yield from batch_indices_product
        else:
            batch_indices_product = np.array(list(batch_indices_product), dtype=object)
            yield from np.random.permutation(batch_indices_product)


class PMFDataset(SparseDataset):
    def __init__(
        self,
        explicit_feedback: scipy.sparse.csr.csr_matrix,
        implicit_feedback: scipy.sparse.csr.csr_matrix,
        normalize=True,
    ):
        assert explicit_feedback.shape == implicit_feedback.shape

        if not normalize:
            self.explicit_feedback = explicit_feedback
            self.implicit_feedback = implicit_feedback
        else:
            self.explicit_feedback = self.normalize_feedback(explicit_feedback)
            self.implicit_feedback = implicit_feedback.astype(bool).astype(np.float32)

    def __len__(self):
        return np.prod(self.explicit_feedback.shape)

    @property
    def shape(self):
        return self.explicit_feedback.shape

    def __getitem__(self, indices):
        user_ids, item_ids = indices
        explicit_sparse_kwargs = self.pack_sparse_slice_into_dict(
            "explicit", self.explicit_feedback, user_ids, item_ids
        )
        implicit_sparse_kwargs = self.pack_sparse_slice_into_dict(
            "implicit", self.implicit_feedback, user_ids, item_ids
        )
        return dict(
            **explicit_sparse_kwargs,
            **implicit_sparse_kwargs,
            user_ids=user_ids,
            item_ids=item_ids,
        )


class SLIMDataset(SparseDataset):
    def __init__(
        self,
        explicit_train: scipy.sparse.csr.csr_matrix,
        explicit_val: scipy.sparse.csr.csr_matrix,
    ):
        assert explicit_train.shape == explicit_val.shape

        self.explicit_train = explicit_train
        self.explicit_val = explicit_val

    def __len__(self):
        return self.explicit_train.shape[1]

    def __getitem__(self, item_ids):
        explicit_train_kwargs = self.pack_sparse_slice_into_dict(
            "explicit_train", self.explicit_train, item_ids=item_ids
        )
        explicit_val_kwargs = self.pack_sparse_slice_into_dict(
            "explicit_val", self.explicit_val, item_ids=item_ids
        )

        return dict(
            **explicit_train_kwargs,
            **explicit_val_kwargs,
            item_ids=item_ids,
            user_ids=None,
        )


class SLIMSampler:
    def __init__(self, n_items, batch_size):
        self.n_items = n_items
        self.batch_size = batch_size
        self.batch_indices = torch.split(torch.arange(n_items), batch_size)

    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        yield from self.batch_indices


class SLIMDataModule(SparseDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_batch = None
        self.batch_is_fitted = False
        self.dataset = SLIMDataset(self.train_explicit, self.val_explicit)
        self.dataloader_iter = iter(self.dataloader)

    @property
    def dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            sampler=SLIMSampler(
                n_items=len(self.dataset), batch_size=self.hparams["batch_size"]
            ),
            batch_size=None,
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["num_workers"] > 1,
        )

    def train_dataloader(self):
        try:
            self.current_batch = next(self.dataloader_iter)
        except StopIteration:
            raise KeyboardInterrupt("Training finished")

        self.batch_is_fitted = False
        while not self.batch_is_fitted:
            yield self.current_batch

    def __iter__(self):
        yield SparseDataset.maybe_unpack_sparse_kwargs(self.current_batch)

    def val_dataloader(self):
        return self
