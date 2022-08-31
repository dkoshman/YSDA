import itertools

import numpy as np
import pytorch_lightning as pl
import scipy.sparse
import torch

from torch.utils.data import Dataset, DataLoader


class SparseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path="local/train_explicit.npz",
        val_path="local/val_explicit.npz",
        batch_size=1e8,
        num_workers=1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = self.build_dataset(train_path)
        self.val_dataset = self.build_dataset(val_path)

    @staticmethod
    def build_dataset(path):
        explicit_train = scipy.sparse.load_npz(path).tocsr()
        implicit_train = explicit_train > 0
        return SparseDataset(explicit_train, implicit_train)

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

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch["explicit"] = self.train_dataset.unpack_sparse_tensor(
            **batch["explicit_sparse_kwargs"]
        )
        batch["implicit"] = self.train_dataset.unpack_sparse_tensor(
            **batch["implicit_sparse_kwargs"]
        )
        return batch

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


class SparseDataset(Dataset):
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
            data = explicit_feedback.data
            lower = np.quantile(data, 0.01)
            upper = np.quantile(data, 0.99)
            data = np.clip(data, lower, upper)
            data = (data - lower) / (upper - lower)
            explicit_feedback.data = data

            self.explicit_feedback = explicit_feedback
            self.implicit_feedback = implicit_feedback.astype(bool).astype(np.float32)

    def __len__(self):
        return np.prod(self.explicit_feedback.shape)

    @property
    def shape(self):
        return self.explicit_feedback.shape

    @staticmethod
    def carve_sparse_matrix_and_pack(sparse_matrix, user_ids, item_ids):
        sparse_matrix = sparse_matrix[user_ids][:, item_ids].tocoo()
        torch_sparse_tensor_init_kwargs = dict(
            indices=np.stack([sparse_matrix.row, sparse_matrix.col]),
            values=sparse_matrix.data,
            size=sparse_matrix.shape,
        )
        return torch_sparse_tensor_init_kwargs

    @staticmethod
    def unpack_sparse_tensor(*, indices, values, size):
        return torch.sparse_coo_tensor(
            indices=indices, values=values, size=size, device=values.device
        )

    def __getitem__(self, indices):
        user_ids, item_ids = indices
        explicit_sparse_kwargs = self.carve_sparse_matrix_and_pack(
            self.explicit_feedback, user_ids, item_ids
        )
        implicit_sparse_kwargs = self.carve_sparse_matrix_and_pack(
            self.implicit_feedback, user_ids, item_ids
        )
        return dict(
            explicit_sparse_kwargs=explicit_sparse_kwargs,
            implicit_sparse_kwargs=implicit_sparse_kwargs,
            user_ids=user_ids,
            item_ids=item_ids,
        )
