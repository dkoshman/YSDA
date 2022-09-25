import functools
import itertools
import os

import numpy as np
import pandas as pd
import pytorch_lightning
import torch

from scipy.sparse import coo_matrix, csr_matrix
from torch.utils.data import Dataset, DataLoader
from utils import torch_sparse_slice


class SparseDataset(Dataset):
    def __init__(
        self,
        explicit_feedback: csr_matrix,
        normalize=True,
        lower_outlier_quantile=0,
        upper_outlier_quantile=1,
    ):
        self.explicit_feedback = explicit_feedback
        if normalize:
            self.explicit_feedback = self.normalize_feedback(
                explicit_feedback, lower_outlier_quantile, upper_outlier_quantile
            )

    @staticmethod
    def normalize_feedback(
        feedback: csr_matrix, lower_outlier_quantile=0, upper_outlier_quantile=1
    ):
        """Clip outliers and project values to the [0, 1] interval."""
        data = feedback.data
        lower = np.quantile(data, lower_outlier_quantile)
        upper = np.quantile(data, upper_outlier_quantile)
        data = np.clip(data, lower, upper)
        data = (data - lower) / (upper - lower)
        feedback.data = data
        return feedback

    def __len__(self):
        return self.explicit_feedback.shape[0]

    @property
    def shape(self):
        return self.explicit_feedback.shape

    def __getitem__(self, user_ids, item_ids=None):
        """
        Here's where implicit feedback is implicitly generated from explicit feedback.
        For datasets where only implicit feedback is available, override this logic.
        """
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


class BatchSampler:
    def __init__(self, n_items, batch_size, shuffle=True):
        self.n_items = n_items
        self.batch_size = batch_size
        indices = torch.randperm(n_items) if shuffle else torch.arange(n_items)
        self.batch_indices = torch.split(indices, batch_size)

    def __len__(self):
        return len(self.batch_indices)

    def __iter__(self):
        yield from self.batch_indices


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


class MovieLens:
    def __init__(self, path_to_movielens_folder):
        self.path_to_movielens_folder = path_to_movielens_folder

    def __getitem__(self, key):
        match key:
            case "u.info":
                return self.read(
                    filename=key,
                    names=["quantity", "index"],
                    index_col="index",
                    sep=" ",
                )
            case "u.genre":
                return self.read(filename=key, names=["name", "id"], index_col="id")
            case "u.occupation":
                return self.read(filename=key, names=["occupation"])
            case "u.user":
                return self.read(
                    filename=key,
                    names=["user id", "age", "gender", "occupation", "zip code"],
                    index_col="user id",
                )
            case "u.item":
                return self.read(
                    filename=key,
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
        if key in ["u.data"] + [
            i + j
            for i in ["u1", "u2", "u3", "u4", "u5", "ua", "ub"]
            for j in [".base", ".test"]
        ]:
            return self.read(
                filename=key,
                names=["user_id", "item_id", "rating", "timestamp"],
                sep="\t",
                dtype="int32",
            )

    def read(self, filename, sep="|", header=None, **kwargs):
        path = os.path.join(self.path_to_movielens_folder, filename)
        dataframe = pd.read_csv(path, sep=sep, header=header, **kwargs)
        return dataframe.squeeze()

    @functools.lru_cache
    @property
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

    def sparse_dataset(self, name, **kwargs):
        explicit = self.explicit_feedback_scipy_csr(name)
        return SparseDataset(explicit, **kwargs)


class MovieLensDataModule(SparseDataModuleMixin, pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        data_folder,
        batch_size=10e8,
        num_workers=1,
        train_explicit_feedback_file=None,
        val_explicit_feedback_file=None,
        test_explicit_feedback_file=None,
        val_dataset_fraction=0.1,
    ):
        super().__init__()
        self.common_dataloader_params = dict(
            num_workers=self.hparams["num_workers"],
            pin_memory=self.hparams["num_workers"] > 1,
        )
        self.save_hyperparameters()
        self.movielens = MovieLens(data_folder)

    #     self.train_dataset, self.val_dataset = self.maybe_split_train_dataset()
    #
    # def maybe_split_train_dataset(self):
    #     train_dataset = self.movielens.sparse_dataset(
    #         self.hparams["train_explicit_feedback_file"]
    #     )
    #     if (val_file := self.hparams["val_explicit_feedback_file"]) is not None:
    #         val_dataset = self.movielens.sparse_dataset(val_file)
    #         return train_dataset, val_dataset
    #
    #     val_fraction = self.hparams["val_dataset_fraction"]
    #     return torch.utils.data.random_split(
    #         train_dataset, [(n := len(train_dataset)) - (m := int(val_fraction * n)), m]
    #     )

    def grid_dataloader(self, filename):
        dataset = self.movielens.sparse_dataset(filename)
        batch_size = (
            self.hparams["batch_size"] * np.prod(dataset.shape) / dataset.shape[0]
        )
        sampler = GridSampler(
            dataset_shape=dataset.shape,
            approximate_batch_size=batch_size,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset=dataset, batch_sampler=sampler, **self.common_dataloader_params
        )
        return dataloader

    def user_wise_dataloader(self, filename):
        dataset = self.movielens.sparse_dataset(filename)
        sampler = BatchSampler(len(dataset), self.hparams["batch_size"])
        dataloader = DataLoader(
            dataset=dataset, batch_sampler=sampler, **self.common_dataloader_params
        )
        return dataloader

    def train_dataloader(self):
        return self.grid_dataloader(self.hparams["train_explicit_feedback_file"])

    def val_dataloader(self):
        return self.user_wise_dataloader(self.hparams["val_explicit_feedback_file"])

    def test_dataloader(self):
        return self.user_wise_dataloader(self.hparams["test_explicit_feedback_file"])
