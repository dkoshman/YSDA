import numpy as np
import scipy
import torch


from torch.utils.data import Dataset


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
    def to_sparse_tensor(sparse_matrix, indices):
        user_ids, item_ids = indices
        sparse_matrix = sparse_matrix[user_ids][:, item_ids].tocoo()
        sparse_tensor = torch.sparse_coo_tensor(
            indices=np.stack([sparse_matrix.row, sparse_matrix.col]),
            values=sparse_matrix.data,
            size=sparse_matrix.shape,
        )
        return sparse_tensor

    def __getitem__(self, indices):
        explicit = self.to_sparse_tensor(self.explicit_feedback, indices)
        implicit = self.to_sparse_tensor(self.implicit_feedback, indices)
        return dict(
            explicit=explicit,
            implicit=implicit,
            user_ids=indices[0],
            item_ids=indices[1],
        )

import itertools


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
        indices = [torch.randperm(i) for i in self.dataset_shape]
        indices = [[j.numpy() for j in i.chunk(self.chunks_per_dim)] for i in indices]
        iterator = itertools.product(*indices)
        if not self.shuffle:
            return iterator
        else:
            yield from np.random.permutation(list(iterator))
