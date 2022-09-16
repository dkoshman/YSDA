import numpy as np
import scipy.sparse

from utils import slice_sparse_matrix, unpack_sparse_tensor


class SparseDatasetMixin:
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
            if key.endswith(SparseDatasetMixin.packed_key_suffix):
                batch[
                    key.removesuffix(SparseDatasetMixin.packed_key_suffix)
                ] = unpack_sparse_tensor(**batch[key])
                del batch[key]

        return batch


class SparseDataModuleMixin:
    def train_explicit(self, train_path):
        return scipy.sparse.load_npz(train_path).tocsr()

    def val_explicit(self, val_path="local/val_explicit.npz"):
        return scipy.sparse.load_npz(val_path).tocsr()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return SparseDatasetMixin.maybe_unpack_sparse_kwargs(batch)
