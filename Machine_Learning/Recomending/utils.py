import numpy as np
import scipy.sparse
import torch

from my_ml_tools.entrypoints import ConfigDispenser
from my_ml_tools.utils import build_class


def slice_sparse_matrix(sparse_matrix, row_ids, col_ids) -> dict:
    if torch.is_tensor(sparse_matrix):
        try:
            sparse_matrix = sparse_matrix.to_sparse_coo()
        except NotImplementedError:
            pass
        sparse_matrix = sparse_matrix.coalesce()
        sparse_matrix = scipy.sparse.coo_matrix(
            (sparse_matrix.values().cpu(), sparse_matrix.indices().cpu().numpy()),
            shape=sparse_matrix.shape,
        )

    sparse_matrix = sparse_matrix.tocsr()[row_ids][:, col_ids].tocoo()
    torch_sparse_tensor_init_kwargs = dict(
        indices=np.stack([sparse_matrix.row, sparse_matrix.col]),
        values=sparse_matrix.data,
        size=sparse_matrix.shape,
    )
    return torch_sparse_tensor_init_kwargs


def unpack_sparse_tensor(*, indices, values, size, device=None):
    return torch.sparse_coo_tensor(
        indices=indices, values=values, size=size, device=device
    )


def torch_sparse_slice(sparse_matrix, row_ids=None, col_ids=None, device=None):
    def maybe_tensor_to_numpy(maybe_tensor):
        if maybe_tensor is None:
            return slice(None)
        elif torch.is_tensor(maybe_tensor):
            return maybe_tensor.cpu().numpy()
        else:
            return maybe_tensor

    row_ids = maybe_tensor_to_numpy(row_ids)
    col_ids = maybe_tensor_to_numpy(col_ids)
    return unpack_sparse_tensor(
        **slice_sparse_matrix(sparse_matrix, row_ids, col_ids), device=device
    )


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
    def train_explicit(self, train_path="local/train_explicit.npz"):
        return scipy.sparse.load_npz(train_path).tocsr()

    def val_explicit(self, val_path="local/val_explicit.npz"):
        return scipy.sparse.load_npz(val_path).tocsr()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return SparseDatasetMixin.maybe_unpack_sparse_kwargs(batch)


class RecommenderMixin:
    def build_model(self, model_config, model_classes):
        model_config = model_config.copy()
        model_name = model_config.pop("name")
        model = build_class(model_name, model_config, class_candidates=model_classes)
        return model

    def configure_optimizers(self):
        optimizer_config = self.hparams["optimizer_config"]
        # Need to copy, otherwise tensor parameters will be saved to hparams
        optimizer_config = optimizer_config.copy()
        optimizer_config["params"] = self.parameters()
        optimizer_name = optimizer_config.pop("name")
        optimizer = build_class(
            optimizer_name,
            optimizer_config,
            modules_to_try_to_import_from=[torch.optim],
        )
        return optimizer


class RecommendingConfigDispenser(ConfigDispenser):
    def debug_config(self, config):
        config["trainer"].update(
            dict(
                devices=None,
                accelerator=None,
            )
        )
        config["lightning_module"].update(
            dict(
                train_path="local/train_explicit_debug.npz",
                val_path="local/val_explicit_debug.npz",
            )
        )
        return config
