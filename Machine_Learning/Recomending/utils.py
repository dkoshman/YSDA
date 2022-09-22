import functools
import os

import numpy as np
import pandas as pd
import scipy.sparse
import torch

from my_ml_tools.entrypoints import ConfigConstructorBase, ConfigDispenser
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


class RecommenderMixin:
    def build_class(self, class_config, class_candidates):
        class_config = class_config.copy()
        model_name = class_config.pop("name")
        model = build_class(model_name, class_config, class_candidates=class_candidates)
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


class RecommendingTrainer(ConfigConstructorBase):
    def main(self):
        self.trainer.fit(self.lightning_module)

class SparseDataModuleMixin:
    def on_after_batch_transfer(self, batch, dataloader_idx):
        return SparseDatasetMixin.maybe_unpack_sparse_kwargs(batch)

class MovieLens:
    def __init__(self, path_to_movielens_folder):
        self.path_to_movielens_folder = path_to_movielens_folder

    def __getitem__(self, key):
        match key:
            case "u.info":
                return self.read(key, ["quantity", "index"], "index", " ",)
            case "u.genre":
                return self.read(key, ["name", "id"], "id")
            case "u.occupation":
                return self.read(key, ["occupation"])
            case "u.user":
                return self.read(key, ["user id", "age", "gender", "occupation", "zip code"], "user id")
            case "u.item":
                return self.read(key, [
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
            "movie id",
            encoding_errors="backslashreplace")
        if key in ["u.data"] + [
            i + j
            for i in ["u1", "u2", "u3", "u4", "u5", "ua", "ub"]
            for j in [".base", ".test"]
        ]:
            return self.read(
            key,
            names=["user_id", "item_id", "rating", "timestamp"],
            sep="\t",
                dtype="int32",
        )

    def read(self, filename, names=None, sep="|", header=None, **kwargs):
        path = os.path.join(self.path_to_movielens_folder, filename)
        dataframe = pd.read_csv(path, names=names, sep=sep, header=header, **kwargs)
        return dataframe.squeeze()

    @functools.lru_cache
    @property
    def shape(self):
        info = self["u.info"]
        return info["users"], info["items"]

    def explicit_feedback_matrix(self, name):
        dataframe = self[name]
        data = dataframe["rating"].to_numpy()
        row_ids = dataframe["user_id"].to_numpy() - 1
        col_ids = dataframe["item_id"].to_numpy() - 1
        explicit_feedback = scipy.sparse.coo_matrix(
            (data, (row_ids, col_ids)), shape=self.shape
        )
        return explicit_feedback.tocsr()
