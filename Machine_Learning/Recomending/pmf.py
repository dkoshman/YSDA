import itertools

import numpy as np
import pytorch_lightning as pl
import scipy.sparse
import torch
from torch.utils.data import DataLoader, Dataset

from my_ml_tools.entrypoints import ConfigDispenser, ConfigConstructorBase
from my_ml_tools.models import l2_regularization
from my_ml_tools.utils import sparse_dense_multiply

from data import SparseDataModuleMixin, SparseDatasetMixin
from lightning import RecommenderMixin
from utils import torch_sparse_slice


def _build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def _build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


class ProbabilityMatrixFactorization(torch.nn.Module):
    """predicted rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(self, n_users, n_items, latent_dimension, weight_decay):
        super().__init__()

        self.weight_decay = weight_decay

        self.user_weight = _build_weight(n_users, latent_dimension)
        self.user_bias = _build_bias(n_users, 1)

        self.item_weight = _build_weight(n_items, latent_dimension)
        self.item_bias = _build_bias(n_items, 1)

        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.sigmoid = torch.nn.Sigmoid()

    def linear_forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]
        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias

        # Need to add regularization here because otherwise optimizer will decay all weights,
        # not only those corresponding to user and item ids. Also it is important to add gradient
        # hooks after the forward calculations, otherwise decay messes with the model.
        for parameter in [user_weight, item_weight, user_bias, item_bias]:
            l2_regularization(parameter, self.weight_decay)

        return rating

    def forward(self, user_ids, item_ids):
        rating = self.linear_forward(user_ids, item_ids)
        rating = self.sigmoid(rating)
        return rating


class ConstrainedProbabilityMatrixFactorization(ProbabilityMatrixFactorization):
    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension,
        weight_decay,
        implicit_feedback,
    ):
        super().__init__(n_users, n_items, latent_dimension, weight_decay)

        self.item_rating_effect_weight = _build_weight(n_items, latent_dimension)

        self.implicit_feedback_normalized = implicit_feedback.multiply(
            1 / (implicit_feedback.sum(axis=1) + 1e-8)
        ).tocsr()

    def forward(self, user_ids, item_ids):
        # Need to clone to avoid gradient hook accumulation on same tensor and subsequent memory leak
        item_rating_effect_weight = self.item_rating_effect_weight.clone()

        item_weight = self.item_weight[item_ids]

        users_implicit_feedback = torch_sparse_slice(
            self.implicit_feedback_normalized, row_ids=user_ids, device=self.bias.device
        )
        user_weights_offset_caused_by_their_ratings = (
            users_implicit_feedback @ item_rating_effect_weight
        )

        rating = super().linear_forward(user_ids, item_ids)
        rating += user_weights_offset_caused_by_their_ratings @ item_weight.T
        rating = self.sigmoid(rating)

        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        scale_down = self.user_weight.shape[0] / len(user_ids)
        l2_regularization(item_rating_effect_weight, self.weight_decay / scale_down)

        return rating


class PMFDataset(SparseDatasetMixin, Dataset):
    def __init__(
        self,
        explicit_feedback: scipy.sparse.csr_matrix,
        implicit_feedback: scipy.sparse.csr_matrix,
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


class LitProbabilityMatrixFactorization(
    SparseDataModuleMixin, RecommenderMixin, pl.LightningModule
):
    def __init__(
        self, *, model_config, optimizer_config, batch_size=10e8, num_workers=1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = self.build_dataset(self.train_explicit)
        self.val_dataset = self.build_dataset(self.val_explicit)
        self.model = self.build_model(
            model_config,
            model_classes=[
                ProbabilityMatrixFactorization,
                ConstrainedProbabilityMatrixFactorization,
            ],
        )

    def build_model(self, model_config, model_classes):
        model_config["n_users"] = self.train_dataset.shape[0]
        model_config["n_items"] = self.train_dataset.shape[1]
        if model_config["name"] == "ConstrainedProbabilityMatrixFactorization":
            model_config["implicit_feedback"] = self.train_dataset.implicit_feedback
        return super().build_model(model_config, model_classes)

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

    @staticmethod
    def loss(explicit, implicit, model_ratings):
        """Loss = 1/|Explicit| * \sum_{ij}Implicit_{ij} * (Explicit_{ij} - ModelRating_{ij})^2"""
        error = model_ratings - explicit
        error = sparse_dense_multiply(sparse_dense_multiply(implicit, error), error)
        loss = torch.sparse.sum(error) / error._values().numel()
        return loss

    def forward(self, **batch):
        return self.model(
            user_ids=batch.get("user_ids", slice(None)),
            item_ids=batch.get("item_ids", slice(None)),
        )

    def training_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit"], batch["implicit"], ratings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit"], batch["implicit"], ratings)
        self.log("val_loss", loss)
        return loss


class PMFTrainer(ConfigConstructorBase):
    def lightning_module_candidates(self):
        return [LitProbabilityMatrixFactorization]

    def main(self):
        self.trainer.fit(self.lightning_module)


@ConfigDispenser
def main(config):
    PMFTrainer(config).main()


if __name__ == "__main__":
    main()
