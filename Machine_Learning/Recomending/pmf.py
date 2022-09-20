import einops
from typing import Literal

import itertools

import numpy as np
import pytorch_lightning as pl
import scipy.sparse
import torch
from torch.utils.data import DataLoader, Dataset

from my_ml_tools.entrypoints import ConfigConstructorBase
from my_ml_tools.lightning import ConvenientCheckpointLogCallback
from my_ml_tools.models import register_regularization_hook
from my_ml_tools.utils import sparse_dense_multiply

from data import SparseDataModuleMixin, SparseDatasetMixin
from lightning import RecommenderMixin
from utils import torch_sparse_slice, RecommendingConfigDispenser


def _build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def _build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


class ProbabilityMatrixFactorization(torch.nn.Module):
    """predicted rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(
        self,
        n_users,
        n_items,
        latent_dimension,
        weight_decay,
        pass_through_sigmoid=False,
    ):
        super().__init__()

        self.weight_decay = weight_decay
        self.pass_through_sigmoid = pass_through_sigmoid

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
            register_regularization_hook(parameter, self.weight_decay)

        return rating

    def forward(self, user_ids, item_ids):
        rating = self.linear_forward(user_ids, item_ids)
        if self.pass_through_sigmoid:
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
        if self.pass_through_sigmoid:
            rating = self.sigmoid(rating)

        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        scale_down = self.user_weight.shape[0] / len(user_ids)
        register_regularization_hook(
            item_rating_effect_weight, self.weight_decay / scale_down
        )

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
        self,
        *,
        model_config,
        optimizer_config,
        loss=Literal["implicit_aware_sparse_loss", "personalized_ranking_loss"],
        confidence_in_rating_quality=0.9,
        batch_size=10e8,
        num_workers=1,
        train_path,
        val_path,
        memory_upper_bound=1e8,
    ):
        """
        Module based on matrix factorization models.
        :param model_config: config dict for the core model
        :param optimizer_config: config dict for optimizer
        :param loss: name of one of available loss functions
        :param confidence_in_rating_quality: probability that
        item with best rating is more relevant than unrated item
        :param batch_size: number of user-item pairs in a batch
        :param num_workers: number of workers for dataloaders
        :param train_path: path to .npz file of scipy sparse matrix containing
        train explicit feedback with rows corresponding to users and columns to items
        :param val_path: path to .npz file of validation explicit feedback
        :param memory_upper_bound: O(n) bound on size of tensor that can fit in the memory
        """
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = self.build_dataset(self.train_explicit(train_path))
        self.val_dataset = self.build_dataset(self.val_explicit(val_path))
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

    def loss(self, explicit, implicit, model_ratings):
        if self.hparams["loss"] == "implicit_aware_sparse_loss":
            return self.implicit_aware_sparse_loss(explicit, implicit, model_ratings)
        elif self.hparams["loss"] == "personalized_ranking_loss":
            return self.personalized_ranking_loss(explicit, implicit, model_ratings)
        else:
            raise ValueError(f"Unknown loss {self.hparams['loss']}")

    @staticmethod
    def implicit_aware_sparse_loss(explicit, implicit, model_ratings):
        """Loss = 1/|Explicit| * \\sum_{ij}Implicit_{ij} * (Explicit_{ij} - ModelRating_{ij})^2"""
        error = model_ratings - explicit
        error = sparse_dense_multiply(sparse_dense_multiply(implicit, error), error)
        loss = torch.sparse.sum(error) / error._values().numel()
        return loss

    def _personalized_ranking_loss(self, explicit, implicit, model_ratings):
        """
        Loss from Bayesian Personalized Ranking paper.
        Caution: this implementation is pretty slow.
        """
        implicit = implicit.coalesce()
        ids_of_users_who_rated_something = implicit._indices()[0].unique()
        from collections import Counter

        self.log("n_batch ratings", implicit._values().numel())
        self.log(
            "n_users who rated something", ids_of_users_who_rated_something.numel()
        )
        if ids_of_users_who_rated_something.numel() == 0:
            return None

        c = Counter(implicit._indices()[0].cpu().numpy())
        self.log("max item ratings", c.most_common(1)[0][1])
        self.log("batch n_users", implicit.size()[0])
        self.log("batch n_items", implicit.size()[1])

        n_users, n_items = implicit.size()
        item_ids = torch.arange(n_items)
        items_mask = torch.full((n_items,), True)
        criterion = 0
        for i, user_id in enumerate(ids_of_users_who_rated_something):
            implicit_user_feedback = implicit[user_id]
            liked_item_ids = implicit_user_feedback._indices().squeeze()
            items_mask[:] = True
            items_mask[liked_item_ids] = False
            uninteracted_item_ids = item_ids[items_mask]

            predicted_user_ratings = model_ratings[user_id]

            user_ranking_correctness_criterion = -torch.log1p(
                torch.exp(
                    -(
                        predicted_user_ratings[liked_item_ids, None]
                        - predicted_user_ratings[uninteracted_item_ids]
                    )
                )
            ).mean()
            criterion += user_ranking_correctness_criterion

        loss = -criterion / len(ids_of_users_who_rated_something)
        return loss

    @staticmethod
    def pairwise_difference(left, right):
        """
        Given two rating matrices A, B of shape (n_users, n_items),
        returns matrix C of shape (n_users, n_items, n_items) such that
        C[u, i, j] = A[u, i] - B[u, j]
        """
        return left[:, :, None] - right[:, None, :]

    def probability_of_user_preferences(
        self, ratings_of_supposedly_better_items, ratings_of_supposedly_worse_items
    ):
        """
        Given two rating matrices A, B of shape (n_users, n_items),
        returns matrix C of shape (n_users, n_items, n_items) such that
        C[u, i, j] = Probability({ user u prefers item A[u, i] over B[u, j] })
        """
        scaling_coefficient = self.hparams[
            "confidence_in_rating_quality"
        ] / torch.sigmoid(
            torch.tensor([1], device=ratings_of_supposedly_better_items.device)
        )
        return scaling_coefficient * torch.sigmoid(
            self.pairwise_difference(
                ratings_of_supposedly_better_items, ratings_of_supposedly_worse_items
            )
        )

    @staticmethod
    def kl_divergence(
        probability_of_user_preferences, predicted_probability_of_user_preferences
    ):
        return -probability_of_user_preferences * torch.log(
            predicted_probability_of_user_preferences
        )

    def dense_cutout_kl_divergence(
        self, explicit_dense_cutout, model_ratings_dense_cutout
    ):
        probability_of_user_preferences = self.probability_of_user_preferences(
            explicit_dense_cutout, explicit_dense_cutout
        )
        predicted_probability_of_user_preferences = (
            self.probability_of_user_preferences(
                model_ratings_dense_cutout, model_ratings_dense_cutout
            )
        )
        kl_divergence = self.kl_divergence(
            probability_of_user_preferences,
            predicted_probability_of_user_preferences,
        )
        return kl_divergence

    def rated_unrated_kl_divergence(
        self,
        explicit_dense_cutout,
        model_ratings_dense_cutout,
        unrated_items_model_ratings,
    ):
        probability_of_user_preferences = self.probability_of_user_preferences(
            explicit_dense_cutout, torch.zeros_like(unrated_items_model_ratings)
        )
        predicted_probability_of_user_preferences = (
            self.probability_of_user_preferences(
                model_ratings_dense_cutout, unrated_items_model_ratings
            )
        )
        kl_divergence = self.kl_divergence(
            probability_of_user_preferences, predicted_probability_of_user_preferences
        )
        complementary_kl_divergence = self.kl_divergence(
            1 - probability_of_user_preferences,
            1 - predicted_probability_of_user_preferences,
        )
        return kl_divergence + complementary_kl_divergence

    def get_dense_and_unrated_splits(self, explicit):
        users_who_rated_something = explicit._indices()[0].unique()
        rated_items, counts = explicit._indices()[1].unique(return_counts=True)

        n_users, n_items = explicit.size()
        discriminant = n_items**2 - 4 * self.hparams["memory_upper_bound"] / n_users
        if discriminant > 0:
            n_rated_items_upper_bound = (n_items - discriminant**0.5) // 2
            if len(rated_items) > n_rated_items_upper_bound:
                self.log(
                    "n_clipped_items", len(rated_items) - n_rated_items_upper_bound
                )
                rated_items = rated_items[
                    counts.argsort(descending=True)[:n_rated_items_upper_bound]
                ]

        item_ids = torch.arange(n_items)
        items_mask = torch.full((n_items,), True)
        items_mask[rated_items] = False
        unrated_items = item_ids[items_mask]

        return users_who_rated_something, rated_items, unrated_items

    def split_explicit_feedback_and_ratings_into_dense_and_unrated(
        self, explicit, model_ratings
    ):
        assert (
            explicit._values().min() >= 0 and explicit._values().max() <= 1
        ), "Explicit feedback is not normalized."

        (
            users_who_rated_something,
            rated_items,
            unrated_items,
        ) = self.get_dense_and_unrated_splits(explicit)

        explicit_dense_cutout = explicit.to_dense()[users_who_rated_something][
            :, rated_items
        ]
        model_ratings_dense_cutout = model_ratings[users_who_rated_something][
            :, rated_items
        ]
        unrated_items_model_ratings = model_ratings[users_who_rated_something][
            :, unrated_items
        ]
        return (
            explicit_dense_cutout,
            model_ratings_dense_cutout,
            unrated_items_model_ratings,
        )

    def personalized_ranking_loss(self, explicit, _implicit, model_ratings):
        """Ranking based loss inspired by Bayesian Personalized Ranking paper."""
        explicit = explicit.coalesce()
        if explicit._values().numel() == 0:
            return None

        (
            explicit_dense_cutout,
            model_ratings_dense_cutout,
            unrated_items_model_ratings,
        ) = self.split_explicit_feedback_and_ratings_into_dense_and_unrated(
            explicit, model_ratings
        )

        dense_cutout_kl_divergence = self.dense_cutout_kl_divergence(
            explicit_dense_cutout, model_ratings_dense_cutout
        )

        rated_unrated_kl_divergence = self.rated_unrated_kl_divergence(
            explicit_dense_cutout,
            model_ratings_dense_cutout,
            unrated_items_model_ratings,
        )

        unrated_items_in_dense_cutout = explicit_dense_cutout != 0
        dense_cutout_mask = (
            unrated_items_in_dense_cutout[:, :, None]
            | unrated_items_in_dense_cutout[:, None, :]
        )
        dense_cutout_mask[
            :, (arange := torch.arange(dense_cutout_mask.shape[-1])), arange
        ] = False
        rated_unrated_mask = einops.repeat(
            unrated_items_in_dense_cutout,
            f"user item -> user item {unrated_items_model_ratings.shape[1]}",
        )

        assert dense_cutout_mask.shape == dense_cutout_kl_divergence.shape
        assert rated_unrated_mask.shape == rated_unrated_kl_divergence.shape

        loss = (dense_cutout_mask * dense_cutout_kl_divergence).sum() + (
            rated_unrated_mask * rated_unrated_kl_divergence
        ).sum()

        loss /= dense_cutout_mask.sum() + rated_unrated_mask.sum()

        return loss

    def forward(self, **batch):
        return self.model(
            user_ids=batch.get("user_ids", slice(None)),
            item_ids=batch.get("item_ids", slice(None)),
        )

    def training_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit"], batch["implicit"], ratings)
        self.log("train_loss", 0 if loss is None else loss)
        return loss

    def validation_step(self, batch, batch_idx):
        ratings = self(**batch)
        loss = self.loss(batch["explicit"], batch["implicit"], ratings)
        self.log("val_loss", 0 if loss is None else loss)
        return loss


class PMFTrainer(ConfigConstructorBase):
    def lightning_module_candidates(self):
        return [LitProbabilityMatrixFactorization]

    def callback_class_candidates(self):
        return [ConvenientCheckpointLogCallback]

    def main(self):
        self.trainer.fit(self.lightning_module)


class PMFConfigDispenser(RecommendingConfigDispenser):
    def debug_config(self, config):
        config = super().debug_config(config)
        config["lightning_module"].update(dict(batch_size=1e6))
        config["trainer"].update(dict(accelerator="gpu", devices=[1], max_epochs=500))
        return config


@PMFConfigDispenser
def main(config):
    PMFTrainer(config).main()


if __name__ == "__main__":
    main()
