import torch
from torch.utils.data import DataLoader

from my_tools.models import register_regularization_hook
from my_tools.utils import to_torch_coo, torch_sparse_slice

from ..lit import LitRecommenderBase
from ..data import SparseDataset, build_recommending_dataloader, GridIterableDataset
from ..interface import RecommenderModuleBase
from ..utils import build_bias, build_weight, batch_size_in_bytes, Timer


class MatrixFactorization(RecommenderModuleBase):
    """Predicted_rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(self, latent_dimension=10, weight_decay=1.0e-3, **kwargs):
        super().__init__(**kwargs)
        self.weight_decay = weight_decay
        self.user_weight = build_weight(self.n_users, latent_dimension)
        self.user_bias = build_bias(self.n_users, 1)
        self.item_weight = build_weight(self.n_items, latent_dimension)
        self.item_bias = build_bias(self.n_items, 1)
        self.bias = build_bias(1)

    def forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]
        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias

        # Need to add regularization here because otherwise optimizer will decay all weights,
        # not only those corresponding to user and item ids. Also, it is important to add gradient
        # hooks after the forward calculations, otherwise decay messes with the model.
        for parameter in [user_weight, item_weight, user_bias, item_bias]:
            register_regularization_hook(parameter, self.weight_decay)

        return rating


class ConstrainedProbabilityMatrixFactorization(MatrixFactorization):
    def __init__(self, *args, latent_dimension=10, **kwargs):
        super().__init__(*args, latent_dimension=latent_dimension, **kwargs)
        self.item_rating_effect_weight = build_weight(self.n_items, latent_dimension)
        self.implicit_feedback_normalized = None

    def init_implicit_feedback_normalized(self):
        implicit_feedback = self.to_scipy_coo(self.explicit) > 0
        implicit_feedback_normalized = implicit_feedback.multiply(
            1 / (implicit_feedback.sum(axis=1) + 1e-8)
        ).astype("float32")
        self.implicit_feedback_normalized = to_torch_coo(implicit_feedback_normalized)

    def forward(self, user_ids, item_ids):
        if self.implicit_feedback_normalized is None:
            self.init_implicit_feedback_normalized()

        # Need to clone to avoid gradient hook accumulation on same tensor and subsequent memory leak
        item_rating_effect_weight = self.item_rating_effect_weight.clone()

        item_weight = self.item_weight[item_ids]

        users_implicit_feedback = torch_sparse_slice(
            self.implicit_feedback_normalized, row_ids=user_ids
        ).to(self.bias.device)
        user_weights_offset_caused_by_their_ratings = (
            users_implicit_feedback @ item_rating_effect_weight
        )

        ratings = super().forward(user_ids=user_ids, item_ids=item_ids)
        ratings += user_weights_offset_caused_by_their_ratings @ item_weight.T

        # Scale down regularization because item_rating_effect_weight is decayed
        # for each batch, whereas other parameters have only their slices decayed.
        scale_down = self.user_weight.shape[0] / len(user_ids)
        register_regularization_hook(
            item_rating_effect_weight, self.weight_decay / scale_down
        )
        return ratings


class MFSlimRecommender(RecommenderModuleBase):
    """
    y = x @ w, w = (1 - I) * (encoder @ decoder)

    Zeroing out the diagonal to prevent model from fitting to
    the identity matrix and enforcing the model to predict
    item's relevance based on other items. Slim-like models
    are nice because they work online, are interpretable,
    simple and scalable.
    """

    def __init__(
        self,
        latent_dimension=10,
        l2_regularization=0.0,
        l1_regularization=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = build_weight(self.n_items, latent_dimension)
        self.decoder = build_weight(latent_dimension, self.n_items)
        self.item_bias = build_bias(self.n_items)
        self.bias = build_bias(1)
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization

    @Timer()
    def online_ratings(self, users_explicit):
        users_explicit = users_explicit.to(self.device, torch.float32)
        encoder = self.encoder.clone()
        decoder = self.decoder.clone()
        diag = torch.einsum("id, di -> i", encoder, decoder)
        ratings = (
            users_explicit @ encoder @ decoder
            - users_explicit.to_dense() * diag
            + self.item_bias
            + self.bias
        )
        if self.l2_regularization > 0 or self.l1_regularization > 0:
            for parameter in [encoder, decoder]:
                register_regularization_hook(
                    tensor=parameter,
                    l2_coefficient=self.l2_regularization,
                    l1_coefficient=self.l1_regularization,
                )
        return ratings


class MFRecommender(LitRecommenderBase):
    @property
    def class_candidates(self):
        return super().class_candidates + [
            MatrixFactorization,
            ConstrainedProbabilityMatrixFactorization,
            MFSlimRecommender,
        ]

    def train_dataloader(self):
        config = self.hparams["datamodule"]
        n_items = self.hparams["n_items"]
        n_users = self.hparams["n_users"]
        batch_size = config.get("batch_size", 100)
        grid_batch_size = int(batch_size**2 * n_items / n_users)
        dataset = GridIterableDataset(
            dataset_shape=(n_users, n_items),
            approximate_batch_size=grid_batch_size,
            shuffle=True,
        )
        num_workers = config.get("num_workers", 0)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=None,
            num_workers=num_workers,
            persistent_workers=config.get("persistent_workers", False),
            pin_memory=isinstance(num_workers, int) and num_workers > 1,
        )
        return dataloader

    def val_dataloader(self):
        config = self.hparams["datamodule"]
        batch_size = config.get(
            "val_batch_size", self.hparams["datamodule"].get("batch_size", 100)
        )
        return build_recommending_dataloader(
            dataset=SparseDataset(self.val_explicit()),
            sampler_type="user",
            batch_size=batch_size,
            num_workers=config.get("num_workers", 0),
            persistent_workers=config.get("persistent_workers", False),
        )

    def test_dataloader(self):
        config = self.hparams["datamodule"]
        batch_size = config.get(
            "test_batch_size", self.hparams["datamodule"].get("batch_size", 100)
        )
        return build_recommending_dataloader(
            dataset=SparseDataset(self.test_explicit()),
            sampler_type="user",
            batch_size=batch_size,
            num_workers=config.get("num_workers", 0),
            persistent_workers=config.get("persistent_workers", False),
        )

    def training_step(self, batch, batch_idx):
        self.log("train_batch_size_in_bytes", float(batch_size_in_bytes(batch)))
        explicit = torch_sparse_slice(
            sparse_matrix=self.model.explicit,
            row_ids=batch["user_ids"],
            col_ids=batch["item_ids"],
        ).to(self.device)
        loss = self.loss(model=self.model, explicit=explicit, **batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log("val_batch_size_in_bytes", float(batch_size_in_bytes(batch)))
        loss = self.loss(model=self.model, **batch)
        self.log("val_loss", loss)
        return loss
