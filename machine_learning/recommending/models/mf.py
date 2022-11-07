import torch
from torch.utils.data import DataLoader

from my_tools.models import register_regularization_hook
from my_tools.utils import to_torch_coo, torch_sparse_slice

from ..lit import LitRecommenderBase
from ..data import SparseDataset, build_recommending_dataloader, GridIterableDataset
from ..interface import RecommenderModuleBase, RecommendingLossInterface
from ..utils import build_bias, build_weight, batch_size_in_bytes, wandb_timeit


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
        super().__init__(persistent_explicit=False, **kwargs)
        self.encoder = build_weight(self.n_items, latent_dimension)
        self.decoder = build_weight(latent_dimension, self.n_items)
        self.item_bias = build_bias(self.n_items)
        self.bias = build_bias(1)
        self.l2_regularization = l2_regularization
        self.l1_regularization = l1_regularization

    def forward(self, user_ids, item_ids):
        users_explicit = torch_sparse_slice(self.explicit, row_ids=user_ids)
        ratings = self.online_ratings(users_explicit=users_explicit)
        return ratings[:, item_ids]

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


class MFSlimConfidenceRecommender(RecommenderModuleBase):
    """
    Recommender based on the following data generation model:
    1. Each user-item pair has a true, unknown, rating
    generated from normal distribution N_{r_ui, var}.
    2. Each rating has a probability to be observed p_ui = f(r_ui),
    where f is a non-decreasing function, f(0) = 0.
    That way higher quality items are more likely to be observed.

    Then the likelihood of given data sample is:
    likelihood(sample) =
    prod_observed_ui(p_ui * N_{r_ui, var}(rating_ui))) * prod_unobserved_ui(1 - p_ui)

    argmax(likelihood(sample)) =
    argmax(prod_observed_ui(f(r_ui) * N_{r_ui, var}(rating_ui))) * prod_unobserved_ui(1 - f(r_ui))) =
    argmax(sum_observed_ui(log f(r_ui) - (r_ui - rating_ui) ** 2 / (2 * var)) + sum_unobserved_ui(log(1 - f(r_ui))))

    Another change is in the way I recommend items to users:
    I want to recommend items which the users haven't previously seen,
    so given probability p_ui that user has seen item, I want to scale
    down the expected rating r_ui with that probability:
    predicted_relevance_ui = r_ui * g(1 - p_ui),
    where g is a non-decreasing function.

    And p_ui can be factorized as p_u * p_i, and p_u and p_i
    can be estimated as their sample frequencies.

    The default for g is g(x) = x, and for f:
    f(x) = sigmoid(x / sigmoid(a) + b), where a, b are parameters.
    """

    def __init__(
        self,
        latent_dimension=10,
        l2_regularization=0.0,
        l1_regularization=0.0,
        **kwargs,
    ):
        super().__init__(persistent_explicit=False, **kwargs)
        self.ratings_mfs = MFSlimRecommender(
            latent_dimension=latent_dimension,
            l2_regularization=l2_regularization,
            l1_regularization=l1_regularization,
            **kwargs,
        )
        self.register_buffer(name="user_activity", tensor=torch.zeros(self.n_users))
        self.register_buffer(name="item_popularity", tensor=torch.zeros(self.n_items))
        if self.explicit is not None:
            self.user_activity = torch.tensor(
                self.to_scipy_coo(self.explicit).mean(1).A.squeeze()
            )
            self.item_popularity = torch.tensor(
                self.to_scipy_coo(self.explicit).mean(0).A.squeeze()
            )

    @staticmethod
    def probability_that_user_seen_item(user_activity, item_popularity):
        return torch.einsum("u, i -> ui", user_activity, item_popularity)

    def forward(self, user_ids, item_ids, predicting=True):
        ratings = self.ratings_mfs(user_ids=user_ids, item_ids=item_ids)
        if predicting:
            ratings *= 1 - self.probability_that_user_seen_item(
                user_activity=self.user_activity[user_ids],
                item_popularity=self.item_popularity[item_ids],
            )
        return ratings

    def online_ratings(self, users_explicit, estimated_users_activity=None):
        users_explicit = users_explicit.to(self.device, torch.float32)
        if estimated_users_activity is None:
            torch.tensor(self.to_scipy_coo(users_explicit).mean(1).A.squeeze())
        ratings = self.ratings_mfs.online_ratings(users_explicit=users_explicit)
        ratings *= 1 - self.probability_that_user_seen_item(
            user_activity=estimated_users_activity,
            item_popularity=self.item_popularity,
        )
        return ratings

    def online_recommend(
        self, users_explicit, n_recommendations: int = 10, estimated_users_activity=None
    ) -> torch.IntTensor:
        users_explicit = self.to_torch_coo(users_explicit).to(self.device)
        ratings = self.online_ratings(
            users_explicit, estimated_users_activity=estimated_users_activity
        )
        ratings = self.filter_already_liked_items(users_explicit, ratings)
        recommendations = self.ratings_to_recommendations(ratings, n_recommendations)
        return recommendations


class MFSlimConfidenceLoss(RecommendingLossInterface):
    def __init__(self, ratings_deviation=1, l1_coefficient=1, mean_unobserved_rating=0):
        self.ratings_variance = ratings_deviation**2
        self.l1_coefficient = l1_coefficient
        self.mean_unobserved_rating = mean_unobserved_rating

    def __call__(self, model, explicit, user_ids, item_ids):
        explicit = explicit.to_dense()
        observed_mask = explicit > 0
        ratings = model(user_ids=user_ids, item_ids=item_ids, predicting=False)
        loss = observed_mask * (ratings - explicit) ** 2 / (2 * self.ratings_variance)
        loss += (
            ~observed_mask
            * (ratings - self.mean_unobserved_rating).abs()
            * self.l1_coefficient
        )
        return loss.sum()


class MFRecommender(LitRecommenderBase):
    @property
    def class_candidates(self):
        return super().class_candidates + [
            MatrixFactorization,
            ConstrainedProbabilityMatrixFactorization,
            MFSlimRecommender,
            MFSlimConfidenceRecommender,
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

    @wandb_timeit("training_step")
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

    @wandb_timeit("validation_step")
    def validation_step(self, batch, batch_idx):
        self.log("val_batch_size_in_bytes", float(batch_size_in_bytes(batch)))
        loss = self.loss(model=self.model, **batch)
        self.log("val_loss", loss)
        return loss
