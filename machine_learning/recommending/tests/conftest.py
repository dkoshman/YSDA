import abc
from typing import Literal, Iterable

import numpy as np
import pytorch_lightning as pl
import scipy
import torch

from torch.utils.data import Dataset, DataLoader

from machine_learning.recommending.data import (
    build_recommending_dataloader,
    SparseTensorUnpacker,
    SparseDataset,
)


TEST_CONFIG_PATH = "tests/config_for_testing.yaml"


def seed_everything(seed=None):
    import os
    import sys
    import time

    seed_file = "_latest_seed.txt"
    environment_variable = "_SEEDED_EVERYTHING"

    if os.environ.get(environment_variable) == "True":
        return

    os.environ[environment_variable] = "True"
    if seed is None:
        seed = int(time.time())
    print(
        f"Seeding everything with seed {seed}. Saving seed to {seed_file}",
        file=sys.stderr,
    )
    print(seed, file=open(seed_file, "w"))
    pl.seed_everything(seed)


def random_explicit_feedback(
    n_users=None, n_items=None, density=None, max_rating=None, min_n_ratings_per_user=2
):
    n_users = n_users or np.random.randint(10, 100)
    n_items = n_items or np.random.randint(10, 100)
    density = density or np.random.uniform(0.1, 0.5)
    max_rating = max_rating or np.random.choice(range(1, 11))

    ratings_range = np.arange(max_rating + 1)
    probs = [1 - density] + [density / max_rating] * max_rating

    for _ in range(100):
        explicit = np.random.choice(
            ratings_range, p=probs, size=[n_users * 10, n_items]
        )
        n_ratings_per_user = (explicit > 0).sum(1)
        explicit = explicit[n_ratings_per_user >= min_n_ratings_per_user]
        if explicit.shape[0] >= n_users:
            explicit = explicit[:n_users]
            break
    else:
        raise ValueError(
            f"""
            Couldn't generate explicit data compliant with restrictions:
            n_users {n_users}
            n_items {n_items}
            density {density}
            max_rating {max_rating}
            min_n_ratings_per_user {min_n_ratings_per_user}
            """
        )
    return scipy.sparse.csr_matrix(explicit)


def is_cuda_error(exception):
    exception_message = str(exception).lower()
    print("\nError" + "-" * 100, exception_message, "\n")
    return (
        isinstance(exception, MemoryError)
        or exception_message.startswith("cublas error")
        or exception_message.startswith("curand error")
    )


def cartesian_products_of_dict_values(dictionary: dict[str, Iterable]):
    if not dictionary:
        yield {}
        return
    dictionary = dictionary.copy()
    key = list(dictionary)[0]
    values = dictionary.pop(key)
    for value in values:
        for d in cartesian_products_of_dict_values(dictionary):
            yield {key: value, **d}


class MockLinearDataset(Dataset):
    def __init__(self, n_samples, n_features, true_parameter):
        self.n_samples = n_samples
        self.n_features = n_features
        self.features = torch.randn(n_samples, n_features)
        noise = torch.randn(n_samples)
        self.target = self.features @ true_parameter + noise

    def __len__(self):
        return self.n_samples

    def __getitem__(self, indices):
        return dict(
            features=self.features[indices],
            target=self.target[indices],
            indices=indices,
        )


class MockLightningModuleInterface(pl.LightningModule):
    @abc.abstractmethod
    def dataloader(self, stage: Literal["train", "val", "test", "predict"]):
        ...

    def train_dataloader(self):
        return self.dataloader(stage="train")

    def val_dataloader(self):
        return self.dataloader(stage="val")

    def test_dataloader(self):
        return self.dataloader(stage="test")

    def predict_dataloader(self):
        return self.dataloader(stage="predict")

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @staticmethod
    def loss(target, prediction):
        return ((target - prediction) ** 2).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters())
        return [optimizer]

    @abc.abstractmethod
    def step(
        self, batch, stage: Literal["train", "val", "test", "predict"]
    ) -> torch.Tensor:
        ...

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.step(batch, stage="predict")


class MockLinearLightningModule(MockLightningModuleInterface):
    def __init__(
        self,
        n_features=10,
        batch_size=100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer(name="true_parameter", tensor=torch.randn(n_features))
        self.parameter = torch.nn.Parameter(torch.randn(n_features))

    def dataloader(self, stage):
        dataset = MockLinearDataset(
            n_samples=np.random.randint(1, 100),
            n_features=self.hparams["n_features"],
            true_parameter=self.true_parameter,
        )
        dataloader = DataLoader(dataset=dataset, batch_size=self.hparams["batch_size"])
        return dataloader

    def forward(self, features):
        return features @ self.parameter

    def step(self, batch, stage):
        prediction = self(batch["features"])
        loss = self.loss(target=batch["target"], prediction=prediction)
        return loss


class MockLitRecommender(SparseTensorUnpacker, MockLightningModuleInterface):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def dataloader(self, stage):
        if stage == "train":
            explicit = self.model.explicit
            shuffle = True
        else:
            explicit = random_explicit_feedback(
                n_users=self.model.n_users, n_items=self.model.n_items
            )
            shuffle = False
        return build_recommending_dataloader(
            dataset=SparseDataset(explicit=explicit),
            sampler_type="user",
            batch_size=np.random.randint(10, 100),
            num_workers=np.random.randint(0, 4),
            shuffle=shuffle,
        )

    def forward(self, **batch):
        return self.model(user_ids=batch["user_ids"], item_ids=batch["item_ids"])

    def step(self, batch, stage):
        relevance = self(**batch)
        loss = self.loss(target=batch["explicit"].to_dense(), prediction=relevance)
        return loss


def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        torch.cuda.init()
        devices.append("cuda")
    return devices
