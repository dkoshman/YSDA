import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix

from machine_learning.recommending.data import (
    build_recommending_dataloader,
    SparseDataset,
    SparseDataModuleBase,
)
from machine_learning.recommending.movielens.data import MovieLens100k, ImdbRatings
from machine_learning.recommending.tests.conftest import (
    random_explicit_feedback,
    cartesian_products_of_dict_values,
)


def test_build_recommending_dataloader():
    explicit = random_explicit_feedback()
    dataset = SparseDataset(explicit)
    grid_kwargs = dict(
        sampler_type=["grid", "user", "item"],
        batch_size=[100],
        num_workers=[0, 4],
        persistent_workers=[True, False],
        shuffle=[True, False],
    )
    for kwargs in cartesian_products_of_dict_values(grid_kwargs):
        if kwargs["persistent_workers"] and kwargs["num_workers"] == 0:
            continue
        dataloader = build_recommending_dataloader(dataset=dataset, **kwargs)
        for batch in dataloader:
            batch = SparseDataset.unpack_sparse_kwargs_to_torch_sparse_csr(batch)
            assert batch["explicit"].is_sparse_csr
            assert batch["implicit"].is_sparse_csr


class MockSparseDataModuleBase(SparseDataModuleBase):
    def train_explicit(self):
        return random_explicit_feedback()

    def val_explicit(self):
        return random_explicit_feedback()

    def test_explicit(self):
        return random_explicit_feedback()


def test_sparse_datamodule_base():
    datamodule = MockSparseDataModuleBase()
    for dataloader in [
        datamodule.train_dataloader(),
        datamodule.val_dataloader(),
        datamodule.test_dataloader(),
    ]:
        for batch in dataloader:
            batch = datamodule.on_after_batch_transfer(batch)
            assert batch["explicit"].is_sparse_csr
            assert batch["implicit"].is_sparse_csr


def test_movielens():
    movielens = MovieLens100k(path_to_movielens_folder="local/ml-100k")
    n_users, n_items = movielens.shape
    assert np.isscalar(n_users)
    assert np.isscalar(n_items)
    explicit_filenames = ["u.data"] + [
        i + j
        for i in ["u1", "u2", "u3", "u4", "u5", "ua", "ub"]
        for j in [".base", ".test"]
    ]
    filenames = [
        "u.info",
        "u.genre",
        "u.occupation",
        "u.user",
        "u.item",
    ] + explicit_filenames
    for filename in filenames:
        frame_or_series = movielens[filename]
        assert isinstance(frame_or_series, pd.DataFrame) or isinstance(
            frame_or_series, pd.Series
        )
    for filename in explicit_filenames:
        explicit = movielens.explicit_feedback_scipy_csr(filename)
        assert isinstance(explicit, csr_matrix)


def test_imdb_ratings():
    imdb_ratings = ImdbRatings()
    ratings = imdb_ratings.imdb_ratings
    assert isinstance(ratings, pd.DataFrame)
    explicit = imdb_ratings.explicit_feedback_scipy()
    assert isinstance(explicit, coo_matrix)
    item_ids = np.arange(imdb_ratings.movielens.shape[1])
    items_description = imdb_ratings.items_description(item_ids)
    assert isinstance(items_description, pd.DataFrame)
