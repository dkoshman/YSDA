import abc
import functools
import os
from argparse import ArgumentParser
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
import wandb

from scipy.sparse import coo_matrix
from tqdm.auto import tqdm


if TYPE_CHECKING:
    from machine_learning.recommending.lit import LitRecommenderBase
    from my_tools.utils import SparseTensor


class MovieLensInterface(abc.ABC):
    def __init__(self, directory):
        self.directory = directory

    @property
    @abc.abstractmethod
    def shape(self) -> "tuple[int, int]":
        """Return [n_users, n_items]"""

    @property
    def ratings_columns(self):
        return "user_ids", "item_ids", "rating", "timestamp"

    @property
    @abc.abstractmethod
    @functools.lru_cache()
    def unique_movielens_user_ids(self) -> np.array:
        """Returns cached unique sorted movielens user ids that the model can encounter during training."""
        all_movielens_user_ids = ...
        return np.unique(all_movielens_user_ids)

    @property
    @abc.abstractmethod
    @functools.lru_cache()
    def unique_movielens_movie_ids(self) -> np.array:
        """Returns cached unique sorted movielens movie ids that the model can encounter during training."""
        all_movielens_movie_ids = ...
        return np.unique(all_movielens_movie_ids)

    def dense_to_movielens_user_ids(self, user_ids: np.array):
        return self.unique_movielens_user_ids[user_ids]

    def movielens_to_dense_user_ids(self, movielens_user_ids: np.array):
        movielens_to_model = pd.Series(
            index=self.unique_movielens_user_ids,
            data=np.arange(len(self.unique_movielens_user_ids)),
        )
        return movielens_to_model.loc[movielens_user_ids].values

    def dense_item_to_movielens_movie_ids(self, item_ids: np.array):
        return self.unique_movielens_movie_ids[item_ids]

    def movielens_movie_to_dense_item_ids(self, movielens_movie_ids: np.array):
        movielens_to_model = pd.Series(
            index=self.unique_movielens_movie_ids,
            data=np.arange(len(self.unique_movielens_movie_ids)),
        )
        return movielens_to_model.loc[movielens_movie_ids].values

    @abc.abstractmethod
    @functools.lru_cache()
    def __getitem__(self, filename):
        return self.read(filename=filename)

    def read(self, filename, **kwargs):
        path = os.path.join(self.directory, filename)
        dataframe = pd.read_csv(path, **kwargs)
        dataframe = dataframe.rename(
            {"user_id": "user_ids", "item_id": "item_ids"}, axis="columns"
        )
        return dataframe.squeeze()

    def explicit_feedback_scipy_csr(self, name):
        return self.ratings_dataframe_to_scipy_csr(self[name])

    def ratings_dataframe_to_scipy_csr(self, ratings_dataframe):
        movielens_ratings = ratings_dataframe["rating"].to_numpy()

        movielens_user_ids = ratings_dataframe["user_ids"].to_numpy()
        user_ids = self.movielens_to_dense_user_ids(movielens_user_ids)

        movielens_movie_ids = ratings_dataframe["item_ids"].to_numpy()
        item_ids = self.movielens_movie_to_dense_item_ids(movielens_movie_ids)

        data = movielens_ratings
        row_ids = user_ids
        col_ids = item_ids
        explicit_feedback = coo_matrix((data, (row_ids, col_ids)), shape=self.shape)
        return explicit_feedback.tocsr()


class MovieLens100k(MovieLensInterface):
    def __init__(self, directory="local/ml-100k"):
        super().__init__(directory)

    def __getitem__(self, filename):
        if filename == "u.info":
            return self.read(
                filename=filename,
                names=["quantity", "index"],
                index_col="index",
                sep=" ",
            )
        elif filename == "u.genre":
            return self.read(filename=filename, names=["name", "id"], index_col="id")
        elif filename == "u.occupation":
            return self.read(filename=filename, names=["occupation"])
        elif filename == "u.user":
            return self.read(
                filename=filename,
                names=["user id", "age", "gender", "occupation", "zip code"],
                index_col="user id",
            )
        elif filename == "u.item":
            return self.read(
                filename=filename,
                names=[
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
                index_col="movie id",
                encoding_errors="backslashreplace",
            )
        if filename in ["u.data"] + [
            i + j
            for i in ["u1", "u2", "u3", "u4", "u5", "ua", "ub"]
            for j in [".base", ".test"]
        ]:
            return self.read(
                filename=filename,
                names=self.ratings_columns,
                sep="\t",
                dtype="int32",
            )
        raise FileNotFoundError(f"File {filename} not found.")

    def read(self, filename, sep="|", header=None, **kwargs):
        return super().read(filename=filename, sep=sep, header=header, **kwargs)

    @property
    @functools.lru_cache()
    def shape(self):
        info = self["u.info"]
        return info["users"], info["items"]

    @property
    @functools.lru_cache()
    def unique_movielens_user_ids(self) -> np.array:
        return np.unique(self["u.user"].index)

    @property
    @functools.lru_cache()
    def unique_movielens_movie_ids(self):
        return np.unique(self["u.item"].index)


class MovieLens25m(MovieLensInterface):
    def __init__(self, directory="local/ml-25m"):
        super().__init__(directory)

    @functools.lru_cache(maxsize=20)
    def __getitem__(self, filename):
        if not filename.endswith(".csv"):
            filename += ".csv"
        if filename == "ratings.csv":
            return self.read(
                filename=filename,
                names=self.ratings_columns,
                header=0,
                dtype={
                    c: (np.int32 if c != "rating" else np.float32)
                    for c in self.ratings_columns
                },
            )
        elif filename == "links.csv":
            return self.read(filename=filename).rename(
                dict(movieId="movielensId"), axis="columns"
            )
        elif filename == "movies.csv":
            return self.read(filename=filename, index_col="movieId")
        else:
            return self.read(filename=filename)

    @property
    @functools.lru_cache()
    def shape(self):
        ratings = self["ratings"]
        n_users = ratings["user_ids"].nunique()
        n_items = ratings["item_ids"].nunique()
        return n_users, n_items

    @property
    @functools.lru_cache()
    def unique_movielens_user_ids(self) -> np.array:
        return np.unique(self["ratings"]["user_ids"])

    @property
    @functools.lru_cache()
    def unique_movielens_movie_ids(self):
        return np.unique(self["ratings"]["item_ids"])

    @property
    @functools.lru_cache()
    def unique_imdb_ids(self) -> np.array:
        return np.unique(self["links"]["imdbId"])

    def movielens_movie_to_imdb_movie_ids(self, movielens_movie_ids: np.array):
        movielens_to_imdb = self["links"].set_index("movielensId")["imdbId"]
        return movielens_to_imdb.loc[movielens_movie_ids].values

    def imdb_movie_to_movielens_movie_id(self, imdb_movie_ids: np.array):
        imdb_to_movielens = self["links"].set_index("imdbId")["movielensId"]
        return imdb_to_movielens.loc[imdb_movie_ids].values

    def imdb_movie_to_dense_item_ids(self, imdb_movie_ids: np.array):
        return self.movielens_movie_to_dense_item_ids(
            self.imdb_movie_to_movielens_movie_id(imdb_movie_ids)
        )

    def dense_item_to_imdb_movie_ids(self, item_ids: np.array):
        return self.movielens_movie_to_imdb_movie_ids(
            self.dense_item_to_movielens_movie_ids(item_ids)
        )

    def quantiles_split(self, quantiles: "list[float]") -> "list[pd.DataFrame]":
        if sum(quantiles) != 1:
            raise ValueError(f"Quantiles must sum up to one, received {quantiles}.")
        ratings = self["ratings"]
        quantile_thresholds = np.cumsum(quantiles)
        thresholds = ratings["timestamp"].quantile(quantile_thresholds)
        thresholds = [ratings["timestamp"].min() - 1] + list(thresholds)
        splits = []
        for begin, end in zip(thresholds, thresholds[1:]):
            splits.append(ratings.query(f"{begin} < timestamp <= {end}"))
        return splits

    def abs_path(self, filename):
        return os.path.join(os.getcwd(), self.directory, filename)

    def save_ratings_split(self, ratings: pd.DataFrame, filename) -> str:
        path = self.abs_path(filename=filename)
        ratings.to_csv(path, index=False)
        return path


class MovieLensMixin:
    @property
    def movielens(self: "LitRecommenderBase"):
        return self.build_class(**self.hparams["datamodule"]["movielens"])

    @property
    def class_candidates(self):
        return super().class_candidates + [MovieLens100k, MovieLens25m]

    def common_explicit(self: "LitRecommenderBase", filename):
        return self.movielens.explicit_feedback_scipy_csr(
            self.hparams["datamodule"][filename]
        )

    def train_explicit(self):
        return self.common_explicit(filename="train_explicit_file")

    def val_explicit(self):
        return self.common_explicit(filename="val_explicit_file")

    def test_explicit(self):
        return self.common_explicit(filename="test_explicit_file")


def read_csv_imdb_ratings(
    path_to_imdb_ratings_csv: str, ratings_scale_max_to_convert_to: float = 5
) -> pd.DataFrame:
    dataframe = pd.read_csv(path_to_imdb_ratings_csv)
    dataframe = dataframe.rename(lambda c: c.lower().replace(" ", "_"), axis="columns")
    dataframe = dataframe.rename({"const": "imdb_id"}, axis="columns")
    dataframe["imdb_id"] = dataframe["imdb_id"].str.removeprefix("tt").astype(int)
    dataframe["your_rating"] *= ratings_scale_max_to_convert_to / 10
    assert list(dataframe) == [
        "imdb_id",
        "your_rating",
        "date_rated",
        "title",
        "url",
        "title_type",
        "imdb_rating",
        "runtime_(mins)",
        "year",
        "genres",
        "num_votes",
        "release_date",
        "directors",
    ]
    return dataframe


def explicit_from_imdb_ratings(
    imdb_ratings: pd.DataFrame,
    movielens_the_model_trained_on: MovieLensInterface,
    movielens_25m: MovieLens25m,
) -> "SparseTensor":
    movielens = movielens_the_model_trained_on
    imdb_ids = imdb_ratings["imdb_id"].values
    imdb_ids = np.intersect1d(movielens_25m.unique_imdb_ids, imdb_ids)
    movielens_ids = movielens_25m.imdb_movie_to_movielens_movie_id(imdb_ids)
    assert all(
        imdb_ids == movielens_25m.movielens_movie_to_imdb_movie_ids(movielens_ids)
    )
    movielens_ids = np.intersect1d(movielens.unique_movielens_movie_ids, movielens_ids)

    item_ids = movielens.movielens_movie_to_dense_item_ids(movielens_ids)
    assert all(movielens_ids == movielens.dense_item_to_movielens_movie_ids(item_ids))
    imdb_ids = movielens_25m.movielens_movie_to_imdb_movie_ids(movielens_ids)
    assert all(
        movielens_ids == movielens_25m.imdb_movie_to_movielens_movie_id(imdb_ids)
    )
    ratings = imdb_ratings.set_index("imdb_id").loc[imdb_ids]["your_rating"].values

    n_items = movielens.shape[1]
    explicit = np.zeros([1, n_items])
    explicit[0, item_ids] = ratings
    return torch.from_numpy(explicit).to_sparse()


def main():
    parser = ArgumentParser(
        description="Split and save movielens 25m dataset by timestamps in provided quantiles."
    )
    parser.add_argument("directory", default="local/ml-25m", type=str, nargs="?")
    parser.add_argument("--quantiles", "-q", type=float, nargs="+")
    parser.add_argument("--names", "-n", type=str, nargs="*", help="names for splits")
    args = parser.parse_args()
    if len(args.quantiles) < 2:
        raise ValueError(
            f"Number of quantiles to split into must be more than 1, received {args.quantiles}."
        )
    elif args.names is not None and len(args.names) != len(args.quantiles):
        raise ValueError(
            f"If provided, number of names for splits must match number of quantiles, "
            f"received n_quantiles: {len(args.quantiles)}, n_names: {len(args.names)}."
        )

    with wandb.init(
        entity="dkoshman", project="Recommending", job_type="data", config=vars(args)
    ):
        movielens = MovieLens25m(directory=args.directory)
        metadata = dict(
            directory=args.directory,
            quantiles=args.quantiles,
            local_abs_paths={
                i: movielens.abs_path(i)
                for i in ["tags.csv", "links.csv", "movies.csv"]
            },
        )
        splits = movielens.quantiles_split(quantiles=args.quantiles)
        names = args.names or [
            f"ratings_split{i}_q{q}" for i, q in enumerate(args.quantiles)
        ]

        for dataframe, name in (tqdm_progress_bar := tqdm(zip(splits, names))):
            tqdm_progress_bar.set_description(f"Preparing split {name}")
            df_describe = dataframe.describe().reset_index()
            wandb.log({f"{name}_describe": wandb.Table(dataframe=df_describe)})
            for kind in ["user", "item"]:
                dataframe = (
                    dataframe.groupby(f"{kind}_ids")
                    .size()
                    .describe()
                    .rename(f"{kind}_groupby_size")
                    .reset_index()
                )
                wandb.log({f"{name}_{kind}_describe": wandb.Table(dataframe=dataframe)})

            filename = f"{name}.csv"
            path = movielens.save_ratings_split(ratings=dataframe, filename=filename)
            metadata["local_abs_paths"][filename] = path

        artifact = wandb.Artifact(
            name="movielens25m",
            type="data",
            description=f"Time split by quantiles {args.quantiles}",
            metadata=metadata,
        )
        for local_abs_path in metadata["local_abs_paths"].values():
            artifact.add_file(local_path=local_abs_path)

        wandb.log_artifact(artifact, name=filename, type="data")


if __name__ == "__main__":
    main()
