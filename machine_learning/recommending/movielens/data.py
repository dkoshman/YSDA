import abc
import functools
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import wandb

from scipy.sparse import coo_matrix


class MovieLensBase:
    def __init__(self, path_to_movielens_folder):
        self.path_to_movielens_folder = path_to_movielens_folder

    @property
    @abc.abstractmethod
    @functools.lru_cache()
    def shape(self) -> "tuple[int, int]":
        """Return [n_users, n_items]"""

    @property
    def ratings_columns(self):
        return "user_id", "item_id", "rating", "timestamp"

    @abc.abstractmethod
    @functools.lru_cache()
    def __getitem__(self, filename):
        return self.read(filename=filename)

    def read(self, filename, **kwargs):
        path = os.path.join(self.path_to_movielens_folder, filename)
        dataframe = pd.read_csv(path, **kwargs)
        return dataframe.squeeze()

    def explicit_feedback_scipy_csr(self, name):
        return self.ratings_dataframe_to_scipy_csr(self[name])

    def ratings_dataframe_to_scipy_csr(self, ratings_dataframe):
        data = ratings_dataframe["rating"].to_numpy()
        # Stored ids start with 1.
        row_ids = ratings_dataframe["user_id"].to_numpy() - 1
        col_ids = ratings_dataframe["item_id"].to_numpy() - 1
        explicit_feedback = coo_matrix((data, (row_ids, col_ids)), shape=self.shape)
        return explicit_feedback.tocsr()


class MovieLens100k(MovieLensBase):
    def __init__(self, path_to_movielens_folder="local/ml-100k"):
        super().__init__(path_to_movielens_folder)

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


class MovieLens25m(MovieLensBase):
    def __init__(self, path_to_movielens_folder="local/ml-25m"):
        super().__init__(path_to_movielens_folder)

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
        else:
            return self.read(filename=filename)

    @property
    @functools.lru_cache()
    def shape(self):
        ratings = self["ratings"]
        n_users = ratings["user_id"].nunique()
        n_items = ratings["item_id"].nunique()
        return n_users, n_items

    @property
    @functools.lru_cache()
    def unique_movielens_movie_ids(self):
        return self["ratings"]["item_id"].unique()

    def train_test_split(self, train_quantile: float):
        ratings = self["ratings"]
        threshold = ratings["timestamp"].quantile(train_quantile)
        train_ratings = ratings.query(f"timestamp <= {threshold}")
        test_ratings = ratings.query(f"timestamp > {threshold}")
        return train_ratings, test_ratings

    def abs_path(self, filename):
        return os.path.join(os.getcwd(), self.path_to_movielens_folder, filename)

    def save_ratings_split(self, ratings: pd.DataFrame, filename) -> str:
        path = self.abs_path(filename=filename)
        ratings.to_csv(path, index=False)
        return path

    def model_item_to_movielens_movie_ids(self, item_ids: np.array):
        return self.unique_movielens_movie_ids[item_ids]

    def movielens_movie_to_model_item_ids(self, movielens_movie_ids: np.array):
        movielens_to_model = pd.Series(
            index=self.unique_movielens_movie_ids,
            data=np.arange(len(self.unique_movielens_movie_ids)),
        )
        return movielens_to_model.loc[movielens_movie_ids].values

    def movielens_movie_to_imdb_movie_ids(self, movielens_movie_ids: np.array):
        movielens_to_imdb = self["links"].set_index("movielensId")["imdbId"]
        return movielens_to_imdb.loc[movielens_movie_ids].values

    def model_item_to_imdb_movie_ids(self, item_ids: np.array):
        return self.movielens_movie_to_imdb_movie_ids(
            self.model_item_to_movielens_movie_ids(item_ids)
        )

    def imdb_movie_to_movielens_movie_id(self, imdb_movie_ids: np.array):
        imdb_to_movielens = self["links"].set_index("imdbId")["movielensId"]
        return imdb_to_movielens.loc[imdb_movie_ids].values

    def imdb_movie_to_model_item_ids(self, imdb_movie_ids: np.array):
        return self.movielens_movie_to_model_item_ids(
            self.imdb_movie_to_movielens_movie_id(imdb_movie_ids)
        )

    @staticmethod
    def movielens_user_ids_to_model_user_ids(movielens_user_ids: np.array):
        return movielens_user_ids - 1

    def ratings_dataframe_to_scipy_csr(self, ratings_dataframe):
        data = ratings_dataframe["rating"].to_numpy()

        user_ids = ratings_dataframe["user_id"].to_numpy()
        row_ids = self.movielens_user_ids_to_model_user_ids(user_ids)

        item_ids = ratings_dataframe["item_id"].to_numpy()
        col_ids = self.movielens_movie_to_model_item_ids(item_ids)

        explicit_feedback = coo_matrix((data, (row_ids, col_ids)), shape=self.shape)
        return explicit_feedback.tocsr()


def read_csv_imdb_ratings(path_to_imdb_ratings_csv: str) -> pd.DataFrame:
    dataframe = pd.read_csv(path_to_imdb_ratings_csv).rename(str.lower, axis="columns")
    dataframe["imdbId"] = dataframe["const"].str.removeprefix("tt").astype(int)
    dataframe = dataframe.set_index("imdbId")
    return dataframe


def main():
    parser = ArgumentParser(
        description="Split and save movielens 25m dataset by timestamps into train and test in provided proportion."
    )
    parser.add_argument("directory", default="local/ml-25m", type=str, nargs="?")
    # TODO: multiple quantiles
    parser.add_argument("quantiles", default=0.8, type=float, nargs="*")
    args = parser.parse_args()

    with wandb.init(
        entity="dkoshman", project="Recommending", job_type="data", config=vars(args)
    ):
        movielens = MovieLens25m(path_to_movielens_folder=args.directory)
        metadata = dict(
            directory=args.directory,
            quantile=args.quantile,
            files={
                i.split(".")[0]: dict(rel_path=i, abs_path=movielens.abs_path(i))
                for i in ["tags.csv", "links.csv", "movies.csv"]
            },
        )
        train, test = movielens.train_test_split(train_quantile=args.quantile)
        for dataframe, stage in zip([train, test], ["train", "test"]):
            df_describe = dataframe.describe().reset_index()
            wandb.log({f"{stage}_describe": wandb.Table(dataframe=df_describe)})
            user_describe = (
                dataframe.groupby("user_id")
                .size()
                .describe()
                .rename("user_groupby_size")
                .reset_index()
            )
            wandb.log({f"{stage}_user_describe": wandb.Table(dataframe=user_describe)})
            item_describe = (
                dataframe.groupby("item_id")
                .size()
                .describe()
                .rename("item_groupby_size")
                .reset_index()
            )
            wandb.log({f"{stage}_item_describe": wandb.Table(dataframe=item_describe)})

            filename = f"{stage}_ratings.csv"
            path = movielens.save_ratings_split(ratings=dataframe, filename=filename)
            metadata["files"][stage] = dict(rel_path=filename, abs_path=path)

        artifact = wandb.Artifact(
            name="movielens25m",
            type="data",
            description=f"Time split quantile {args.quantile}",
            metadata=metadata,
        )
        for name, paths_dict in metadata["files"].items():
            artifact.add_file(local_path=paths_dict["abs_path"], name=name)

        wandb.log_artifact(artifact, name=filename, type="data")


if __name__ == "__main__":
    main()
