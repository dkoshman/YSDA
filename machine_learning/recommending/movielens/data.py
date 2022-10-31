import abc
import functools
import os
import string

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

from my_tools.utils import to_torch_coo


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
    def __getitem__(self, filename):
        return self.read(filename=filename)

    def read(self, filename, **kwargs):
        path = os.path.join(self.path_to_movielens_folder, filename)
        dataframe = pd.read_csv(path, **kwargs)
        return dataframe.squeeze()

    def explicit_feedback_scipy_csr(self, name):
        return self.to_scipy_csr(self[name])

    def to_scipy_csr(self, dataframe):
        data = dataframe["rating"].to_numpy()
        # Stored ids start with 1.
        row_ids = dataframe["user_id"].to_numpy() - 1
        col_ids = dataframe["item_id"].to_numpy() - 1
        explicit_feedback = coo_matrix((data, (row_ids, col_ids)), shape=self.shape)
        return explicit_feedback.tocsr()


class MovieLens100k(MovieLensBase):
    def __init__(self, path_to_movielens_folder="local/ml-100k"):
        super().__init__(path_to_movielens_folder)

    def __getitem__(self, filename):
        # match filename:
        #     case "u.info":
        #         return self.read(
        #             filename=filename,
        #             names=["quantity", "index"],
        #             index_col="index",
        #             sep=" ",
        #         )
        #     case "u.genre":
        #         return self.read(
        #             filename=filename, names=["name", "id"], index_col="id"
        #         )
        #     case "u.occupation":
        #         return self.read(filename=filename, names=["occupation"])
        #     case "u.user":
        #         return self.read(
        #             filename=filename,
        #             names=["user id", "age", "gender", "occupation", "zip code"],
        #             index_col="user id",
        #         )
        #     case "u.item":
        #         return self.read(
        #             filename=filename,
        #             names=[
        #                 "movie id",
        #                 "movie title",
        #                 "release date",
        #                 "video release date",
        #                 "IMDb URL",
        #                 "unknown",
        #                 "Action",
        #                 "Adventure",
        #                 "Animation",
        #                 "Children's",
        #                 "Comedy",
        #                 "Crime",
        #                 "Documentary",
        #                 "Drama",
        #                 "Fantasy",
        #                 "Film-Noir",
        #                 "Horror",
        #                 "Musical",
        #                 "Mystery",
        #                 "Romance",
        #                 "Sci-Fi",
        #                 "Thriller",
        #                 "War",
        #                 "Western",
        #             ],
        #             index_col="movie id",
        #             encoding_errors="backslashreplace",
        #         )
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
        else:
            return self.read(filename=filename)

    @property
    @functools.lru_cache()
    def shape(self):
        ratings = self["ratings"]
        n_users = ratings["user_id"].max()
        n_items = ratings["item_id"].max()
        return n_users, n_items

    def train_test_split(
        self,
        test_proportion=0.1,
        random_state=42,
    ):
        ratings = self["ratings"]
        test_size = int(len(ratings) * test_proportion)
        test_index = (
            ratings.iloc[np.random.permutation(ratings.index)]
            .reset_index()
            .groupby("user_id")
            .nth(range(10, 100))
            .reset_index()
            .sample(test_size, random_state=random_state)["index"]
            .values
        )
        test_ratings = ratings.loc[test_index]
        train_ratings = ratings.loc[ratings.index.difference(test_index)]
        return train_ratings, test_ratings

    def save_train_test_split(self, train_ratings, test_ratings):
        for dataframe, stage in [(train_ratings, "train"), (test_ratings, "test")]:
            dataframe.to_csv(
                os.path.join(self.path_to_movielens_folder, f"{stage}_ratings.csv"),
                index=False,
            )


class ImdbRatings:
    """
    Class to transform ratings downloaded from IMDB user profile
    web page as csv table into movielens explicit feedback.
    """

    def __init__(
        self,
        path_to_imdb_ratings_csv="local/my_imdb_ratings.csv",
        path_to_movielens_folder="local/ml-100k",
    ):
        self.path_to_imdb_ratings_csv = path_to_imdb_ratings_csv
        self.movielens = MovieLens100k(path_to_movielens_folder)

    @property
    def imdb_ratings(self):
        return pd.read_csv(self.path_to_imdb_ratings_csv)

    @staticmethod
    def normalize_titles(titles):
        titles = (
            titles.str.lower()
            .str.normalize("NFC")
            .str.replace(f"[{string.punctuation}]", "", regex=True)
            .str.replace("the", "")
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        return titles

    def explicit_feedback_scipy(self):
        imdb_ratings = self.imdb_ratings
        imdb_ratings["Title"] = self.normalize_titles(imdb_ratings["Title"])
        imdb_ratings["Your Rating"] = (imdb_ratings["Your Rating"] + 1) // 2

        movielens_titles = self.movielens["u.item"]["movie title"]
        movielens_titles = self.normalize_titles(movielens_titles.str.split("(").str[0])

        liked_items_ids = []
        liked_items_ratings = []
        for title, rating in zip(imdb_ratings["Title"], imdb_ratings["Your Rating"]):
            for movie_id, ml_title in movielens_titles.items():
                if title == ml_title:
                    liked_items_ids.append(movie_id - 1)
                    liked_items_ratings.append(rating)

        data = liked_items_ratings
        row = np.zeros(len(liked_items_ids))
        col = np.array(liked_items_ids)
        shape = (1, self.movielens.shape[1])
        explicit_feedback = coo_matrix((data, (row, col)), shape=shape)
        return explicit_feedback

    def explicit_feedback_torch(self):
        return to_torch_coo(self.explicit_feedback_scipy())

    def items_description(self, item_ids):
        item_ids += 1
        items_description = self.movielens["u.item"].loc[item_ids]
        imdb_urls = "https://www.imdb.com/find?q=" + items_description[
            "movie title"
        ].str.split("(").str[0].str.replace(r"\s+", "+", regex=True)
        items_description = pd.concat(
            [items_description["movie title"], imdb_urls], axis="columns"
        )
        return items_description


if __name__ == "__main__":
    movielens = MovieLens25m()
    movielens.save_train_test_split(*movielens.train_test_split())
