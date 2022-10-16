import functools
import os
import string

import numpy as np
import pandas as pd

from scipy.sparse import coo_matrix

from ..utils import scipy_coo_to_torch_sparse


class MovieLens:
    def __init__(self, path_to_movielens_folder="local/ml-100k"):
        self.path_to_movielens_folder = path_to_movielens_folder

    def __getitem__(self, filename):
        match filename:
            case "u.info":
                return self.read(
                    filename=filename,
                    names=["quantity", "index"],
                    index_col="index",
                    sep=" ",
                )
            case "u.genre":
                return self.read(
                    filename=filename, names=["name", "id"], index_col="id"
                )
            case "u.occupation":
                return self.read(filename=filename, names=["occupation"])
            case "u.user":
                return self.read(
                    filename=filename,
                    names=["user id", "age", "gender", "occupation", "zip code"],
                    index_col="user id",
                )
            case "u.item":
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
                names=["user_id", "item_id", "rating", "timestamp"],
                sep="\t",
                dtype="int32",
            )
        raise ValueError(f"File {filename} not found.")

    def read(self, filename, sep="|", header=None, **kwargs):
        path = os.path.join(self.path_to_movielens_folder, filename)
        dataframe = pd.read_csv(path, sep=sep, header=header, **kwargs)
        return dataframe.squeeze()

    @property
    @functools.cache
    def shape(self):
        info = self["u.info"]
        return info["users"], info["items"]

    def explicit_feedback_scipy_csr(self, name):
        dataframe = self[name]
        data = dataframe["rating"].to_numpy()
        # Stored ids start with 1.
        row_ids = dataframe["user_id"].to_numpy() - 1
        col_ids = dataframe["item_id"].to_numpy() - 1
        explicit_feedback = coo_matrix((data, (row_ids, col_ids)), shape=self.shape)
        return explicit_feedback.tocsr()


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
        self.movielens = MovieLens(path_to_movielens_folder)

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
        return scipy_coo_to_torch_sparse(self.explicit_feedback_scipy())
