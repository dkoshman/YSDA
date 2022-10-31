import asyncio
from enum import Enum
from typing import TYPE_CHECKING

import gradio as gr
import numpy as np
import pandas as pd
import requests
import scipy
import torch
import wandb

from machine_learning.recommending.models import SLIM, SVDRecommender
from machine_learning.recommending.movielens.data import MovieLens25m

if TYPE_CHECKING:
    from machine_learning.recommending.interface import RecommenderModuleBase

state_dict_path = "/Users/dimakoshman/MovieRecommender/svd.pt"
movie_lens_25m_directory = "/Users/dimakoshman/MovieRecommender/ml-25m"


def load_model(torch_nn_module, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model = torch_nn_module()
    model.load_state_dict(state_dict)
    return model


class Feedback(Enum):
    NONE = "Haven't seen it"
    ONE, TWO, THREE, FOUR, FIVE = map(str, range(1, 6))

    @property
    def int(self):
        return {v: i for i, v in enumerate(Feedback)}[self]


class MovieMarkdownGenerator:
    def __init__(
        self,
        links_dataframe,
        movies_dataframe,
        tmdb_api_token="0369096d5891e7959f8e5ae35bae71bc",
    ):
        self.links = links_dataframe
        self.movies_dataframe = movies_dataframe
        for dataframe in [self.links, self.movies_dataframe]:
            dataframe.set_index("movieId", inplace=True)
            min_index = dataframe.index.min()
            if min_index == 0:
                pass
            elif min_index == 1:
                dataframe.index -= 1
            else:
                raise ValueError("Index of dataframe should start with 0 or 1.")
        self.tmdb_api_token = tmdb_api_token

    def imdb_id(self, movie_id) -> str:
        imdb_id = self.links["imdbId"].loc[movie_id]
        return f"{imdb_id:07d}"

    def imdb_url(self, movie_id):
        return f"https://www.imdb.com/title/tt{self.imdb_id(movie_id)}"

    def tmdb_request_url(self, imdb_id):
        return (
            f"https://api.themoviedb.org/3/find/tt{imdb_id}"
            f"?api_key={self.tmdb_api_token}&external_source=imdb_id"
        )

    @staticmethod
    def poster_url_from_tmdb_poster_path(poster_path, size="w200"):
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"

    def __call__(self, movie_id):
        imdb_id = self.imdb_id(movie_id)
        tmdb_request_url = self.tmdb_request_url(imdb_id)
        request = requests.get(tmdb_request_url)
        if not request.ok:
            wandb.log(
                dict(
                    failed_request=dict(
                        reason=request.reason,
                        status_code=request.status_code,
                        text=request.text,
                    )
                )
            )
            markdown = self.movies_dataframe.loc[movie_id]["movie title"]
            return markdown, ""

        movie_results = request.json()["movie_results"][0]
        poster_url = self.poster_url_from_tmdb_poster_path(
            poster_path=movie_results["poster_path"]
        )
        title = movie_results["title"]
        imdb_url = self.imdb_url(movie_id)
        markdown = f"""[{title}]({imdb_url})"""
        return markdown, poster_url


class AsyncRecommender:
    def __init__(self, recommender: "RecommenderModuleBase"):
        self.recommender = recommender
        self.user_explicit = pd.Series(
            data=np.full(self.recommender.n_items, fill_value=np.nan)
        )
        self.movie_id = self.recommend()
        self.task = asyncio.create_task(self.async_recommend())

    def recommend(self):
        user_explicit = scipy.sparse.coo_matrix(
            self.user_explicit.fillna(0).values.reshape(1, -1)
        )
        user_recommendations = self.recommender.online_recommend(
            user_explicit, n_recommendations=self.recommender.n_items
        )[0].numpy()
        new_items_mask = self.user_explicit.isna().values
        new_items_positions = new_items_mask[user_recommendations].nonzero()[0]
        movie_id = user_recommendations[new_items_positions[0]]
        return movie_id

    async def async_recommend(self):
        self.user_explicit.loc[self.movie_id] = 0
        return self.recommend()

    async def next_recommendation(self, rating_of_previous_movie: int):
        wandb.log(
            dict(
                movie_id=self.movie_id,
                rating=rating_of_previous_movie,
            )
        )
        self.user_explicit.loc[self.movie_id] = rating_of_previous_movie
        self.movie_id = await self.task
        self.task = asyncio.create_task(self.async_recommend())
        return self.movie_id

    def __del__(self):
        wandb.log(
            dict(feedback=wandb.Table(dataframe=self.user_explicit.dropna().to_frame()))
        )


movielens = MovieLens25m(path_to_movielens_folder=movie_lens_25m_directory)
movie_markdown_generator = MovieMarkdownGenerator(
    links_dataframe=movielens["links"],
    movies_dataframe=movielens["movies"],
)
model = load_model(torch_nn_module=SVDRecommender, state_dict_path=state_dict_path)

with gr.Blocks() as app:
    async_recommender = AsyncRecommender(recommender=model)
    gr.Markdown("This movie recommender adapts to your preferences")
    with gr.Row():
        first_movie_markdown, first_image_url = movie_markdown_generator(
            async_recommender.movie_id
        )
        image = gr.Image(first_image_url, interactive=False)
        image.style(width=200)
        with gr.Column(scale=4):
            markdown = gr.Markdown(first_movie_markdown)

            def make_button(feedback):
                async def wrapper():
                    movie_id = await async_recommender.next_recommendation(feedback.int)
                    movie_markdown, image_url = movie_markdown_generator(movie_id)
                    return movie_markdown, image_url

                button = gr.Button(feedback.value)
                button.click(
                    fn=wrapper,
                    inputs=[],
                    outputs=[markdown, image],
                )
                return button

            none_button = make_button(Feedback.NONE)
            with gr.Row():
                buttons = [make_button(feedback) for feedback in list(Feedback)[1:]]
