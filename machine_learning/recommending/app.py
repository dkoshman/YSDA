import asyncio
import os
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
from machine_learning.recommending.utils import wandb_timeit

if TYPE_CHECKING:
    from machine_learning.recommending.interface import RecommenderModuleBase


class Feedback(Enum):
    NONE = "Haven't seen it"
    ONE, TWO, THREE, FOUR, FIVE = map(str, range(1, 6))

    @property
    def int(self):
        return {v: i for i, v in enumerate(Feedback)}[self]


def load_model(torch_nn_module, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model = torch_nn_module()
    model.load_state_dict(state_dict)
    return model


class MovieMarkdownGenerator:
    def __init__(self, movielens, tmdb_api_token):
        self.movielens = movielens
        self.tmdb_api_token = tmdb_api_token
        self.movies_dataframe = movielens["movies"].set_index("movieId")

    @staticmethod
    def imdb_url(imdb_id):
        return f"https://www.imdb.com/title/tt{imdb_id:07d}"

    @staticmethod
    def tmdb_request_url(imdb_id, tmdb_api_token):
        return (
            f"https://api.themoviedb.org/3/find/tt{imdb_id:07d}"
            f"?api_key={tmdb_api_token}&external_source=imdb_id"
        )

    @staticmethod
    def poster_url_from_tmdb_poster_path(poster_path, size="w200"):
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"

    def __call__(self, item_id):
        tmdb_id = self.movielens.model_item_ids_to_tmdb_movie_ids(np.array([item_id]))
        imdb_id = self.movielens.tmdb_movie_ids_to_imdb_movie_ids(tmdb_id)[0]
        tmdb_request_url = self.tmdb_request_url(
            imdb_id=imdb_id, tmdb_api_token=self.tmdb_api_token
        )
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
            markdown = self.movies_dataframe.loc[tmdb_id]["movie title"]
            return markdown, ""

        movie_results = request.json()["movie_results"][0]
        poster_url = self.poster_url_from_tmdb_poster_path(
            poster_path=movie_results["poster_path"]
        )
        title = movie_results["title"]
        imdb_url = self.imdb_url(imdb_id=imdb_id)
        markdown = f"""[{title}]({imdb_url})"""
        return markdown, poster_url


class AheadRecommender:
    def __init__(self, recommender: "RecommenderModuleBase"):
        self.recommender = recommender
        self.user_explicit = pd.Series(
            data=np.full(self.recommender.n_items, fill_value=np.nan)
        )
        self.movie_id = self.recommend()
        self.next_movie_id = None

    def recommend(self):
        user_explicit = scipy.sparse.coo_matrix(
            self.user_explicit.fillna(0).values.reshape(1, -1)
        )
        with wandb_timeit("online_recommend"):
            user_recommendations = self.recommender.online_recommend(
                user_explicit, n_recommendations=self.recommender.n_items
            )[0].numpy()
        new_items_mask = self.user_explicit.isna().values
        new_items_positions = new_items_mask[user_recommendations].nonzero()[0]
        movie_id = user_recommendations[new_items_positions[0]]
        return movie_id

    def next_recommendation(self):
        if self.next_movie_id is not None:
            raise ValueError(
                "Before generating another recommendation, "
                "rating for previous recommendation must be saved."
            )
        self.user_explicit.loc[self.movie_id] = 0
        self.next_movie_id = self.recommend()
        return self.next_movie_id

    def save_rating(self, movie_id: int, rating: int):
        if movie_id != self.movie_id:
            raise ValueError(
                f"Provided movie id {movie_id} doesn't "
                f"match stored movie id {self.movie_id}"
            )
        wandb.log(dict(movie_id=self.movie_id, rating=rating))
        self.user_explicit.loc[self.movie_id] = rating
        self.movie_id, self.next_movie_id = self.next_movie_id, None


class AsyncRecommendingAppContentDispenser:
    def __init__(self, recommender: "RecommenderModuleBase", movie_markdown_generator):
        self.ahead_recommender = AheadRecommender(recommender=recommender)
        self.movie_markdown_generator = movie_markdown_generator
        self.movie_id = self.ahead_recommender.movie_id
        self.initial_content = self.movie_markdown_generator(self.movie_id)
        self.next_content = asyncio.create_task(self.content_ahead())

    async def content_ahead(self):
        with wandb_timeit("recommending"):
            movie_id = self.ahead_recommender.next_recommendation()
        with wandb_timeit("tmdb"):
            content = self.movie_markdown_generator(movie_id)
        return movie_id, content

    async def content(self, feedback: Feedback):
        self.ahead_recommender.save_rating(movie_id=self.movie_id, rating=feedback.int)
        self.movie_id, content = await self.next_content
        self.next_content = asyncio.create_task(self.content_ahead())
        return content


def build_app_blocks(
    recommender: "RecommenderModuleBase", movie_markdown_generator
) -> None:
    dispenser = AsyncRecommendingAppContentDispenser(
        recommender=recommender,
        movie_markdown_generator=movie_markdown_generator,
    )
    gr.Markdown("This movie recommender adapts to your preferences")
    with gr.Row():
        first_movie_markdown, first_image_url = dispenser.initial_content
        image = gr.Image(first_image_url, interactive=False)
        image.style(width=200)
        with gr.Column(scale=4):
            markdown = gr.Markdown(first_movie_markdown)

            def build_button(feedback):
                async def wrapper():
                    return await dispenser.content(feedback=feedback)

                button = gr.Button(feedback.value)
                button.click(fn=wrapper, inputs=[], outputs=[markdown, image])
                return button

            build_button(Feedback.NONE)
            with gr.Row():
                for rated_feedback in list(Feedback)[1:]:
                    build_button(rated_feedback)


def main():
    project = "Recommending"
    entity = "dkoshman"
    path_to_movielens_folder = "local/ml-25m"
    state_dict_path = "svd.pt"
    tmdb_api_token = os.environ["TMDB_API_TOKEN"]

    movielens = MovieLens25m(path_to_movielens_folder=path_to_movielens_folder)
    movie_markdown_generator = MovieMarkdownGenerator(
        movielens=movielens,
        tmdb_api_token=tmdb_api_token,
    )
    model = load_model(torch_nn_module=SVDRecommender, state_dict_path=state_dict_path)
    model.eval()

    with gr.Blocks() as app:
        build_app_blocks(
            recommender=model,
            movie_markdown_generator=movie_markdown_generator,
        )

    wandb.finish()
    wandb.init(job_type="app", project=project, entity=entity)
    app.launch()


if __name__ == "__main__":
    main()
