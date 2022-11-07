import time
from enum import Enum
from typing import TYPE_CHECKING

import gradio as gr
import numpy as np
import requests
import scipy
import torch
import wandb

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


class Session:
    def __init__(
        self,
        recommender: "RecommenderModuleBase",
        movie_markdown_generator: MovieMarkdownGenerator,
    ):
        self.recommender = recommender
        self.movie_markdown_generator = movie_markdown_generator
        self.session_id = time.time()
        wandb.define_metric("session_length", summary="max")
        self.session_length = 0
        self.explicit = np.full(self.recommender.n_items, fill_value=np.nan)
        self.estimated_user_activity = self.recommender.user_activity.mean().reshape(1)
        self.movie_id = self.recommend()
        self.initial_markdown, self.initial_poster_url = self.movie_markdown_generator(
            item_id=self.movie_id
        )

    def estimate_user_activity(self):
        if n_interacted_items := (~np.isnan(self.explicit)).sum():
            n_rated_items = (self.explicit > 0).sum()
            self.estimated_user_activity = (
                self.estimated_user_activity + n_rated_items / n_interacted_items
            ) / 2
        return self.estimated_user_activity

    def recommend(self):
        explicit = scipy.sparse.coo_matrix(np.nan_to_num(self.explicit).reshape(1, -1))
        with torch.no_grad():
            user_recommendations = self.recommender.online_recommend(
                explicit,
                n_recommendations=self.recommender.n_items,
                estimated_users_activity=self.estimate_user_activity(),
            )[0].numpy()
        new_items_mask = np.isnan(self.explicit)
        new_items_positions = new_items_mask[user_recommendations].nonzero()[0]
        movie_id = user_recommendations[new_items_positions[0]]
        return movie_id

    def save_rating(self, rating):
        if not np.isnan(self.explicit[self.movie_id]):
            raise ValueError(
                f"Movie {self.movie_id} was already "
                f"rated {self.explicit[self.movie_id]}"
            )
        self.explicit[self.movie_id] = rating
        self.session_length += 1
        wandb.log(
            dict(
                session_id=self.session_id,
                movie_id=self.movie_id,
                rating=rating,
                session_length=self.session_length,
            )
        )

    def next_content(self, feedback: Feedback):
        with wandb_timeit("save_rating"):
            self.save_rating(rating=feedback.int)
        with wandb_timeit("recommend"):
            self.movie_id = self.recommend()
        with wandb_timeit("tmdb"):
            markdown, poster_url = self.movie_markdown_generator(item_id=self.movie_id)
        return markdown, poster_url


def build_app_blocks(
    recommender: "RecommenderModuleBase", movie_markdown_generator
) -> None:

    gr.Markdown("This movie recommender adapts to your preferences")
    session_state = gr.State(
        value=Session(
            recommender=recommender, movie_markdown_generator=movie_markdown_generator
        )
    )
    with gr.Row():
        image_block = gr.Image(
            session_state.value.initial_poster_url, interactive=False
        )
        image_block.style(width=200)
        with gr.Column(scale=4):
            markdown_block = gr.Markdown(session_state.value.initial_markdown)

            def build_button(feedback):
                def click(session: Session):
                    markdown, poster_url = session.next_content(feedback=feedback)
                    return session, markdown, poster_url

                button = gr.Button(feedback.value)
                button.click(
                    fn=click,
                    inputs=[session_state],
                    outputs=[session_state, markdown_block, image_block],
                )
                return button

            build_button(Feedback.NONE)
            with gr.Row():
                for rated_feedback in list(Feedback)[1:]:
                    build_button(rated_feedback)
