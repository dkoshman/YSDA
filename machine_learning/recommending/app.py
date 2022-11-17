import asyncio
import functools
import os.path
import time
from typing import TYPE_CHECKING

from aenum import Enum, MultiValue, Unique
import gradio as gr
import numpy as np
import requests
import scipy
import torch
import wandb

from machine_learning.recommending.movielens.data import (
    read_csv_imdb_ratings,
    explicit_from_imdb_ratings,
)
from machine_learning.recommending.utils import wandb_timeit


if TYPE_CHECKING:
    from machine_learning.recommending.interface import RecommenderModuleBase


class Feedback(Enum, settings=(MultiValue, Unique), init="str rating"):
    not_interested = "Haven't seen it and not interested", 0
    interested = "Haven't seen it, but I'm interested", 3.5
    # one = "1", 1
    # two = "2", 2
    # three = "3", 3
    # four = "4", 4
    # five = "5", 5
    one = "⭐️", 1
    two = "⭐️⭐️", 2
    three = "⭐️⭐️⭐️", 3
    four = "⭐️⭐️⭐️⭐️", 4
    five = "⭐️⭐️⭐️⭐️⭐️", 5


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

    @staticmethod
    @functools.lru_cache(maxsize=10_000)
    def cached_request(url):
        """Caching requests saves a lot of time."""
        return requests.get(url)

    def __call__(self, item_id):
        movielens_id = self.movielens.dense_item_to_movielens_movie_ids([item_id])[0]
        imdb_id = self.movielens.movielens_movie_to_imdb_movie_ids([movielens_id])[0]
        tmdb_request_url = self.tmdb_request_url(
            imdb_id=imdb_id, tmdb_api_token=self.tmdb_api_token
        )
        request = self.cached_request(url=tmdb_request_url)
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
            markdown = self.movies_dataframe.loc[movielens_id]["movie title"]
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
        n_recommendations=10,
    ):
        self.recommender = recommender
        self.movie_markdown_generator = movie_markdown_generator
        self.movielens = self.movie_markdown_generator.movielens
        self.user_activity = None
        self.session_length = 0
        self.n_recommendations = n_recommendations
        wandb.define_metric("session_length", summary="max")
        self.session_id = time.time()
        self.explicit = np.full(self.recommender.n_items, fill_value=np.nan)
        self.movie_id, self.initial_markdown, self.initial_poster_url = self.content()
        self.content_task = asyncio.create_task(self.async_content())

    def __deepcopy__(self, memo):
        """
        Gradio makes a copy of state block at some point, I don't know why exactly.
        The problem is that you can't copy a task, and returning new instance instead
        doesn't break anything, so that's what I'm going to do.
        """
        return Session(
            recommender=self.recommender,
            movie_markdown_generator=self.movie_markdown_generator,
        )

    def content(self):
        explicit = scipy.sparse.coo_matrix(np.nan_to_num(self.explicit).reshape(1, -1))
        with torch.no_grad(), wandb_timeit("online_recommend"):
            user_recommendations = self.recommender.online_recommend(
                explicit, n_recommendations=self.recommender.n_items
            )[0].numpy()

        new_items_mask = np.isnan(self.explicit)
        new_items_positions = new_items_mask[user_recommendations].nonzero()[0]
        movie_id = user_recommendations[new_items_positions[0]]

        # To not recommend same item twice before rating gets saved.
        self.explicit[movie_id] = 0

        with wandb_timeit("tmdb"):
            markdown, poster_url = self.movie_markdown_generator(item_id=movie_id)

        return movie_id, markdown, poster_url

    async def async_content(self):
        return self.content()

    def save_rating(self, rating, movie_id):
        if self.explicit[movie_id] != 0:
            raise ValueError(
                f"Movie {movie_id} was already rated {self.explicit[movie_id]}."
            )
        self.explicit[movie_id] = rating
        self.session_length += 1
        wandb.log(
            dict(
                session_id=self.session_id,
                movie_id=movie_id,
                rating=rating,
                session_length=self.session_length,
                user_activity=self.user_activity,
            )
        )

    async def next_content(self, feedback: Feedback):
        with wandb_timeit("await_content"):
            next_movie_id, markdown, poster_url = await self.content_task
        self.save_rating(rating=feedback.rating, movie_id=self.movie_id)
        wandb.log(
            dict(
                interested=float(feedback == Feedback.interested),
                not_interested=float(feedback == Feedback.not_interested),
            )
        )
        self.movie_id = next_movie_id
        self.content_task = asyncio.create_task(self.async_content())
        return markdown, poster_url

    def upload(self, file_object) -> str:
        filename = file_object.name
        try:
            imdb_ratings = read_csv_imdb_ratings(filename)
        except Exception as e:
            wandb.log(dict(upload_error=str(e)))
            return f"Couldn't read imdb ratings file: \n{str(e)}"

        explicit = explicit_from_imdb_ratings(
            imdb_ratings=imdb_ratings,
            movielens_25m=self.movielens,
            movielens_the_model_trained_on=self.movielens,
        )
        with torch.no_grad(), wandb_timeit("online_recommend"):
            recommendations = self.recommender.online_recommend(
                explicit, n_recommendations=self.n_recommendations
            )

        markdown = ""
        for i, item_id in enumerate(recommendations[0].numpy(), start=1):
            movie_markdown, poster_url = self.movie_markdown_generator(item_id=item_id)
            markdown += f"""{i}. {movie_markdown} ![poster]({poster_url}) \n"""
        return markdown


def interactive_feedback_click(feedback: Feedback):
    async def click_inner(session: Session):
        markdown, poster_url = await session.next_content(feedback=feedback)
        return session, markdown, poster_url

    return click_inner


def gradio_relative_html_path(relative_path):
    return os.path.join("file", relative_path)


def build_app_blocks(
    recommender: "RecommenderModuleBase",
    movie_markdown_generator: MovieMarkdownGenerator,
    media_directory: str,
) -> None:

    gr.Markdown("This movie recommender adapts to your preferences")
    session_state = gr.State(
        value=Session(
            recommender=recommender, movie_markdown_generator=movie_markdown_generator
        )
    )

    with gr.Tab("Interactively rate movies"):
        with gr.Row():
            image_block = gr.Image(
                session_state.value.initial_poster_url, interactive=False
            )
            image_block.style(width=200)
            with gr.Column(scale=4):
                recommendations_markdown_block = gr.Markdown(
                    session_state.value.initial_markdown
                )

                def build_button(feedback: Feedback):
                    button = gr.Button(feedback.str)
                    button.click(
                        fn=interactive_feedback_click(feedback),
                        inputs=[session_state],
                        outputs=[
                            session_state,
                            recommendations_markdown_block,
                            image_block,
                        ],
                    )
                    return button

                with gr.Row():
                    build_button(Feedback.not_interested)
                    build_button(Feedback.interested)
                with gr.Row():
                    for rated_feedback in list(Feedback)[2:]:
                        build_button(rated_feedback)

    with gr.Tab("Upload IMDb ratings"):

        def image_html(filename, width=400):
            path = gradio_relative_html_path(os.path.join(media_directory, filename))
            return f"""<img src="{path}" width="{width}">"""

        howto_markdown = gr.Markdown(
            f"""
            If you have a profile on IMDb, you can upload the ratings from your profile page:
            
            1. Go to [imdb.com](https://www.imdb.com)
            2. Click on your profile and then on the "Your ratings" tab
            {image_html("step1.png")}
            3. Click on the ellipsis menu on "Your ratings" page
            {image_html("step2.png")}
            4. Select "Export"
            {image_html("step3.png")}
            5. Now your ratings will be downloaded in csv format, and you can upload them below
            """
        )
        file_block = gr.File(interactive=True)
        recommendations_markdown_block = gr.Markdown("")
        file_block.upload(
            fn=lambda file: (
                howto_markdown.update(visible=False),
                session_state.value.upload(file),
            ),
            inputs=[file_block],
            outputs=[howto_markdown, recommendations_markdown_block],
        )