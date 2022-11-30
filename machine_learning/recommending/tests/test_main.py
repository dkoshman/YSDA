from machine_learning.recommending.movielens.data import (
    csv_imdb_ratings_to_dataframe,
    MovieLens100k,
)
from .conftest import IMDB_RATINGS_PATH, MOVIELENS100K_DIRECTORY

from ..movielens.lit import MovieLensNonGradientRecommender
from ..utils import fetch_artifact, load_path_from_artifact


def test_fitted_model():
    artifact_name = "PopularRecommenderTesting"
    artifact = fetch_artifact(
        entity="dkoshman", project="Testing", artifact_name=artifact_name
    )
    checkpoint_path = load_path_from_artifact(artifact)
    module = MovieLensNonGradientRecommender.load_from_checkpoint(checkpoint_path)
    model = module.model
    imdb_ratings = csv_imdb_ratings_to_dataframe(
        path_to_imdb_ratings_csv=IMDB_RATINGS_PATH
    )
    movielens = MovieLens100k(directory=MOVIELENS100K_DIRECTORY)
    explicit = movielens.imdb_ratings_dataframe_to_explicit(imdb_ratings=imdb_ratings)
    recommendations = model.online_recommend(explicit)
    items_description = movielens.items_description(dense_item_ids=recommendations)
    assert items_description is not None
