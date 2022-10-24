import os

import torch
import yaml

from ..entrypoint import RecommendingBuilder
from ..movielens.data import ImdbRatings
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
    imdb = ImdbRatings()
    explicit = imdb.explicit_feedback_torch()
    recommendations = model.online_recommend(explicit)
    items_description = imdb.items_description(recommendations[0])
    assert items_description is not None
