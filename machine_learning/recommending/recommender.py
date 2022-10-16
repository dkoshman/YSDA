from typing import Literal

import torch

from .models import baseline


class RatingsToRecommendations:
    def __init__(
        self,
        explicit_feedback,
        model,
        nearest_neighbors_model: Literal[
            "BM25Recommender",
            "CosineRecommender",
            "TFIDFRecommender",
        ] = "BM25Recommender",
        num_neighbors=20,
    ):
        """
        :param explicit_feedback: explicit feedback matrix
        :param model: Callable[[user_ids], ratings]
        :param nearest_neighbors_model: name of nearest neighbors
        model to use for generating recommendations fo new users
        :param num_neighbors: number of nearest neighbors to use
        """
        self.explicit_feedback = explicit_feedback
        self.nearest_neighbours = baseline.ImplicitNearestNeighborsRecommender(
            n_users=self.explicit_feedback.shape[0],
            n_items=self.explicit_feedback.shape[1],
            implicit_model=nearest_neighbors_model,
            num_neighbors=num_neighbors,
        )
        self.nearest_neighbours.fit(self.explicit_feedback)
        self.model = model

    @torch.inference_mode()
    def nearest_neighbours_ratings(self, explicit_feedback):
        """
        Predicts ratings for new user defined by its explicit feedback by
        searching for closest neighbors and taking weighted average of
        their predicted ratings.
        """

        nn_dict = self.nearest_neighbours.similar_users(
            users_feedback=explicit_feedback
        )
        similar_users = nn_dict["similar_users"]
        similarity = nn_dict["similarity"]
        similarity /= similarity.sum(axis=1)[:, None]
        print(similarity.shape)

        ratings = torch.empty(*explicit_feedback.shape, dtype=torch.float32)
        for i, (similar_users_row, similarity_row) in enumerate(
            zip(similar_users, similarity)
        ):
            similar_ratings = self.model(user_ids=similar_users_row)
            if not torch.is_tensor(similar_ratings):
                similar_ratings = torch.from_numpy(similar_ratings)

            similar_ratings = similar_ratings.to("cpu", torch.float32)
            ratings[i] = similar_ratings.transpose(0, 1) @ similarity_row
        return ratings

    @torch.inference_mode()
    def __call__(
        self,
        user_ids=None,
        users_explicit_feedback=None,
        filter_already_liked_items=True,
        n_recommendations=10,
    ):
        if user_ids is None and users_explicit_feedback is None:
            raise ValueError(
                "At least one of user_ids, user_explicit_feedback must be passed."
            )

        if user_ids is None:
            ratings = self.nearest_neighbours_ratings(users_explicit_feedback)
        else:
            ratings = self.model(user_ids=user_ids)

        if not torch.is_tensor(ratings):
            ratings = torch.from_numpy(ratings)

        if filter_already_liked_items:
            if user_ids is None:
                already_liked_items = users_explicit_feedback > 0
            else:
                already_liked_items = self.explicit_feedback[user_ids] > 0

            ratings = torch.where(
                torch.from_numpy(already_liked_items.toarray()),
                torch.finfo().min,
                ratings,
            )

        if n_recommendations is None:
            recommendations = torch.argsort(ratings, descending=True)
        else:
            values, recommendations = torch.topk(input=ratings, k=n_recommendations)
        return recommendations
