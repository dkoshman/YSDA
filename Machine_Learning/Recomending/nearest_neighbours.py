import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix

from utils import torch_sparse_to_scipy_coo


class NearestNeighbours:
    def __init__(self, train_explicit: csr_matrix, n_neighbors=10):
        self.train_explicit = train_explicit
        self.n_neighbors = n_neighbors

    @staticmethod
    def norm(sparse_matrix: coo_matrix):
        return np.maximum(sparse_matrix.multiply(sparse_matrix).sum(axis=1), 1e-8)

    def sparse_cosine_similarity(self, left, right):
        if left.shape[-1] != right.shape[-1]:
            raise ValueError(
                "Cannot compute similarity between tensors with non matching"
                f"last dimensions: {left.shape}, {right.shape}"
            )
        similarity = left @ right.T
        similarity = similarity / self.norm(left) / self.norm(right).T
        return np.asarray(similarity)

    def __call__(self, explicit_feedback):
        if torch.is_tensor(explicit_feedback):
            explicit_feedback = torch_sparse_to_scipy_coo(explicit_feedback)
        similarity = self.sparse_cosine_similarity(
            explicit_feedback, self.train_explicit
        )
        similarity = torch.from_numpy(similarity)
        neighbors_similarity, indices = torch.topk(similarity, k=self.n_neighbors)
        neighbors_similarity /= neighbors_similarity.sum(axis=1)[:, None]

        ratings = np.empty(explicit_feedback.shape)
        for i, (similarity, ind) in enumerate(
            zip(neighbors_similarity.numpy(), indices.numpy())
        ):
            ratings[i] = self.train_explicit[ind].T @ similarity

        return dict(
            ratings=ratings,
            similar_users=indices,
            similarity=neighbors_similarity.to(torch.float32),
        )
