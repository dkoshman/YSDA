from typing import TypeVar, Callable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .utils import SparseTensor


Distance = TypeVar(
    "Distance", bound=Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
)


def cosine_distance(
    left: torch.Tensor or "SparseTensor", right: torch.Tensor or "SparseTensor"
) -> torch.Tensor:
    """
    Given two matrices: left of shape [n, d] and right of shape [m, d],
    containing in rows vectors from d-dimensional space, returns matrix S
    of shape [n, m], where S[i, j] = cosine distance between left[i] and right[j].
    """

    def norm(tensor):
        if tensor.is_sparse:
            return torch.sqrt(torch.sparse.sum(tensor**2, dim=1)).to_dense()
        else:
            return torch.sqrt(torch.sum(tensor**2, dim=1))

    left = left.to(torch.float32)
    right = right.to(torch.float32)
    similarity = (left @ right.transpose(0, 1)).to_dense()
    similarity /= norm(left).reshape(-1, 1)
    similarity /= norm(right).reshape(1, -1)
    return 1 - similarity


def weighted_average(tensor, weights):
    """
    Given tensor T of shape [n, m] and weights W of shape [n],
    returns weighted average A of shape [m]:
    A[i] = \sum_j T[j, i] * W[j]
    """
    if (weights < 0).any():
        raise ValueError("Weights must be positive.")
    if (weights == 0).all():
        weights = torch.ones_like(weights)
    weights /= weights.sum()
    return tensor.T @ weights


def safe_log(tensor: torch.Tensor) -> torch.Tensor:
    """Returns 0 if x == 0, else log(x)"""
    tensor[tensor == 0] = 1
    return torch.log(tensor)


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if (p < 0).any() or not torch.isclose(p.sum(), torch.tensor(1.0)):
        raise ValueError("p must be a probability distribution")
    if (q < 0).any() or not torch.isclose(q.sum(), torch.tensor(1.0)):
        raise ValueError("q must be a probability distribution")
    return (p * (safe_log(p) - safe_log(q))).sum()


def pairwise_difference(left, right):
    """
    Given two matrices left, right of shape (n, m),
    returns matrix D of shape (n, m, m) such that
    D[u, i, j] = left[u, i] - right[u, j]
    """
    return left[:, :, None] - right[:, None, :]
