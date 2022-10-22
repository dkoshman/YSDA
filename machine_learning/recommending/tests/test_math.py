import numpy as np
import torch

from machine_learning.recommending.maths import (
    cosine_distance,
    weighted_average,
    safe_log,
    kl_divergence,
)


def test_cosine_similarity():
    for _ in range(10):
        dimension = np.random.randint(1, 100)
        left_size = np.random.randint(1, 100)
        right_size = np.random.randint(1, 100)
        left = torch.randn(left_size, dimension)
        right = torch.randn(right_size, dimension)
        distance = cosine_distance(left, right)
        torch_distance = 1 - torch.cosine_similarity(
            left.unsqueeze(-1), right.T.unsqueeze(0)
        )
        assert torch.isclose(distance, torch_distance, atol=1e-5).all()


def test_weighted_average():
    for _ in range(10):
        tensor = torch.randn(np.random.randint(1, 1000), np.random.randint(1, 1000))
        weights = torch.ones(tensor.shape[0]) * np.random.uniform(1, 1000)
        wa = weighted_average(tensor, weights)
        assert wa.shape == (tensor.shape[1],)
        mean = tensor.mean(0)
        assert torch.isclose(wa, mean, atol=1e-5).all()
        wa = weighted_average(tensor, torch.randn(tensor.shape[0]).abs())
        assert (~wa.isnan()).all()


def test_safe_log():
    for _ in range(100):
        tensor = (
            torch.randint(
                low=0,
                high=np.random.randint(1, 10),
                size=[np.random.randint(0, 10), np.random.randint(0, 10)],
            )
            * np.random.randn()
        ).abs()
        zero_mask = tensor == 0
        assert torch.isclose(
            safe_log(tensor)[~zero_mask], torch.log(tensor)[~zero_mask]
        ).all()
        assert torch.isclose(
            safe_log(tensor)[zero_mask], torch.zeros_like(tensor)[zero_mask]
        ).all()
        assert (~safe_log(tensor).isnan()).all()
        assert (~safe_log(torch.zeros_like(tensor)).isnan()).all()


def test_kl_divergence():
    for _ in range(100):
        p = torch.randn(np.random.randint(1, 1000)).abs()
        q = torch.randn_like(p).abs()
        p = p / p.sum()
        q = q / q.sum()
        p_q_kl_divergence = kl_divergence(p, q)
        assert p_q_kl_divergence.numel() == 1
        assert p_q_kl_divergence >= 0
        assert torch.isclose(kl_divergence(p, p), torch.tensor(0.0))
