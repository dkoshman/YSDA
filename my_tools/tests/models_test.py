import torch

import my_ml_tools.models as models


def test_weight_decay():
    t = torch.tensor([1, 2, 3.0], requires_grad=True)
    regularization_lambda = 0.1
    models.register_regularization_hook(t, regularization_lambda=regularization_lambda)
    t.sum().backward()
    assert (t.grad == t * regularization_lambda + 1).all()

    t.grad.zero_()
    (t**2).sum().backward()
    assert (t.grad == t * regularization_lambda + 2 * t).all()
