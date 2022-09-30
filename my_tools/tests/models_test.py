import torch

import my_tools.models as models


def test_weight_decay():
    t = torch.tensor([1, 2, 3.0], requires_grad=True)
    l2_coefficient = 0.1
    models.register_regularization_hook(t, l2_coefficient=l2_coefficient)
    t.sum().backward()
    assert (t.grad == t * l2_coefficient + 1).all()

    t.grad.zero_()
    (t**2).sum().backward()
    assert (t.grad == t * l2_coefficient + 2 * t).all()
