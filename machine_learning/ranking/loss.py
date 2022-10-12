import einops
import torch
import torch.nn as nn

from sklearn.metrics import dcg_score


def pairwise_difference(x):
    """If given vector x, returns matrix X_ij = x_i - x_j"""
    n = x.shape[-1]
    return einops.repeat(x, f"... -> ... {n}") - einops.repeat(x, f"... n -> ... {n} n")


class GradientHook:
    def __init__(self, grad):
        self.grad_values = grad.detach().clone()

    def __call__(self, grad):
        grad.data = self.grad_values
        return grad


class LambdaLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    @staticmethod
    def precalculate_true_relevance_functions(y):
        idcg = (
            torch.tensor(
                [dcg_score(i, i) for i in einops.rearrange(y, "b n -> b () n").cpu()]
            )
            if y.shape[-1] > 1
            else torch.ones(y.shape[0])
        ).to(dtype=torch.float32, device=y.device)
        Y_one = pairwise_difference(2**y) / (
            einops.rearrange(idcg, "b -> b () ()") + torch.finfo(torch.float32).eps
        )
        Y_two = torch.sign(pairwise_difference(y))
        return Y_one, Y_two

    def forward(self, a, y):
        """
        :param a: predicted relevance tensor of shape (batch_size, documents_count)
        :param y: true relevance tensor of same size
        """

        with torch.no_grad():
            pi = torch.argsort(torch.argsort(-a)) + 1
            Y_one, Y_two = self.precalculate_true_relevance_functions(y)
            delta_ndcg = Y_one * pairwise_difference(torch.log2(pi + 1) ** -1)

        A = pairwise_difference(a)
        G = -self.sigma * Y_two * torch.abs(delta_ndcg) / (1 + torch.exp(Y_two * A))
        grad = einops.reduce(2 * G, "b t j -> b t", "sum").float()

        a.register_hook(GradientHook(grad))
        placeholder_loss = a.sum()
        return placeholder_loss
