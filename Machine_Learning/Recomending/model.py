import torch

class WeightDecayGradientHook:
    def __init__(self, weight, regularization_lambda):
        self.decay = regularization_lambda * weight.detach().clone()

    def __call__(self, grad):
        grad.data += self.decay
        return grad


def weight_decay(tensor, regularization_lambda):
    if not tensor.requires_grad:
        return tensor

    hook = WeightDecayGradientHook(tensor, regularization_lambda)
    tensor.register_hook(hook)
    return tensor


class ProbabilityMatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, latent_dimension, regularization_lambda):
        super().__init__()

        self.regularization_lambda = regularization_lambda

        self.user_weight = torch.nn.Parameter(torch.empty(n_users, latent_dimension))
        torch.nn.init.xavier_normal_(self.user_weight)
        self.user_bias = torch.nn.Parameter(torch.zeros(n_users, 1))

        self.item_weight = torch.nn.Parameter(torch.empty(n_items, latent_dimension))
        torch.nn.init.xavier_normal_(self.item_weight)
        self.item_bias = torch.nn.Parameter(torch.zeros(n_items, 1))

        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, user_ids, item_ids):
        user_weight = self.user_weight[user_ids]
        user_bias = self.user_bias[user_ids]

        item_weight = self.item_weight[item_ids]
        item_bias = self.item_bias[item_ids]

        rating = user_weight @ item_weight.T + user_bias + item_bias.T + self.bias
        rating = self.sigmoid(rating)

        weight_decay(user_weight, self.regularization_lambda)
        weight_decay(item_weight, self.regularization_lambda)
        weight_decay(user_bias, self.regularization_lambda)
        weight_decay(item_bias, self.regularization_lambda)

        return rating