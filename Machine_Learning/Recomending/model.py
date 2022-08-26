import torch

from my_ml_tools.models import weight_decay


class ProbabilityMatrixFactorization(torch.nn.Module):
    """predicted rating = user_factors @ item_factors, with bias and L2 regularization"""

    def __init__(self, n_users, n_items, latent_dimension, regularization_lambda):
        super().__init__()

        self.regularization_lambda = regularization_lambda

        self.user_factors = torch.nn.Parameter(torch.empty(n_users, latent_dimension))
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
