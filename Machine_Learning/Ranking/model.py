import einops
import torch

from torch import nn


class DSSM(nn.Module):
    def __init__(self, hidden_dimensions=None):
        """:param hidden_dimensions: list of hidden dimensions sizes"""
        super().__init__()
        self.embedding = DSSM.net(DSSM.dssm_layer, hidden_dimensions)

    @staticmethod
    def dssm_layer(in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.Tanh()
        )

    @staticmethod
    def net(layer, layer_dimensions):
        return nn.Sequential(
            *[layer(i, j) for i, j in zip(layer_dimensions, layer_dimensions[1:])]
        )

    @staticmethod
    def calculate_relevance(query_embedding, documents_embeddings):
        query_embedding = einops.repeat(
            query_embedding, f"b n -> b {documents_embeddings.shape[1]} n"
        )
        relevance = nn.CosineSimilarity(dim=-1)(
            query_embedding,
            documents_embeddings,
        )
        return relevance

    def forward(self, batch):
        query = self.embedding(batch["query"])
        documents = self.embedding(batch["documents"])
        relevance = self.calculate_relevance(query, documents)
        return relevance


class DSSMExtended(DSSM):
    def __init__(self, hidden_dimensions=None, head_dimensions=None):
        """
        :param hidden_dimensions: list of hidden dimensions sizes
        :param head_dimensions: list of dimensions sizes for head
        """
        super().__init__(hidden_dimensions)
        self.head = DSSM.net(DSSMExtended.head_layer, head_dimensions + [1])

    @staticmethod
    def head_layer(in_dim, out_dim):
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.Tanh())

    def forward(self, batch):
        text_relevance = super().forward(batch)
        features = torch.cat(
            [
                einops.rearrange(text_relevance, "b d -> b d ()"),
                batch["pairwise_numeric_features"],
            ],
            dim=-1,
        )
        relevance = einops.rearrange(self.head(features), "b d 1 -> b d")
        return relevance
