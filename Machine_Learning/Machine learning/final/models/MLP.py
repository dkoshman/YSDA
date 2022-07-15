import torch
from torch import nn

import numpy as np


class MLP(nn.Module):
    def __init__(self, in_shape, n_classes, blocks=[32, 64, 32], Dense=nn.Linear):
        super().__init__()
    
        in_features = [np.prod(in_shape), *blocks]
        out_features = [*blocks, n_classes]
        
        self.flatten = nn.Flatten()

        self.stack = nn.Sequential(*[
            nn.Sequential(
                Dense(in_features[idx], out_features[idx]),
                nn.ReLU(),
            )
            for idx in range(len(blocks) + 1)
        ])
        
        
    def forward(self, X):
        return self.stack(self.flatten(X))
