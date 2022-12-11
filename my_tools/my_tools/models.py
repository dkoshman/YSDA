import time
from typing import Mapping, Any

import torch
import wandb


class RegularizationGradientHook:
    def __init__(self, tensor, l2_coefficient, l1_coefficient):
        tensor = tensor.clone().detach()
        self.decay = 2 * l2_coefficient * tensor + l1_coefficient * tensor.sign()

    def __call__(self, grad):
        return grad.clone().detach() + self.decay


def register_regularization_hook(tensor, l2_coefficient, l1_coefficient=0.0):
    """Adds weight decay gradient hook to tensor in place"""
    if tensor.requires_grad and torch.is_grad_enabled():
        hook = RegularizationGradientHook(
            tensor=tensor, l2_coefficient=l2_coefficient, l1_coefficient=l1_coefficient
        )
        tensor.register_hook(hook)


class WandbLoggerMixin:
    def log(self, dict_to_log: dict) -> None:
        if wandb.run is not None:
            wandb.log(dict_to_log)


class StoppingMonitor:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.impatience = 0
        self.min_delta = min_delta
        self.lowest_loss = torch.inf

    def is_time_to_stop(self, loss):
        if loss < self.lowest_loss - self.min_delta:
            self.lowest_loss = loss
            self.impatience = 0
            return False
        self.impatience += 1
        if self.impatience > self.patience:
            self.impatience = 0
            self.lowest_loss = torch.inf
            return True
        return False
