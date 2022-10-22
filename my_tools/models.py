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


def register_regularization_hook(tensor, l2_coefficient, l1_coefficient=0):
    """Adds weight decay gradient hook to tensor in place"""
    if tensor.requires_grad:
        hook = RegularizationGradientHook(tensor, l2_coefficient, l1_coefficient)
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


class IgnoreShapeMismatchOnLoadStateDictMixin:
    def load_state_dict(
        self: torch.nn.Module, state_dict: Mapping[str, Any], strict: bool = True
    ):
        for key, value in state_dict.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, torch.nn.Parameter):
                    value = torch.nn.Parameter(
                        data=value, requires_grad=attr.requires_grad
                    )
                setattr(self, key, value)
        return super().load_state_dict(state_dict=state_dict, strict=strict)
