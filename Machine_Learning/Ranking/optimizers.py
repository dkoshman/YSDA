import functools
import torch


class OptimizerBuilder:
    def __init__(
        self,
        optimizer_name="adam",
        scheduler_name=None,
        min_lr=1e-3,
        max_lr=1e-2,
    ):
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.min_lr = min_lr
        self.max_lr = max_lr

    def __call__(self, module):
        initial_lr = {
            None: self.min_lr,
            "cosine": self.max_lr,
            "cycle": self.min_lr,
            "lambda": self.max_lr,
        }[self.scheduler_name]

        match self.optimizer_name:
            case "adam":
                optimizer = torch.optim.Adam(module.parameters(), lr=initial_lr)
            case "sgd_with_momentum":
                optimizer = torch.optim.SGD(
                    module.parameters(), lr=initial_lr, momentum=0.9
                )
            case _:
                raise RuntimeError(f"Invalid optimizer '{self.optimizer_name}'")

        match self.scheduler_name:
            case None:
                scheduler = None
            case "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=10
                )
            case "cycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.max_lr, total_steps=30
                )
            case "lambda":
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=LRLambda(
                        num_parameters=sum(p.numel() for p in module.parameters())
                    ),
                )
            case _:
                raise RuntimeError(f"Invalid scheduler '{self.scheduler_name}'")

        return optimizer, scheduler


class LRLambda:
    """Learning rate scheduler with warmup and subsequent reverse square root fading"""

    def __init__(self, num_parameters, warmup_epochs=20):
        """
        :param num_parameters: number of parameters in model
        :param warmup_epochs: number of warmup epochs
        """
        self.num_parameters = num_parameters
        self.warmup_epochs = warmup_epochs

    def __call__(self, epoch):
        epoch += 1
        learning_rate_multiplier = self.num_parameters**-0.5 * min(
            (self.warmup_epochs / epoch) ** 0.5, epoch / self.warmup_epochs
        )
        return learning_rate_multiplier
