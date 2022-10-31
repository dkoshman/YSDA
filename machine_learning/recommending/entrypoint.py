import os

import torch

from my_tools.entrypoints import LightningConfigBuilder, ConfigDispenser

from . import callbacks, metrics, models
from .movielens import callbacks as movielens_callbacks
from .movielens import lit as movielens_lit
from .utils import wandb_context_manager


class RecommendingBuilder(LightningConfigBuilder):
    @property
    def module_candidates(self):
        return super().module_candidates + [
            callbacks,
            metrics,
            movielens_callbacks,
            models,
            movielens_lit,
        ]

    def build_lightning_module(self):
        lightning_module = self.build_class(
            datamodule_config=self.datamodule_config,
            model_config=self.model_config,
            loss_config=self.loss_config,
            optimizer_config=self.optimizer_config,
            **self.lightning_config,
        )
        return lightning_module


class RecommendingConfigDispenser(ConfigDispenser):
    def from_cli(self):
        parser = self.cli_argument_parser
        parser.add_argument(
            "--cross-validate",
            "-cv",
            action="store_true",
            help="whether to run cross validation",
        )
        parser.add_argument(
            "--wandb-save",
            "-s",
            action="store_true",
            help="whether to run save wandb checkpoint artifact",
        )
        cli_args = vars(parser.parse_args())
        self.__init__(**cli_args)


def fit(config):
    with wandb_context_manager(config):
        if torch.cuda.is_available():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            torch.cuda.init()
        constructor = RecommendingBuilder(config)
        lightning_module = constructor.build_lightning_module()
        trainer = constructor.build_trainer()
        trainer.fit(lightning_module)
        trainer.test(lightning_module)
        return lightning_module
