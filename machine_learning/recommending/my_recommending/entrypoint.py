from my_tools.entrypoint import LightningConfigBuilder, ConfigDispenser

from . import callbacks, metrics, models
from .movielens import lit as movielens_lit
from .movielens import callbacks as movielens_callbacks
from .utils import wandb_context_manager, init_torch


class RecommendingBuilder(LightningConfigBuilder):
    @property
    def module_candidates(self):
        return super().module_candidates + [
            callbacks,
            metrics,
            models,
            movielens_callbacks,
            movielens_lit,
        ]

    def build_lightning_module(self):
        class_name = self.config["lightning_module"]["class_name"]
        lightning_module = self.build_class(class_name=class_name, **self.config)
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
        init_torch(debug=True)
        constructor = RecommendingBuilder(config)
        lightning_module = constructor.build_lightning_module()
        trainer = constructor.build_trainer()
        trainer.fit(lightning_module)
        trainer.test(lightning_module)
    return lightning_module
