import contextlib

import torch
import wandb

from my_tools.entrypoints import ConfigDispenser

from machine_learning.recommending.entrypoint import RecommendingBuilder
from machine_learning.recommending.utils import update_from_base_config


def context_manager(config):
    if wandb.run is None and config.get("logger") is not None:
        return wandb.init(project=config.get("project"), config=config)
    return contextlib.nullcontext()


def fit(config):
    with context_manager(config):
        if torch.cuda.is_available():
            torch.cuda.init()
        constructor = RecommendingBuilder(config)
        lightning_module = constructor.build_lightning_module()
        trainer = constructor.build_trainer()
        trainer.fit(lightning_module)
        trainer.test(lightning_module)
        return lightning_module


def cross_validation_configs(config):
    for i in [1, 2, 3, 4, 5]:
        config["datamodule"].update(
            dict(
                train_explicit_file=f"u{i}.base",
                val_explicit_file=f"u{i}.test",
                test_explicit_file=f"u{i}.test",
            )
        )
        yield config


def main(config):
    config = update_from_base_config(
        config, base_config_file="configs/base_config.yaml"
    )
    if not config["wandb_save"]:
        if "WandbCheckpointCallback" in config.get("callbacks", []):
            del config["callbacks"]["WandbCheckpointCallback"]
    if config["cross_validate"]:
        with context_manager(config):
            for config in cross_validation_configs(config):
                fit(config)
    else:
        fit(config)


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


if __name__ == "__main__":
    dispenser = RecommendingConfigDispenser()
    dispenser.from_cli()
    dispenser.launch(main)
