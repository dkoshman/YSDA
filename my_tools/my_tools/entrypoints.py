import abc
import sys

from argparse import ArgumentParser
from typing import Callable

import pytorch_lightning as pl
import wandb
import yaml

from my_tools.utils import full_traceback, build_class


class WandbSweepDispatcher:
    def __init__(self, main: Callable[[dict], None], config, args):
        self.main = main
        self.args = args
        self.sweep_id = self.initialize_sweep(config, args["config_path"])
        self.project = config["project"]
        self.count = args.get("runs", 1)

    def initialize_sweep(self, config, config_file_path):
        """Returns sweep id if it is a field in config, otherwise initializes new sweep."""
        sweep_id = config.get("sweep_id")
        if sweep_id:
            print(f"Using sweep id {sweep_id} from config file", file=sys.stderr)
        else:
            config = self.preprocess_sweep_parameters(config)
            sweep_id = wandb.sweep(config)
            yaml.safe_dump({"sweep_id": sweep_id}, open(config_file_path, "a"))
            print("Added sweep id to config file", file=sys.stderr)
        return sweep_id

    def preprocess_sweep_parameters(self, config):
        """
        Convenience function, will transform sweep config from the following:

            name: "My sweep"
            parameters:
              model:
                dimension:
                  min: 10
                  max: 20
                scheduling: "{0: 10}"

        to:

            name: "My sweep"
            parameters:
              model:
                parameters:
                  dimension:
                    min: 10
                    max: 20
                  scheduling:
                    value: "{0: 10}"
        """
        parameters = config["parameters"]
        for parameter, subconfig in parameters.items():
            parameters[parameter] = self.insert_wandb_sweep_specific_keys(
                config=subconfig,
                depth=self.determine_config_depth(parameter),
            )
        return config

    def insert_wandb_sweep_specific_keys(self, config, depth):
        """
        Inserts "parameters" and "values" keys to conform with sweep configuration.
        https://docs.wandb.ai/guides/sweeps/configuration
        This function will ignore dicts in parameter values as it assumes that dicts
        are already conforming with configuration.
        """
        if depth > 1:
            new_config = {}
            for key, value in config.items():
                new_config[key] = self.insert_wandb_sweep_specific_keys(
                    value, depth - 1
                )
            return {"parameters": new_config}

        if not isinstance(config, dict):
            return {"value": config}
        else:
            return config

    @staticmethod
    def determine_config_depth(parameter):
        """How many insertions of "parameters" keys to perform for each config section."""
        if parameter in {"callbacks"}:
            return 3
        else:
            return 2

    def dispatch(self):
        wandb.agent(
            project=self.project,
            sweep_id=self.sweep_id,
            function=self.wandb_main_wrapper,
            count=self.count,
        )

    @full_traceback
    def wandb_main_wrapper(self):
        """Takes config from sweep controller, processes it and calls main with it."""
        with wandb.init() as wandb_run:
            config = dict(wandb_run.config)
            self.expand_string_configs(config)
            config.update(self.args)
            self.main(config)

    def expand_string_configs(self, config):
        """
        If a sweep config yaml file had strings as values, this function
        will attempt to parse them into objects, for example the following:

            scheduling:
              value: "{0: 10}"

        will be parsed into:

            scheduling:
              value: {0: 10}

        This function is needed because wandb sweep errors on dict parameters.
        """
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = yaml.safe_load(value)
            elif isinstance(value, dict):
                self.expand_string_configs(config[key])


class ConfigDispenser:
    def __init__(self, main: Callable[[dict], None]):
        self.main = main
        self.args = None

    @staticmethod
    def is_sweep(config):
        return "method" in config and "parameters" in config

    def parser(self, parser: ArgumentParser) -> ArgumentParser:
        """Overwrite this method to extend default parser or build new one."""
        return parser

    def debug_config(self, config: dict) -> dict:
        """Overwrite this method to change config for debugging."""
        config["trainer"].update(
            dict(
                fast_dev_run=True,
                devices=None,
                accelerator=None,
            )
        )
        config["lightning_module"]["num_workers"] = 1
        del config["logger"]
        return config

    def update_config(self, config: dict) -> dict:
        """
        Overwrite this method to modify config before it gets dispensed.
        It isn't called when performing a sweep for consistency between runs.
        """
        return config

    @property
    def default_parser(self):
        parser = ArgumentParser(
            description="Launch agent to explore the sweep provided by the yaml file or a debug run"
        )
        parser.add_argument(
            "config_path",
            type=str,
            help="yaml config file",
        )
        parser.add_argument(
            "runs",
            default=1,
            type=int,
            help="How many runs for the agent to try",
            nargs="?",
        )
        parser.add_argument(
            "gpu",
            default=None,
            type=int,
            help="Id of gpu to run on",
            nargs="?",
        )
        parser.add_argument(
            "--debug",
            "-d",
            action="store_true",
            help="Run debug hooks",
        )
        return parser

    def launch(self, config_path=None, **args):
        """If at least config is not passed, args will be parsed from the cli."""
        if config_path is None:
            parser = self.parser(parser=self.default_parser)
            args = vars(parser.parse_args())
        else:
            args.update(config=config_path)

        config = yaml.safe_load(open(args["config_path"]))

        if self.is_sweep(config):
            return WandbSweepDispatcher(self.main, config, args).dispatch()

        if args.get("debug"):
            config = self.debug_config(config)
        if gpu := args.get("gpu"):
            config["trainer"]["accelerator"] = "gpu"
            config["trainer"]["devices"] = [gpu]
        config.update(args)
        config = self.update_config(config)
        self.main(config)

    __call__ = launch


class Hooker:
    """Skeleton for cooperative hooks inheritance.
    (Which is probably isn't useful for hooks to be honest)"""

    def __init__(self):
        self.on_before_main()
        self.main()
        self.on_after_main()

    def on_before_main(self):
        assert not hasattr(super(), "on_before_main")

    def main(self):
        assert not hasattr(super(), "main")

    def on_after_main(self):
        assert not hasattr(super(), "on_after_main")


class ConfigConstructorBase(abc.ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config.copy()

    @property
    def build_class(self):
        return build_class

    def main(self):
        """Example implementation, feel free to override."""
        lightning_module = self.build_lightning_module()
        datamodule = self.build_datamodule()
        trainer = self.build_trainer()
        trainer.fit(lightning_module, datamodule=datamodule)
        trainer.test(lightning_module, datamodule=datamodule)

    def lightning_candidates(self):
        return ()

    def datamodule_candidates(self):
        return ()

    def callback_candidates(self):
        return ()

    def build_lightning_module(self, lightning_candidates=()):
        lightning_candidates = list(lightning_candidates) + list(
            self.lightning_candidates()
        )
        lightning_config = self.config["lightning_module"]
        lightning_module = self.build_class(
            class_candidates=lightning_candidates,
            **lightning_config,
            model_config=self.config["model"],
            loss_config=self.config["loss"],
            optimizer_config=self.config["optimizer"],
        )
        return lightning_module

    def build_datamodule(self, datamodule_candidates=()):
        datamodule_candidates = list(datamodule_candidates) + list(
            self.datamodule_candidates()
        )
        datamodule = self.build_class(
            class_candidates=datamodule_candidates, **self.config["datamodule"]
        )
        return datamodule

    def build_callbacks(self, callback_candidates=()) -> dict:
        callback_candidates = list(callback_candidates) + list(
            self.callback_candidates()
        )
        callbacks = {}
        if callbacks_config := self.config.get("callbacks"):
            for callback_name, single_callback_config in callbacks_config.items():
                callback = self.build_class(
                    class_candidates=callback_candidates,
                    modules=[pl.callbacks],
                    **single_callback_config,
                    name=callback_name,
                )
                callbacks[callback_name] = callback
        return callbacks

    def build_logger(self):
        if logger_config := self.config.get("logger"):
            logger = self.build_class(
                modules=[pl.loggers],
                **logger_config,
                project=self.config.get("project"),
            )
            return logger

    def build_trainer(self, callback_candidates=()):
        trainer = pl.Trainer(
            **self.config["trainer"],
            callbacks=list(self.build_callbacks(callback_candidates).values()),
            logger=self.build_logger(),
        )
        return trainer
