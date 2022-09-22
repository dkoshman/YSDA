import abc
import sys
import traceback

from argparse import ArgumentParser

import pytorch_lightning as pl
import wandb
import yaml

from my_ml_tools.utils import build_class


def _full_traceback(function):
    """Enables full traceback, useful as wandb agent truncates it."""

    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exception:
            print(traceback.print_exc(), file=sys.stderr)
            raise exception

    return wrapper


class ConfigDispenser:
    def __init__(self, main=None):
        self.main = main

    def build_parser(self, default_parser: ArgumentParser) -> ArgumentParser:
        """Overwrite this method to extend default parser or build new one."""
        return default_parser

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

    @staticmethod
    def is_sweep(config):
        return "method" in config and "parameters" in config

    def insert_wandb_sweep_specific_keys(self, config, depth):
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
    def determine_config_depth(parameter_name):
        if parameter_name in {
            "model",
            "optimizer",
            "trainer",
            "logger",
            "lightning_module",
        }:
            return 2
        elif parameter_name in {"callbacks"}:
            return 3
        else:
            raise ValueError(f"Unknown parameter name {parameter_name}")

    def preprocess_sweep_parameters(self, config):
        config_parameters = config["parameters"]
        for parameter_name, subconfig in config_parameters.items():
            config_parameters[parameter_name] = self.insert_wandb_sweep_specific_keys(
                config=subconfig,
                depth=self.determine_config_depth(parameter_name),
            )
        return config

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

    @staticmethod
    def read_config(config_file_path):
        config = yaml.safe_load(open(config_file_path))
        return config

    @property
    def _default_parser(self):
        parser = ArgumentParser(
            description="Launch agent to explore the sweep provided by the yaml file or a debug run"
        )
        parser.add_argument(
            "config",
            type=str,
            help="yaml sweep config",
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

    def parse_args(self, args):
        if not args:
            parser = self.build_parser(default_parser=self._default_parser)
            args = vars(parser.parse_args())
        self.args = args
        return self.args

    def launch(self, **args):
        args = self.parse_args(args)
        config = self.read_config(args["config"])

        if self.is_sweep(config):
            sweep_id = self.initialize_sweep(config, args["config"])
            wandb.agent(
                project=config["project"],
                sweep_id=sweep_id,
                function=self._main_dispatch_wrapper_for_wandb_agent,
                count=args["runs"],
            )
        else:
            self.dispatch_main(config)

    __call__ = launch

    @_full_traceback
    def _main_dispatch_wrapper_for_wandb_agent(self):
        with wandb.init() as wandb_run:
            config = dict(wandb_run.config)
            self.dispatch_main(config)

    def expand_string_configs(self, config):
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = yaml.safe_load(value)
            elif isinstance(value, dict):
                self.expand_string_configs(config[key])

    def dispatch_main(self, config):
        self.expand_string_configs(config)
        if self.args.get("debug"):
            config = self.debug_config(config)
        if gpu := self.args.get("gpu"):
            config["trainer"]["accelerator"] = "gpu"
            config["trainer"]["devices"] = [gpu]
        config.update(self.args)
        self.main(config)


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
    """
    Example of intended class usage:

    def main(self):
          lightning_module = self.build_lightning_module(type("MyLightningModuleClass"))
          datamodule = self.build_datamodule(type("MyDatamodule"))
          trainer = self.build_trainer()
          trainer.fit(lightning_module, datamodule=datamodule)
          trainer.test(lightning_module, datamodule=datamodule)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config.copy()

    def build_lightning_module(self, lightning_candidates):
        lightning_config = self.config["lightning_module"]
        lightning_name = lightning_config.pop("name")
        lightning_config["model_config"] = self.config["model"]
        lightning_config["optimizer_config"] = self.config["optimizer"]
        return build_class(lightning_name, lightning_config, lightning_candidates)

    def build_datamodule(self, datamodule_candidates):
        datamodule_config = self.config["datamodule"]
        datamodule_name = datamodule_config.pop("name")
        return build_class(datamodule_name, datamodule_config, datamodule_candidates)

    def build_callbacks(self, callback_candidates=()) -> dict:
        callbacks = {}
        if callbacks_config := self.config.get("callbacks"):
            for callback_name, single_callback_config in callbacks_config.items():
                callbacks[callback_name] = build_class(
                    callback_name,
                    single_callback_config,
                    class_candidates=callback_candidates,
                    modules_to_try_to_import_from=[pl.callbacks],
                )
        return callbacks

    def build_logger(self):
        if logger_config := self.config.get("logger"):
            logger_config["project"] = self.config.get("project")
            logger_name = logger_config.pop("name")
            logger = build_class(
                logger_name, logger_config, modules_to_try_to_import_from=[pl.loggers]
            )
            return logger

    def build_trainer(self, callback_candidates=()):
        trainer = pl.Trainer(
            **self.config["trainer"],
            callbacks=list(self.build_callbacks(callback_candidates).values()),
            logger=self.build_logger(),
        )
        return trainer
