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
        config["trainer"]["devices"] = None
        config["trainer"]["accelerator"] = None
        return config

    @staticmethod
    def is_sweep(config):
        return "method" in config and "parameters" in config

    def initialize_sweep(self, config, config_file_path):
        """Returns sweep id if it is a field in config, otherwise initializes new sweep."""
        sweep_id = config.get("sweep_id")
        if sweep_id:
            print(f"Using sweep id {sweep_id} from config file", file=sys.stderr)
        else:
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
            default=0,
            type=int,
            help="Id of gpu to run on",
            nargs="?",
        )
        parser.add_argument(
            "--debug",
            "-d",
            action="store_true",
            help="Run debug hooks and don't associate this run with the sweep",
        )
        return parser

    def parse_args(self, args):
        if not args:
            parser = self.build_parser(default_parser=self._default_parser)
            args = vars(parser.parse_args())
        self.args = args
        return self.args

    def __call__(self, **args):
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

    launch = __call__

    @_full_traceback
    def _main_dispatch_wrapper_for_wandb_agent(self):
        with wandb.init() as wandb_run:
            config = dict(wandb_run.config)
            self.dispatch_main(config)

    def dispatch_main(self, config):
        if self.args.get("debug"):
            config = self.debug_config(config)
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lightning_module = self.build_lightning_module()
        self.callbacks = self.build_callbacks()
        self.logger = self.build_logger()
        self.trainer = self.build_trainer()

    @abc.abstractmethod
    def lightning_module_candidates(self) -> list[type]:
        ...

    def callback_class_candidates(self) -> list[type]:
        return []

    def build_lightning_module(self):
        lightning_classes = self.lightning_module_candidates()
        lightning_config = self.config["lightning_module"]
        lightning_name = lightning_config.pop("name")
        lightning_config["model_config"] = self.config["model"]
        lightning_config["optimizer_config"] = self.config["optimizer"]

        for lightning_class in lightning_classes:
            if lightning_name == lightning_class.__name__:
                lightning_module = lightning_class(**lightning_config)
                break
        else:
            raise ValueError(
                f"Lightning module {lightning_name} not found among classes:\n"
                f"{lightning_classes}"
            )
        return lightning_module

    def build_callbacks(self):
        callback_classes = self.callback_class_candidates()
        callbacks = {}
        if callbacks_config := self.config.get("callbacks"):
            for callback_name, single_callback_config in callbacks_config.items():
                callbacks[callback_name] = build_class(
                    callback_name,
                    single_callback_config,
                    class_candidates=callback_classes,
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

    def build_trainer(self):
        trainer_config = self.config["trainer"]
        trainer = pl.Trainer(
            **trainer_config,
            callbacks=list(self.callbacks.values()),
            logger=self.logger,
        )
        return trainer
