import sys

from argparse import ArgumentParser
from typing import Callable, Any

import pytorch_lightning as pl
import wandb
import yaml

from pytorch_lightning import loggers as pl_loggers

from my_tools.utils import full_traceback, BuilderMixin


class WandbSweepProcessor:
    @staticmethod
    def is_sweep(config):
        return "method" in config and "parameters" in config

    def initialize_sweep(self, config_path, config=None):
        """Returns sweep id if it is a field in config, otherwise initializes new sweep."""
        if config is None:
            config = yaml.safe_load(config_path)
        sweep_id = config.get("sweep_id")
        if sweep_id:
            print(f"Using sweep id {sweep_id} from config file", file=sys.stderr)
        else:
            config = self.preprocess_sweep_config(config)
            sweep_id = wandb.sweep(config)
            yaml.safe_dump({"sweep_id": sweep_id}, open(config_path, "a"))
            print("Added sweep id to config file", file=sys.stderr)
        return sweep_id

    def preprocess_sweep_config(self, config):
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
            parameters[parameter] = self._insert_wandb_sweep_specific_keys(
                config=subconfig,
                depth=self._determine_config_depth(parameter),
            )
        return config

    def _insert_wandb_sweep_specific_keys(self, config, depth):
        """
        Inserts "parameters" and "values" keys to conform with sweep configuration.
        https://docs.wandb.ai/guides/sweeps/configuration
        This function will ignore dicts in parameter values as it assumes that dicts
        are already conforming with configuration.
        """
        if depth > 1:
            new_config = {}
            for key, value in config.items():
                new_config[key] = self._insert_wandb_sweep_specific_keys(
                    value, depth - 1
                )
            return {"parameters": new_config}

        if not isinstance(config, dict):
            return {"value": config}
        else:
            return config

    @staticmethod
    def _determine_config_depth(parameter):
        """How many insertions of "parameters" keys to perform for each config section."""
        if parameter in {"callbacks"}:
            return 3
        else:
            return 2

    def postprocess_config(self, config):
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
                self.postprocess_config(config[key])


class ConfigDispenser(WandbSweepProcessor):
    """
    If config_path doesn't correspond to a sweep, then
    this class is basically just an identity callable.
    Otherwise, it attains the sweep_id, launches a
    wandb agent and passes it the provided function.
    """

    def __init__(
        self,
        config_path: str or None = None,
        runs_count: int = 1,
        **extra_config_kwargs,
    ):
        self.config_path = config_path
        self.runs_count = runs_count
        self.extra_config_kwargs = extra_config_kwargs
        self.sweep_id = None
        self.function: Callable[[dict], Any] or None = None

    def from_cli(self):
        cli_args = vars(self.cli_argument_parser.parse_args())
        self.__init__(**cli_args)

    @property
    def cli_argument_parser(self):
        parser = ArgumentParser(
            description="Launch agent to explore the sweep provided by the yaml file or a debug run"
        )
        parser.add_argument("config_path", type=str, help="yaml config file")
        parser.add_argument(
            "runs_count",
            default=1,
            type=int,
            help="How many runs for the agent to try",
            nargs="?",
        )
        parser.add_argument(
            "gpu_device_ids",
            default=None,
            type=int,
            help="Id of gpu to run on",
            nargs="?",
        )
        return parser

    def function_wrapper(self, config):
        self.postprocess_config(config)
        config.update(self.extra_config_kwargs)
        if (gpus := config.get("gpu_device_ids")) is not None:
            config["trainer"]["accelerator"] = "gpu"
            config["trainer"]["devices"] = gpus
        return self.function(config)

    @full_traceback
    def sweep_agent_wrapper(self):
        with wandb.init() as wandb_run:
            config = dict(wandb_run.config)
            self.function_wrapper(config)

    def launch(self, function: Callable[[dict], Any]):
        self.function = function
        config = yaml.safe_load(open(self.config_path))
        if not self.is_sweep(config):
            return self.function_wrapper(config)
        else:
            self.sweep_id = self.initialize_sweep(self.config_path, config)
            wandb.agent(
                project=config.get("project"),
                sweep_id=self.sweep_id,
                function=self.sweep_agent_wrapper,
                count=self.runs_count,
            )


class LightningConfigBuilder(BuilderMixin):
    def __init__(self, config: dict):
        """
        :param config: the parsed nested config dict with string and numeric values
        in the following format:
        BaseConfig := {class_name: ClassName, **init_kwargs}
        config = {
            datamodule: BaseConfig
            model: BaseConfig
            lightning_module: BaseConfig
            logger: BaseConfig # for logger class_name is optional, default WandbLogger
            trainer: BaseConfig # for trainer class_name is optional, default Trainer
            callbacks: {callback_class_name: callback_init_kwargs, ...}
        }
        """
        self.config = config.copy()
        self.datamodule_config = config.get("datamodule")
        self.model_config = config.get("model")
        self.loss_config = config.get("loss")
        self.lightning_config = config.get("lightning_module")
        self.optimizer_config = config.get("optimizer")
        self.logger_config = config.get("logger")
        if self.logger_config:
            if "class_name" not in self.logger_config:
                self.logger_config.update(class_name="WandbLogger")
            if "project" not in self.logger_config:
                self.logger_config.update(project=config.get("project"))
        self.callbacks_config = config.get("callbacks")
        self.trainer_config = config.get("trainer", {})
        if "class_name" not in self.trainer_config:
            self.trainer_config.update(class_name="Trainer")

    @property
    def module_candidates(self):
        return super().module_candidates + [pl.callbacks, pl_loggers]

    @property
    def class_candidates(self):
        return super().class_candidates + [pl.Trainer, pl_loggers.WandbLogger]

    def build_lightning_module(self) -> pl.LightningModule:
        return self.build_class(**self.lightning_config)

    def build_datamodule(self) -> pl.LightningDataModule:
        return self.build_class(**self.datamodule_config)

    def build_model(self):
        return self.build_class(**self.model_config)

    def build_logger(self) -> pl_loggers.logger.Logger:
        return self.build_class(**self.logger_config)

    def build_callbacks_dict(self) -> "dict[str, pl.Callback]":
        callbacks = {}
        if self.callbacks_config:
            for callback_name, callback_config in self.callbacks_config.items():
                callbacks[callback_name] = self.build_class(
                    class_name=callback_name, **callback_config
                )
        return callbacks

    def build_trainer(self) -> pl.Trainer:
        trainer = self.build_class(
            callbacks=list(self.build_callbacks_dict().values()),
            logger=self.build_logger(),
            **self.trainer_config,
        )
        return trainer
