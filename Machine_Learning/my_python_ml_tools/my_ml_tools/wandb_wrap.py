import sys
import traceback

from argparse import ArgumentParser
from typing import Callable

import wandb
import yaml

from .utils import free_cuda


class WandbCLIWrapper:
    def __init__(self, train_function: Callable[[dict], None] = None):
        if train_function:
            self.main = train_function

    def build_parser(self, default_parser: ArgumentParser) -> ArgumentParser:
        """Overwrite this method to extend default parser or build new one."""
        return default_parser

    def debug_config(self, config: dict) -> dict:
        """Overwrite this method to change config for debugging."""
        return config

    def initialize_sweep(self, config_file_path) -> str:
        """Returns sweep id if it is a field in config file, otherwise initializes new sweep."""
        config = self._read_config(config_file_path)
        sweep_id = config.get("sweep_id")
        if sweep_id:
            print(f"Using sweep id {sweep_id} from config file", file=sys.stderr)
        else:
            sweep_id = wandb.sweep(config)
            yaml.safe_dump({"sweep_id": sweep_id}, open(config_file_path, "a"))
            print("Added sweep id to config file", file=sys.stderr)
        return sweep_id

    def _read_config(self, config_file_path):
        sweep_config = yaml.safe_load(open(config_file_path))
        self.project = sweep_config["project"]
        return sweep_config

    @property
    def _default_parser(self):
        parser = ArgumentParser(
            description="Launch agent to explore the sweep provided by the yaml file"
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

    def _parse_args(self):
        parser = self.build_parser(default_parser=self._default_parser)
        args = parser.parse_args()
        self.cli_args = args
        return args

    def launch_agent(self):
        args = self._parse_args()
        sweep_id = self.initialize_sweep(args.config)
        wandb.agent(
            project=self.project,
            sweep_id=sweep_id,
            function=self._main_wrapper,
            count=args.runs,
        )

    def __call__(self):
        """To enable use of this class as decorator."""
        self.launch_agent()

    @staticmethod
    def _full_traceback(function):
        """Enables full traceback, useful as wandb agent truncates it."""

        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exception:
                print(traceback.print_exc(), file=sys.stderr)
                raise exception

        return wrapper

    @_full_traceback
    def _main_wrapper(self):
        free_cuda()
        debug = self.cli_args.debug
        with wandb.init(job_type="debug" if debug else "train") as wandb_run:
            config = dict(wandb_run.config)
            config.update(vars(self.cli_args))
            if debug:
                config = self.debug_config(config)
            self.main(config)

    def main(self, config: dict) -> None:
        raise NotImplementedError
