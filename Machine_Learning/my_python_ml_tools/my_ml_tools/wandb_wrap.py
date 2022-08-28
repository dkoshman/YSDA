import argparse
import sys

import wandb
import yaml

from .utils import free_cuda


class WandbCLIWrapper:
    @staticmethod
    def initialize_sweep(config_file_path):
        """Returns sweep id if it is a field in config file, otherwise initializes new sweep."""
        sweep_config = yaml.safe_load(open(config_file_path))
        sweep_id = sweep_config.get("sweep_id")
        if sweep_id:
            print(f"Using sweep id {sweep_id} from config file", file=sys.stderr)
        else:
            sweep_id = wandb.sweep(sweep_config)
            yaml.safe_dump({"sweep_id": sweep_id}, open(config_file_path, "a"))
            print("Added sweep id to config file", file=sys.stderr)
        return sweep_id

    @property
    def default_parser(self):
        parser = argparse.ArgumentParser(
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
        return parser

    def parser(self):
        return self.default_parser

    def launch_agent(self):
        parser = self.parser()
        args = parser.parse_args()
        self.cli_args = args
        sweep_id = self.initialize_sweep(args.config)
        wandb.agent(
            sweep_id=sweep_id,
            function=self._main_wrapper,
            count=args.runs,
        )

    def _main_wrapper(self):
        free_cuda()
        with wandb.init() as wandb_run:
            self.experiment = wandb_run
            try:
                self.main(wandb_run, self.cli_args)
            except Exception as exception:
                import traceback

                print(traceback.print_exc(), file=sys.stderr)
                raise exception

    def main(self, experiment, cli_args):
        raise NotImplementedError


class WandbCLIDecorator(WandbCLIWrapper):
    def __init__(self, train_function):
        self.main = train_function

    def __call__(self):
        self.launch_agent()
