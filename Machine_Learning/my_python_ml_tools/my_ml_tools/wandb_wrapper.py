from typing import Callable

import argparse
import wandb
import yaml


def cli_initialize_sweep(env_variable_for_sweep_id="WANDB_SWEEP_ID"):
    parser = argparse.ArgumentParser(
        description="Configure a new sweep using provided yaml file"
    )
    parser.add_argument("config", type=str, help="yaml config file")
    args = parser.parse_args()
    sweep_id = initialize_sweep(args.config, )

    if env_variable_for_sweep_id:
        import os

        os.environ[env_variable_for_sweep_id] = sweep_id
        print(f"Sweep id saved to {env_variable_for_sweep_id} environment variable")


def initialize_sweep(config_file_path):
    sweep_config = yaml.safe_load(open(config_file_path))
    wandb.login()
    sweep_id = wandb.sweep(sweep_config)
    print(f"Sweep initialized, sweep_id:\n{sweep_id}")
    return sweep_id


def cli_agent_dispatch(function: Callable[[dict], None]) -> None:
    """
    Intended to run as part of script, parses command line arguments
    and launches a wandb agent to perform a sweep. The agent then runs
    the provided function with config kwargs as argument

    :param function: the function that performs training, it will be provided a config
    """
    parser = argparse.ArgumentParser(
        description="Launch agent to explore the sweep provided by the yaml file",
        allow_abbrev=True,
    )
    parser.add_argument(
        "sweep_id",
        type=str,
        help="wandb sweep id",
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
    args = vars(parser.parse_args())

    agent_dispatch(
        function,
        sweep_id=args["sweep_id"],
        runs_count=args["runs"],
        common_run_kwargs=args,
    )


def agent_dispatch(
        function: Callable[[dict], None],
        sweep_id: str,
        runs_count: int = 1,
        common_run_kwargs: dict = None,
):
    """
    Launches wandb agent and passes it provided function

    :param function: training function
    :param sweep_id: id of wandb sweep
    :param runs_count: number of runs for the agent
    :param common_run_kwargs: these kwargs will be passed to the function
    """
    wrapped_function = _WandbAgentFunctionWrapper(
        common_run_kwargs=common_run_kwargs or {}, function=function
    )
    wandb.agent(
        sweep_id,
        function=wrapped_function,
        count=runs_count,
    )


class _WandbAgentFunctionWrapper:
    def __init__(self, common_run_kwargs, function):
        self.common_run_kwargs = common_run_kwargs
        self.function = function

    def __call__(self, config):
        with wandb.init(dir="./local/wandb", config=config) as wandb_run:
            config = wandb_run.config
            config.update(self.common_run_kwargs)
            self.function(config)
