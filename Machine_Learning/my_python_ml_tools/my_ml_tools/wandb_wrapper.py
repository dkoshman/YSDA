from typing import Callable

import argparse
import wandb
import yaml


def cli_agent_dispatch(
    function: Callable[[wandb.sdk.wandb_run.Run, dict], None]
) -> None:
    """
    Intended to run as part of script, parses command line arguments
    and launches a wandb agent to perform a sweep. The agent then runs
    the provided function

    :param function: the function that performs training,
        it will be provided a wandb Run instance and config dict
    """
    parser = argparse.ArgumentParser(
        description="Launch agent to explore the sweep provided by the yaml file",
        allow_abbrev=True,
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
    args = vars(parser.parse_args())

    sweep_id = initialize_sweep(args["config"])

    agent_dispatch(
        function,
        sweep_id=sweep_id,
        runs_count=args["runs"],
        common_run_kwargs=args,
    )


def initialize_sweep(config_file_path):
    """Returns sweep id if it is a field in config file, otherwise initializes new sweep."""
    sweep_config = yaml.safe_load(open(config_file_path))
    sweep_id = sweep_config.get("sweep_id")
    if not sweep_id:
        sweep_id = wandb.sweep(sweep_config)
        yaml.safe_dump({"sweep_id": sweep_id}, open(config_file_path, "a"))
    return sweep_id


def agent_dispatch(
    function: Callable[[wandb.sdk.wandb_run.Run, dict], None],
    sweep_id: str,
    runs_count: int = 1,
    common_run_kwargs: dict = None,
):
    """
    Launches wandb agent and passes it provided function

    :param function: training function which takes a wandb Run and config
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

    def __call__(self):
        with wandb.init() as wandb_run:
            config = wandb_run.config
            config.update(self.common_run_kwargs)
            self.function(experiment=wandb_run, config=config)
