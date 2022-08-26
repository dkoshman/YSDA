import argparse
import yaml
import wandb


class WandbAgentDispatcher:
    def cli_agent_dispatch(self, function):
        args = self.parse_args()
        self.launch_agent(
            function,
            config_file_name=args.config,
            runs_count=args.runs,
            common_run_kwargs=dict(gpu=args.gpu),
        )

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(
            description="Launch agent to explore the sweep provided by the yaml file",
            allow_abbrev=True,
        )
        parser.add_argument(
            "config",
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
            default=0,
            type=int,
            help=f"Id of gpu to run on",
            nargs="?",
        )
        args = parser.parse_args()
        return args

    @staticmethod
    def launch_agent(
        function,
        config_file_name=None,
        runs_count=1,
        common_run_kwargs=None,
    ):
        sweep_config = yaml.safe_load(open(config_file_name))
        sweep_id = sweep_config.get("sweep_id")
        if not sweep_id:
            print("sweep_id not found in config")
            sweep_id = wandb.sweep(sweep_config)
            yaml.dump(data={"sweep_id": sweep_id}, stream=open(config_file_name, "a"))
            print(
                f"Generated new sweep_id {sweep_id} and added to config {config_file_name}"
            )

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
