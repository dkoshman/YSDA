import argparse
import os
import pathlib
import subprocess

import torch
import wandb

from agent import Agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch agents to explore current sweep. To configure a new sweep, see sweep.py"
        "Output of child agents will be written to 'pipes' directory.",
        allow_abbrev=True,
    )
    parser.add_argument(
        "runs",
        default=1,
        type=int,
        help="how many runs for each agent to try, default 1",
        nargs="?",
    )
    parser.add_argument(
        "gpu",
        default=0,
        type=int,
        help=f"Id of gpu in range 0..{torch.cuda.device_count() - 1} to train on, default 0",
        nargs="?",
    )
    parser.add_argument(
        "agents",
        default=1,
        type=int,
        help="how many parallel agents to launch, default 1",
        nargs="?",
    )
    parser.add_argument(
        "-m",
        "--max-epochs",
        default=100,
        type=int,
        help=f"max epochs for each run, default 100",
        nargs=1,
        dest="max_epochs",
    )
    args = parser.parse_args()
    return args


def determine_sweep():
    with open("sweep_id") as file:
        sweep_id = file.read()

    wandb.login(key=os.environ["WANDB_API_KEY"])
    return sweep_id


def main():
    # If you get error 'Too many open files', try setting `ulimit -n 64000`
    args = parse_args()

    if args.agents > 1:
        pathlib.Path("pipes").mkdir(exist_ok=True)
        with open(
            f"pipes/out_agent_{args.agents}_gpu_{args.gpu}_pid_{os.getpid()}", "wb"
        ) as out:
            print(f"Child process logs will be written to {out.name}")
            subprocess.Popen(
                f"python main.py {args.runs} {args.gpu} {args.agents - 1}",
                shell=True,
                stdout=out,
                stderr=out,
            )

    sweep_id = determine_sweep()
    wandb.agent(
        sweep_id,
        project="DSSM",
        function=Agent("DSSM", gpu=args.gpu, max_epochs=args.max_epochs),
        count=args.runs,
    )


if __name__ == "__main__":
    main()
