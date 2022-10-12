import argparse
import os
import wandb

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Configure a new sweep using provided yaml file and save sweep id to 'sweep_id' file",
        allow_abbrev=True,
    )
    parser.add_argument(
        "yaml",
        type=str,
        help="yaml config file",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    sweep_config = yaml.safe_load(open(args.yaml))
    wandb.login(key=os.environ["WANDB_API_KEY"])
    sweep_id = wandb.sweep(sweep_config, project="DSSM")
    with open("sweep_id", "w") as f:
        f.write(sweep_id)


if __name__ == "__main__":
    main()
