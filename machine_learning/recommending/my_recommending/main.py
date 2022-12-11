from my_recommending.entrypoint import RecommendingConfigDispenser, fit
from my_recommending.utils import update_from_base_config, wandb_context_manager


def cross_validation_configs(config):
    for i in [1, 2, 3, 4, 5]:
        config["datamodule"].update(
            dict(
                train_explicit_file=f"u{i}.base",
                val_explicit_file=f"u{i}.test",
                test_explicit_file=f"u{i}.test",
            )
        )
        yield config


def main(config):
    config = update_from_base_config(
        config, base_config_file="configs/base_config.yaml"
    )
    if not config["wandb_save"]:
        if "WandbCheckpointCallback" in config.get("callbacks", []):
            del config["callbacks"]["WandbCheckpointCallback"]
    if config["cross_validate"]:
        if "CatBoostMetrics" in config.get("callbacks", []):
            del config["callbacks"]["CatBoostMetrics"]
        with wandb_context_manager(config):
            for config in cross_validation_configs(config):
                fit(config)
    else:
        fit(config)


if __name__ == "__main__":
    dispenser = RecommendingConfigDispenser()
    dispenser.from_cli()
    dispenser.launch(main)
