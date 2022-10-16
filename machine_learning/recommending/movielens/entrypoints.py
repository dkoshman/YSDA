import abc
import torch
import wandb
import yaml

from my_tools.entrypoints import ConfigConstructorBase, ConfigDispenser

from . import callbacks as movielens_callbacks
from . import lit
from .. import callbacks, metrics


class MovieLensConfigDispenser(ConfigDispenser):
    def update_config(self, config, config_mixin_path="movielens_mixin.yaml"):
        config_mixin = yaml.safe_load(config_mixin_path)
        for key in config_mixin:
            if key not in config:
                config[key] = {}
            config[key].update(config_mixin[key])
        return config


class MovieLensDispatcher(ConfigConstructorBase, abc.ABC):
    @property
    def module_candidates(self):
        return super().module_candidates + [
            callbacks,
            lit,
            metrics,
            movielens_callbacks,
        ]

    def build_lightning_module(self):
        lightning_module = self.build_class(
            datamodule_config=self.config["datamodule"],
            model_config=self.config["model"],
            loss_config=self.config.get("loss"),
            optimizer_config=self.config.get("optimizer"),
            **self.config["lightning_module"],
        )
        return lightning_module

    def main(self):
        lightning_module = self.build_lightning_module()
        trainer = self.build_trainer()
        trainer.fit(lightning_module)
        trainer.test(lightning_module)

    def update_tune_data(self):
        self.config["datamodule"].update(
            dict(
                train_explicit_file="u1.base",
                val_explicit_file="u1.base",
                test_explicit_file="u1.test",
            )
        )

    def tune(self):
        self.update_tune_data()
        if wandb.run is None and self.config.get("logger") is not None:
            with wandb.init(project=self.config.get("project"), config=self.config):
                return self.main()
        else:
            return self.main()

    def test_datasets_iter(self):
        for i in [2, 3, 4, 5]:
            self.config["datamodule"].update(
                dict(
                    train_explicit_file=f"u{i}.base",
                    val_explicit_file=f"u{i}.base",
                    test_explicit_file=f"u{i}.test",
                )
            )
            yield

    def test(self):
        with wandb.init(project=self.config["project"], config=self.config):
            for _ in self.test_datasets_iter():
                self.main()

    def dispatch(self):
        if self.config.get("stage") == "test":
            self.test()
        else:
            self.tune()
