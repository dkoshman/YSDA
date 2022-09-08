import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from my_ml_tools.wandb_wrap import WandbCLIWrapper

from data import PMFDataModule, SLIMDataModule
from lit_module import LitProbabilityMatrixFactorization, LitSLIM
from model import (
    ConstrainedProbabilityMatrixFactorization,
    ProbabilityMatrixFactorization,
    SLIM,
)


class RecommendingWandbCLIWrapper(WandbCLIWrapper):
    def build_datamodule(self, config):
        if config["model"] != "slim":
            self.datamodule = PMFDataModule(
                batch_size=config["batch_size"], num_workers=config["num_workers"]
            )
            self.n_users = self.datamodule.train_dataset.shape[0]
            self.n_items = self.datamodule.train_dataset.shape[1]
        else:
            self.datamodule = SLIMDataModule(
                batch_size=config["batch_size"], num_workers=config["num_workers"]
            )
        return self.datamodule

    def build_model(self, config):
        if config["model"] == "probabilistic_matrix_factorization":
            self.model = ProbabilityMatrixFactorization(
                n_users=self.n_users,
                n_items=self.n_items,
                latent_dimension=config["latent_dimension"],
                regularization_lambda=config["weight_decay"],
            )
        elif config["model"] == "constrained_probabilistic_matrix_factorization":
            self.model = ConstrainedProbabilityMatrixFactorization(
                n_users=self.n_users,
                n_items=self.n_items,
                latent_dimension=config["latent_dimension"],
                regularization_lambda=config["weight_decay"],
                implicit_feedback=self.datamodule.train_dataset.implicit_feedback,
            )
        elif config["model"] == "slim":
            self.model = SLIM(
                explicit_feedback=self.datamodule.dataset.explicit_train,
                l2_coefficient=config["l2_coefficient"],
                l1_coefficient=config["l1_coefficient"],
                density=config["density"],
            )
        else:
            raise ValueError(f"Unknown model type {config['model']}")

        return self.model

    def build_lightning_module(self, model, config):
        if config["model"] != "slim":
            self.lit_module = LitProbabilityMatrixFactorization(
                model=model,
                optimizer_kwargs=config["optimizer_kwargs"],
            )
        else:
            self.lit_module = LitSLIM(
                model=model,
                patience=config["patience"],
            )
        return self.lit_module

    def build_callbacks(self, config):
        early_stopping_config = config["early_stopping"]
        swa_config = config["stochastic_weight_averaging"]

        callbacks = [
            GradientAccumulationScheduler(
                scheduling={0: config["n_batches_for_gradient_accumulation"]}
            ),
            EarlyStopping(
                monitor=early_stopping_config["monitor"],
                mode=early_stopping_config["mode"],
                min_delta=early_stopping_config["min_delta"],
                patience=early_stopping_config["patience"],
            ),
        ]

        if swa_config["enable"]:
            callbacks.append(
                StochasticWeightAveraging(
                    swa_epoch_start=swa_config["swa_epoch_start"],
                    swa_lrs=swa_config["swa_lrs"],
                    annealing_epochs=swa_config["annealing_epochs"],
                    annealing_strategy=swa_config["annealing_strategy"],
                )
            )

        return callbacks

    def build_trainer(self, config):
        if config["model"] != "slim":
            callbacks = self.build_callbacks(config)

            amp_config = config["automatic_mixed_precision"]
            amp_backend = amp_config["amp_backend"]
            amp_level = (
                amp_config["amp_level"] if amp_config["amp_backend"] == "apex" else None
            )
            self.trainer = pl.Trainer(
                logger=WandbLogger(),
                check_val_every_n_epoch=config["check_val_every_n_epoch"],
                default_root_dir="local",
                max_epochs=config["max_epochs"],
                accelerator="gpu",
                gpus=[config["gpu"]],
                callbacks=callbacks,
                amp_backend=amp_backend,
                amp_level=amp_level,
            )
        else:
            # TODO: trainer section in config
            # trainer_kwargs = {}
            self.trainer = pl.Trainer(
                # logger=WandbLogger(),
                # log_every_n_steps=1,
                val_check_interval=3,
                default_root_dir="local",
                num_sanity_val_steps=0,
                max_epochs=1,
                reload_dataloaders_every_n_epochs=1,
                # accelerator='mps', devices=1,
                # accelerator="gpu",
                # gpus=config["gpus"],
            )
        return self.trainer

    def debug_config(self, config):

        if config["model"] == "probabilistic_matrix_factorization":
            print("Overwriting config", file=open(2, "a", closefd=False))
            config["check_val_every_n_epoch"] = 1
            config["max_epochs"] = 1

        return config

    def on_fit_end(self, config):
        if not config["debug"]:
            checkpoint_path = self.trainer.checkpoint_callback.best_model_path
            wandb.log(
                {
                    "best checkpoint path": wandb.Html(
                        f"<span> {checkpoint_path} </span>"
                    )
                }
            )

    def main(self, config):
        datamodule = self.build_datamodule(config)
        model = self.build_model(config)
        lit_module = self.build_lightning_module(model, config)
        trainer = self.build_trainer(config)
        trainer.fit(lit_module, datamodule=datamodule)
        self.on_fit_end(config)


if __name__ == "__main__":
    # TODO: config parser and builder
    RecommendingWandbCLIWrapper().launch_agent()
