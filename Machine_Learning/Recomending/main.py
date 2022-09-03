import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from my_ml_tools.wandb_wrap import WandbCLIWrapper

from data import SparseDataModule
from lit_module import LitProbabilityMatrixFactorization
from model import (
    ConstrainedProbabilityMatrixFactorization,
    ProbabilityMatrixFactorization,
)


class RecommendingWandbCLIWrapper(WandbCLIWrapper):
    def build_datamodule(self, config):
        self.datamodule = SparseDataModule(
            batch_size=config["batch_size"], num_workers=config["num_workers"]
        )
        self.n_users = self.datamodule.train_dataset.shape[0]
        self.n_items = self.datamodule.train_dataset.shape[1]
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
        else:
            raise ValueError(f"Unknown model type {config['model']}")

        return self.model

    def build_lightning_module(self, model, config):
        self.lit_module = LitProbabilityMatrixFactorization(
            model=model,
            n_items=self.n_items,
            optimizer_kwargs=config["optimizer_kwargs"],
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
        return self.trainer

    # def debug_config(self, config):
    #
    #     if config["model"] == "probabilistic_matrix_factorization":
    #         print("Overwriting config", file=open(2, "a", closefd=False))
    #         config["check_val_every_n_epoch"] = 1
    #         config["max_epochs"] = 1
    #
    #     return config

    def on_fit_end(self):
        checkpoint_path = self.trainer.checkpoint_callback.best_model_path
        wandb.log(
            {"best checkpoint path": wandb.Html(f"<span> {checkpoint_path} </span>")}
        )

    def main(self, config):
        datamodule = self.build_datamodule(config)
        model = self.build_model(config)
        lit_module = self.build_lightning_module(model, config)
        trainer = self.build_trainer(config)
        trainer.fit(lit_module, datamodule=datamodule)

        self.on_fit_end()


if __name__ == "__main__":
    RecommendingWandbCLIWrapper().launch_agent()
