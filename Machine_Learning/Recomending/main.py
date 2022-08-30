import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from my_ml_tools.wandb_wrap import WandbCLIWrapper

from data import SparseDataModule
from lit_module import LitProbabilityMatrixFactorization


def build_trainer(config):
    callbacks = [
        GradientAccumulationScheduler(
            scheduling={0: config["n_batches_for_gradient_accumulation"]}
        )
    ]

    early_stopping_config = config["early_stopping"]
    callbacks.append(
        EarlyStopping(
            monitor=early_stopping_config["monitor"],
            mode=early_stopping_config["mode"],
            min_delta=early_stopping_config["min_delta"],
            patience=early_stopping_config["patience"],
        )
    )
    if (swa_config := config["stochastic_weight_averaging"])["enable"]:
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=swa_config["swa_epoch_start"],
                swa_lrs=swa_config["swa_lrs"],
                annealing_epochs=swa_config["annealing_epochs"],
                annealing_strategy=swa_config["annealing_strategy"],
            )
        )
    amp_backend = config["automatic_mixed_precision"]["amp_backend"]
    amp_level = (
        config["automatic_mixed_precision"]["amp_level"]
        if amp_backend == "apex"
        else None
    )
    trainer = pl.Trainer(
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
    return trainer


class RecommendingWandbCLIWrapper(WandbCLIWrapper):
    def debug_config(self, config):
        config["check_val_every_n_epoch"] = 1
        return config


@RecommendingWandbCLIWrapper
def main(config):
    datamodule = SparseDataModule(
        batch_size=config["batch_size"], num_workers=config["num_workers"]
    )

    lit_module = LitProbabilityMatrixFactorization(
        n_users=datamodule.train_dataset.shape[0],
        n_items=datamodule.train_dataset.shape[1],
        latent_dimension=config["latent_dimension"],
        weight_decay=config["weight_decay"],
        optimizer_kwargs=config["optimizer_kwargs"],
    )

    trainer = build_trainer(config)

    trainer.fit(lit_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
