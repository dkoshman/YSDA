import pytorch_lightning as pl

from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import SimpleProfiler

from my_ml_tools.wandb_wrap import WandbCLIDecorator

from data import SparseDataModule
from lit_module import LitProbabilityMatrixFactorization


@WandbCLIDecorator
def main(experiment, cli_args):
    config = experiment.config

    datamodule = SparseDataModule(batch_size=config.batch_size)

    lit_module = LitProbabilityMatrixFactorization(
        n_users=datamodule.train_dataset.shape[0],
        n_items=datamodule.train_dataset.shape[1],
        latent_dimension=config.latent_dimension,
        weight_decay=config.weight_decay,
        optimizer_kwargs=config.optimizer_kwargs,
    )

    callbacks = [
        GradientAccumulationScheduler(
            scheduling={0: config.n_batches_for_gradient_accumulation}
        ),
        EarlyStopping(
            monitor=config.early_stopping["monitor"],
            mode=config.early_stopping["mode"],
            min_delta=config.early_stopping["min_delta"],
            patience=config.early_stopping["patience"],
        ),
    ]
    if (swa_config := config.stochastic_weight_averaging)["enable"]:
        callbacks.append(
            StochasticWeightAveraging(
                swa_epoch_start=swa_config["swa_epoch_start"],
                swa_lrs=swa_config["swa_lrs"],
                annealing_epochs=swa_config["annealing_epochs"],
                annealing_strategy=swa_config["annealing_strategy"],
            )
        )
    amp_backend = config.automatic_mixed_precision["amp_backend"]
    amp_level = (
        config.automatic_mixed_precision["amp_level"] if amp_backend == "apex" else None
    )
    trainer = pl.Trainer(
        logger=WandbLogger(experiment=experiment),
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        default_root_dir="local",
        max_epochs=config.max_epochs,
        accelerator="gpu",
        gpus=[cli_args.gpu],
        callbacks=callbacks,
        profiler=SimpleProfiler(dirpath="local", filename="perf_logs"),
        amp_backend=amp_backend,
        amp_level=amp_level,
    )

    trainer.fit(lit_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
