import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, GradientAccumulationScheduler
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import SimpleProfiler

from my_ml_tools.utils import free_cuda
from my_ml_tools.wandb_wrapper import cli_agent_dispatch

from data import SparseDataModule
from lit_module import LitPMF


def main():
    cli_agent_dispatch(train)


def train(experiment, config):
    free_cuda()

    datamodule = SparseDataModule(batch_size=config.batch_size)
    lit_module = LitPMF(
        n_users=datamodule.train_dataset.shape[0],
        n_items=datamodule.train_dataset.shape[1],
    )

    logger = WandbLogger(experiment=experiment)
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir="local",
        max_epochs=config.max_epochs,
        accelerator="gpu",
        gpus=[config.gpu],
        callbacks=[
            GradientAccumulationScheduler(scheduling={0: 10}),
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
        ],
        amp_backend="apex",
        amp_level="O2",
        profiler=SimpleProfiler(dirpath="local", filename="perf_logs"),
    )

    trainer.fit(lit_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
