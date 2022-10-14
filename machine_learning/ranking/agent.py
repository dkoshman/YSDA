import pickle
import traceback
import sys

import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger

from loss import LambdaLoss
from model import DSSM, DSSMExtended
from module import DSSMModule
from optimizers import OptimizerBuilder


class Agent:
    def __init__(self, project_name, gpu, max_epochs):
        self.project_name = project_name
        self.gpu = gpu
        self.max_epochs = max_epochs

    def __call__(self, config=None):
        with wandb.init(
            project=self.project_name, job_type="train", config=config
        ) as experiment:
            try:
                self.run(experiment)
            except Exception:
                print(traceback.print_exc(), file=sys.stderr)
                exit(1)

    def run(self, experiment):
        config = experiment.config

        data = pickle.load(open(f"data/{config['datasets']}.pickle", "rb"))

        match config.model:
            case "dssm":
                dssm = DSSM(
                    hidden_dimensions=[len(data["vectorizer"].vocabulary_)]
                    + config.hidden_dimensions
                )
            case "dssm_extended":
                dssm = DSSMExtended(
                    hidden_dimensions=[len(data["vectorizer"].vocabulary_)]
                    + config.hidden_dimensions,
                    head_dimensions=[
                        1
                        + data["train_dataset"][0]["pairwise_numeric_features"].shape[
                            -1
                        ]
                    ]
                    + config.head_dimensions,
                )
            case _:
                raise RuntimeError("Invalid model specification")

        loss_fn = LambdaLoss()

        optimizers_config = OptimizerBuilder(
            optimizer_name=config.optimizer,
            scheduler_name=config.lr_scheduler,
            min_lr=config.lr_range[0],
            max_lr=config.lr_range[1],
        )

        dssm_module = DSSMModule(
            train_dataset=data["train_dataset"],
            val_dataset=data["val_dataset"],
            test_dataset=data["test_dataset"],
            vectorizer=data["vectorizer"],
            dssm=dssm,
            loss_fn=loss_fn,
            optimizers_config=optimizers_config,
            num_workers=config.num_workers,
        )

        logger = WandbLogger(experiment=experiment, log_model=True)

        checkpoint_callback = ModelCheckpoint(monitor="val_ndcg", mode="max")

        trainer = Trainer(
            logger=logger,
            max_epochs=self.max_epochs,
            accelerator="gpu",
            gpus=[self.gpu],
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval="step"),
                EarlyStopping(monitor="val_ndcg", mode="max", patience=10),
            ],
        )

        trainer.fit(dssm_module)
        trainer.test(ckpt_path=checkpoint_callback.best_model_path)
