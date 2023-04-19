import os

import pandas as pd
import pytorch_lightning as pl
import transformers
import wandb

from config import CONFIG
from data import (
    get_annotation_ground_truth_str_from_image_index,
    load_train_image_ids,
    build_dataloader,
    Split,
    Batch,
)
from metrics import benetech_score_string_prediction
from model import generate_token_strings, LightningModule
from utils import set_tokenizers_parallelism, set_torch_device_order_pci_bus


class MetricsCallback(pl.callbacks.Callback):
    def on_validation_batch_start(
        self, trainer, pl_module, batch: Batch, batch_idx, dataloader_idx=0
    ):
        predicted_strings = generate_token_strings(pl_module.model, images=batch.images)

        for expected_data_index, predicted_string in zip(
            batch.data_indices, predicted_strings, strict=True
        ):
            benetech_score = benetech_score_string_prediction(
                expected_data_index=expected_data_index,
                predicted_string=predicted_string,
            )
            wandb.log(dict(benetech_score=benetech_score))

        ground_truth_strings = [
            get_annotation_ground_truth_str_from_image_index(i)
            for i in batch.data_indices
        ]
        string_ids = [load_train_image_ids()[i] for i in batch.data_indices]
        strings_dataframe = pd.DataFrame(
            dict(
                string_ids=string_ids,
                ground_truth=ground_truth_strings,
                predicted=predicted_strings,
            )
        )
        wandb.log(dict(strings=wandb.Table(dataframe=strings_dataframe)))


class TransformersPreTrainedModelsCheckpointIO(pl.plugins.CheckpointIO):
    def __init__(
        self, pretrained_models: list[transformers.modeling_utils.PreTrainedModel]
    ):
        super().__init__()
        self.pretrained_models = pretrained_models

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        for pretrained_model in self.pretrained_models:
            pretrained_model.save_pretrained(path)

    def load_checkpoint(self, path, storage_options=None):
        self.pretrained_models = [
            pm.from_pretrained(path) for pm in self.pretrained_models
        ]

    def remove_checkpoint(self, path):
        os.remove(path)


def train():
    set_tokenizers_parallelism(False)
    set_torch_device_order_pci_bus()

    pl_module = LightningModule(CONFIG)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=CONFIG.training_directory,
        monitor="val_loss",
        save_top_k=CONFIG.save_top_k_checkpoints,
    )
    metrics_callback = MetricsCallback()

    logger = pl.loggers.WandbLogger(
        project=CONFIG.wandb_project_name, save_dir=CONFIG.training_directory
    )

    plugin = TransformersPreTrainedModelsCheckpointIO(
        [pl_module.model.processor, pl_module.model.encoder_decoder]
    )

    trainer = pl.Trainer(
        accelerator=CONFIG.accelerator,
        devices=CONFIG.devices,
        plugins=[plugin],
        callbacks=[model_checkpoint, metrics_callback],
        logger=logger,
        limit_train_batches=CONFIG.limit_train_batches,
        limit_val_batches=CONFIG.limit_val_batches,
    )

    trainer.fit(
        model=pl_module,
        train_dataloaders=build_dataloader(
            Split.train, pl_module.model.batch_collate_function
        ),
        val_dataloaders=build_dataloader(
            Split.val, pl_module.model.batch_collate_function
        ),
    )


if __name__ == "__main__":
    train()
