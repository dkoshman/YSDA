import pytorch_lightning as pl


class ConvenientCheckpointLogCallback(pl.callbacks.ModelCheckpoint):
    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)
        if not self.best_model_path:
            self.save_checkpoint(trainer)
        trainer.logger.log_text(
            key="best checkpoint", columns=["path"], data=[[self.best_model_path]]
        )
