from pytorch_lightning.callbacks import Callback


class ConvenientCheckpointLogCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        super().on_fit_end(trainer, pl_module)
        if logger := trainer.logger:
            checkpoint_path = trainer.checkpoint_callback.best_model_path
            logger.log_text(
                key="best checkpoint", columns=["path"], data=[[checkpoint_path]]
            )
