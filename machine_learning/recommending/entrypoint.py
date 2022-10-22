from my_tools.entrypoints import LightningConfigBuilder

from . import callbacks, metrics, models
from .movielens import callbacks as movielens_callbacks
from .movielens import lit as movielens_lit


class RecommendingBuilder(LightningConfigBuilder):
    @property
    def module_candidates(self):
        return super().module_candidates + [
            callbacks,
            metrics,
            movielens_callbacks,
            models,
            movielens_lit,
        ]

    def build_lightning_module(self):
        lightning_module = self.build_class(
            datamodule_config=self.datamodule_config,
            model_config=self.model_config,
            loss_config=self.loss_config,
            optimizer_config=self.optimizer_config,
            **self.lightning_config,
        )
        return lightning_module
