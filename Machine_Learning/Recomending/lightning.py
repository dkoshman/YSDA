import torch

from my_ml_tools.utils import build_class


class RecommenderMixin:
    def build_model(self, model_config, model_classes):
        model_name = model_config.pop("name")
        model = build_class(model_name, model_config, class_candidates=model_classes)
        return model

    def configure_optimizers(self):
        optimizer_config = self.hparams["optimizer_config"]
        # Need to copy, otherwise parameters will be saved to hparams
        optimizer_config = optimizer_config.copy()
        optimizer_config["params"] = self.parameters()
        optimizer_name = optimizer_config.pop("name")
        optimizer = build_class(
            optimizer_name,
            optimizer_config,
            modules_to_try_to_import_from=[torch.optim],
        )
        return optimizer
