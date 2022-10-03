from my_tools.entrypoints import ConfigDispenser
from my_tools.utils import build_class

import baseline

from entrypoints import NonLitToLitAdapterRecommender
from movielens import MovielensDispatcher


class LitBaselineRecommender(NonLitToLitAdapterRecommender):
    def build_model(self):
        model = build_class(
            modules=[baseline],
            explicit_feedback=self.trainer.datamodule.train_explicit,
            **self.hparams["model_config"],
        )
        return model


@ConfigDispenser
def main(config):
    MovielensDispatcher(
        config=config,
        lightning_candidates=[LitBaselineRecommender],
    ).dispatch()


if __name__ == "__main__":
    main()
