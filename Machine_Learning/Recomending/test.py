import wandb

from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger

from my_ml_tools.wandb_wrap import WandbCLIWrapper

from data import PMFDataModule, SLIMDataModule
from lit_module import LitProbabilityMatrixFactorization, LitSLIM
from model import (
    ConstrainedProbabilityMatrixFactorization,
    ProbabilityMatrixFactorization,
    SLIM,
)

if __name__ == "__main__":
    print(wandb.__version__)
