import pytorch_lightning as pl
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning.profiler import SimpleProfiler

from my_ml_tools.utils import free_cuda
from my_ml_tools.wandb_wrapper import cli_agent_dispatch

from data import SparseDataset
from lit_module import LitPMF


def main():
    cli_agent_dispatch(train)


def train(config):
    free_cuda()

    trainer = pl.Trainer(
        default_root_dir="local",
        max_epochs=config.max_epochs,
        accelerator="gpu",
        gpus=config.gpus,
        callbacks=[GradientAccumulationScheduler(scheduling={0: 10})],
        amp_backend="apex",
        amp_level="O2",
        profiler=SimpleProfiler(dirpath="local", filename="perf_logs"),
    )

    explicit_feedback, implicit_feedback = get_feedback()
    dataset = SparseDataset(
        explicit_feedback=explicit_feedback, implicit_feedback=implicit_feedback
    )

    lit_module = LitPMF(
        dataset=dataset,
        n_users=explicit_feedback.shape[0],
        n_items=explicit_feedback.shape[1],
        batch_size=config.batch_size,
    )

    trainer.fit(lit_module)


def get_feedback():
    import pickle

    with open(
        "/external2/dkkoshman/YSDA/Machine_Learning/Recomending/local/b650e41133f22902a9a115ff22a3626463a09ba3d48f6fa51e97197ad7528419.blake2s",
        "rb",
    ) as f:
        sparse_interface = pickle.load(f)

    explicit_feedback = sparse_interface.interactions_weighted
    implicit_feedback = explicit_feedback > 0
    return explicit_feedback, implicit_feedback


if __name__ == "__main__":
    main()
