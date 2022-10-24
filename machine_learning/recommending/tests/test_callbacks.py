import pytorch_lightning as pl
import wandb
import yaml

from ..callbacks import (
    RecommendingDataOverviewCallback,
    WandbCheckpointCallback,
    WandbWatcher,
)

from ..utils import fetch_artifact, load_path_from_artifact
from .conftest import (
    _test_lightning_module,
    random_explicit_feedback,
    MockLinearLightningModule,
    TEST_CONFIG_PATH,
    cartesian_products_of_dict_values,
)


def _test_callback(callback, logger=None):
    with wandb.init(project="Testing", dir="local"):
        lit = MockLinearLightningModule()
        trainer = pl.Trainer(max_epochs=10, callbacks=[callback], logger=logger)
        trainer.fit(lit)
        trainer.test(lit)
        trainer.predict(lit)


def test_wandb_artifact_checkpointing():
    artifact_name = "TestingCheckpoint"
    callback = WandbCheckpointCallback(artifact_name=artifact_name)
    _test_callback(callback)
    artifact = fetch_artifact(
        entity="dkoshman", project="Testing", artifact_name=artifact_name
    )
    checkpoint_path = load_path_from_artifact(artifact)
    loaded_lit = MockLinearLightningModule.load_from_checkpoint(checkpoint_path)
    trainer = pl.Trainer()
    trainer.test(loaded_lit)
    trainer.predict(loaded_lit)


def test_recommending_data_overview_callback():
    explicit = random_explicit_feedback()
    callback = RecommendingDataOverviewCallback(explicit=explicit)
    with wandb.init(project="Testing", dir="local", mode="offline"):
        callback.log_data_overview()
    _test_callback(callback)


def test_recommending_imdb_callback():
    config = yaml.safe_load(open(TEST_CONFIG_PATH))
    config.update(callbacks=dict(RecommendingIMDBCallback=dict()))
    _test_lightning_module(config)


def test_recommending_metrics_callback():
    config = yaml.safe_load(open(TEST_CONFIG_PATH))
    config.update(callbacks=dict(RecommendingMetricsCallback=dict()))
    _test_lightning_module(config)


def test_catboost_callback():
    config = yaml.safe_load(open(TEST_CONFIG_PATH))
    config.update(
        dict(
            model=dict(class_name="CatboostExplicitRecommender"),
            callbacks=dict(CatBoostMetrics=dict()),
        )
    )
    _test_lightning_module(config)


def test_wandb_watcher_callback():
    logger = pl.loggers.WandbLogger()
    grid_kwargs = dict(
        log_what=["parameters"], log_every_n_steps=[200], log_graph=[False]
    )
    for kwargs in cartesian_products_of_dict_values(grid_kwargs):
        callback = WandbWatcher(**kwargs)
        _test_callback(callback, logger=logger)
