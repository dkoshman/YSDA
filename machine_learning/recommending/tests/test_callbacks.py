import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import yaml

from my_tools.utils import to_torch_coo
from ..callbacks import (
    RecommendingDataOverviewCallback,
    WandbCheckpointCallback,
    WandbWatcher,
    RecommendingExplanationCallback,
)
from ..entrypoint import RecommendingBuilder
from ..models import SLIMRecommender, SLIM, ALSjit

from ..utils import fetch_artifact, load_path_from_artifact
from .conftest import (
    random_explicit_feedback,
    MockLinearLightningModule,
    TEST_CONFIG_PATH,
    cartesian_products_of_dict_values,
)


def _test_lightning_module(config):
    constructor = RecommendingBuilder(config)
    lightning_module = constructor.build_lightning_module()
    trainer = constructor.build_trainer()
    trainer.fit(lightning_module)
    trainer.test(lightning_module)
    model = lightning_module.model

    recommendations = model.recommend(user_ids=torch.arange(100))
    explicit = random_explicit_feedback(n_items=model.n_items)
    explicit = to_torch_coo(explicit)
    online = model.online_recommend(explicit)
    state_dict = model.state_dict()

    loaded_model = type(model)()
    loaded_model.load_state_dict(state_dict)
    loaded_recommendations = loaded_model.recommend(user_ids=torch.arange(100))
    loaded_online = loaded_model.online_recommend(explicit)
    assert (recommendations == loaded_recommendations).all()
    assert (online == loaded_online).all()


def _test_callback(callback, logger=None, lightning_module=None, **trainer_kwargs):
    with wandb.init(project="Testing", dir="local"):
        if lightning_module is None:
            lightning_module = MockLinearLightningModule()
        trainer = pl.Trainer(
            max_epochs=10, callbacks=[callback], logger=logger, **trainer_kwargs
        )
        trainer.fit(lightning_module)
        trainer.test(lightning_module)
        trainer.predict(lightning_module)


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


def test_recommending_explanation_callback():
    explicit = random_explicit_feedback()
    users_explicit = random_explicit_feedback(
        n_users=np.random.randint(explicit.shape[0], size=np.random.randint(1, 3)),
        n_items=explicit.shape[1],
    )
    callback = RecommendingExplanationCallback(
        user_ids=torch.randint(explicit.shape[0], size=(np.random.randint(1, 3),)),
        users_explicit=users_explicit,
        n_recommendations=np.random.randint(1, 3),
    )
    with wandb.init(project="Testing", dir="local", mode="offline"):
        for cls in [SLIM, ALSjit]:
            callback.on_epoch_end(model=cls(explicit=explicit),stage="test")


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
    grid_kwargs = dict(
        log_what=["parameters"], log_every_n_steps=[200], log_graph=[False]
    )
    for kwargs in cartesian_products_of_dict_values(grid_kwargs):
        callback = WandbWatcher(**kwargs)
        _test_callback(callback, logger=pl.loggers.WandbLogger())
