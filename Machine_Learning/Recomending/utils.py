import abc

import numpy as np
import scipy.sparse
import torch
import wandb

from my_tools.entrypoints import ConfigConstructorBase, ConfigDispenser
from my_tools.utils import get_class

from data import MovieLens
from metrics import RecommendingMetricsCallback


def build_weight(*dimensions):
    weight = torch.nn.Parameter(torch.empty(*dimensions))
    torch.nn.init.xavier_normal_(weight)
    return weight


def build_bias(*dimensions):
    bias = torch.nn.Parameter(torch.zeros(*dimensions))
    return bias


def torch_sparse_to_scipy_coo(sparse_matrix):
    try:
        sparse_matrix = sparse_matrix.to_sparse_coo()
    except NotImplementedError:
        # Torch errors if coo tensor is cast to coo again.
        pass
    sparse_matrix = sparse_matrix.coalesce().cpu()
    sparse_matrix = scipy.sparse.coo_matrix(
        (sparse_matrix.values().numpy(), sparse_matrix.indices().numpy()),
        shape=sparse_matrix.shape,
    )
    return sparse_matrix


def torch_sparse_slice(sparse_matrix, row_ids=None, col_ids=None, device=None):
    if torch.is_tensor(sparse_matrix):
        sparse_matrix = torch_sparse_to_scipy_coo(sparse_matrix)

    if row_ids is None:
        row_ids = slice(None)
    elif torch.is_tensor(row_ids):
        row_ids = row_ids.cpu().numpy()

    if col_ids is None:
        col_ids = slice(None)
    elif torch.is_tensor(col_ids):
        col_ids = col_ids.cpu().numpy()

    sparse_matrix = sparse_matrix.tocsr()[row_ids][:, col_ids].tocoo()

    torch_sparse_coo_tensor = torch.sparse_coo_tensor(
        indices=np.stack([sparse_matrix.row, sparse_matrix.col]),
        values=sparse_matrix.data,
        size=sparse_matrix.shape,
        device=device,
    )
    return torch_sparse_coo_tensor


class RecommenderMixin:
    def build_class(self, class_config, class_candidates):
        class_config = class_config.copy()
        model_name = class_config.pop("name")
        Model = get_class(model_name, class_candidates=class_candidates)
        return Model(class_config)

    def configure_optimizers(self):
        # Need to copy the dict, otherwise tensor parameters will be saved to hparams
        optimizer_config = self.hparams["optimizer_config"].copy()
        optimizer_name = optimizer_config.pop("name")
        Optimizer = get_class(
            optimizer_name, modules_to_try_to_import_from=[torch.optim]
        )
        optimizer = Optimizer(params=self.parameters(), **optimizer_config)
        return optimizer


class RecommendingConfigDispenser(ConfigDispenser):
    def parser(self, parser):
        parser.add_argument(
            "--stage",
            "-s",
            default="tune",
            type=str,
            help="One of: tune, test.",
            nargs="?",
        )
        return parser


class MovielensTuner(ConfigConstructorBase, abc.ABC):
    def __init__(self, config):
        config["data"].update(
            dict(
                train_explicit_feedback_file="u1.base",
                val_explicit_feedback_file="u1.test",
            )
        )
        super().__init__(config)


class MovielensTester(ConfigConstructorBase, abc.ABC):
    def __init__(self, config):
        super().__init__(config)
        movielens = MovieLens(config["data"]["folder"])
        explicit = movielens.explicit_feedback_scipy_csr("u.data")
        relevant_pairs = explicit.tocoo().indices().T
        self.metrics_callback = RecommendingMetricsCallback(relevant_pairs, config["k"])

    def datasets_iter(self):
        for i in [2, 3, 4, 5]:
            self.config["data"].update(
                dict(
                    train_explicit_feedback_file=f"u{i}.base",
                    test_explicit_feedback_file=f"u{i}.test",
                )
            )
            yield

    def build_callbacks(self, callback_candidates=()):
        callbacks = super().build_callbacks(callback_candidates)
        callbacks["metrics"] = self.metrics_callback
        return callbacks

    def test(self):
        with wandb.init(project=self.config["project"], config=self.config):
            for _ in self.datasets_iter():
                self.main()
