from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import catboost
import numpy as np
import pandas as pd
import torch

from ..interface import FitExplicitInterfaceMixin, RecommenderModuleBase

if TYPE_CHECKING:
    from scipy.sparse import spmatrix


class CatboostInterface(RecommenderModuleBase, FitExplicitInterfaceMixin, ABC):
    model: catboost.CatBoost

    def __init__(self, explicit=None, **cb_params):
        super().__init__(explicit=explicit)
        self.model = catboost.CatBoostRanker(**cb_params)

    def explicit_dataframe(self, explicit: "spmatrix"):
        if explicit.shape != self.explicit_scipy_coo().shape:
            raise ValueError(
                "Provided explicit feedback shape doesn't match "
                "train shape, user and item ids may be incorrect."
            )
        explicit = explicit.tocoo()
        dataframe = pd.DataFrame(
            dict(
                user_ids=explicit.row,
                item_ids=explicit.col,
                explicit=explicit.data,
            )
        )
        return dataframe

    @staticmethod
    def dense_dataframe(user_ids, item_ids):
        dataframe = pd.DataFrame(
            dict(
                user_ids=np.repeat(user_ids.numpy(), len(item_ids)),
                item_ids=np.tile(item_ids.numpy(), len(user_ids)),
            )
        )
        return dataframe

    @abstractmethod
    def train_dataframe(self, explicit):
        ...

    @abstractmethod
    def predict_dataframe(self, user_ids, item_ids):
        ...

    @property
    def cat_features(self) -> list[str]:
        """The categorical columns in dataframe being passed to pool."""
        return []

    @property
    def text_features(self) -> list[str]:
        """The text columns in dataframe being passed to pool."""
        return []

    def pool(self, dataframe):
        if "user_ids" not in dataframe:
            raise ValueError("Passed dataframe must at least have a user_ids column.")
        dataframe = dataframe.astype(
            {c: "string" for c in self.cat_features + self.text_features}
        )
        for column in dataframe.columns.drop(self.cat_features + self.text_features):
            dataframe[column] = pd.to_numeric(dataframe[column])
        if "explicit" not in dataframe:
            label = None
        else:
            label = dataframe["explicit"].to_numpy()
            dataframe = dataframe.drop(["explicit"], axis="columns")

        pool = catboost.Pool(
            data=dataframe,
            cat_features=self.cat_features,
            text_features=self.text_features,
            label=label,
            group_id=dataframe["user_ids"].to_numpy(),
        )
        return pool

    def fit(self):
        dataframe = self.train_dataframe(self.explicit_scipy_coo())
        pool = self.pool(dataframe)
        self.model.fit(pool, verbose=100)

    def forward(self, user_ids, item_ids):
        dataframe = self.predict_dataframe(user_ids, item_ids)
        pool = self.pool(dataframe)
        ratings = self.model.predict(pool)
        ratings = torch.from_numpy(ratings.reshape(len(user_ids), -1))
        return ratings.to(torch.float32)

    def feature_importance(self, explicit: "spmatrix" or None = None) -> pd.Series:
        """
        Returns series with feature names in index and
        their importance in values, sorted by decreasing importance.
        """
        if explicit is None:
            explicit = self.explicit_scipy_coo()
        dataframe = self.train_dataframe(explicit)
        pool = self.pool(dataframe)
        feature_importance = self.model.get_feature_importance(pool, prettified=True)
        return feature_importance

    def get_extra_state(self):
        self.model.save_model("tmp")
        with open("tmp", "rb") as f:
            binary_bytes = f.read()
        return binary_bytes

    def set_extra_state(self, binary_bytes):
        with open("tmp", "wb") as f:
            f.write(binary_bytes)
        self.model.load_model("tmp")


class CatboostExplicitRecommender(CatboostInterface):
    @property
    def cat_features(self) -> list[str]:
        return ["user_ids", "item_ids"]

    def train_dataframe(self, explicit):
        return self.explicit_dataframe(explicit)

    def predict_dataframe(self, user_ids, item_ids):
        return self.dense_dataframe(user_ids, item_ids)
