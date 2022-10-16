from ..models import als, baseline, mf, slim


from ..lit import LitRecommenderBase, NonGradientRecommenderMixin
from .data import MovieLens
from . import callbacks as movielens_callbacks


class MovieLensRecommender(LitRecommenderBase):
    @property
    def movielens(self):
        return MovieLens(self.hparams["datamodule_config"]["directory"])

    @property
    def train_explicit(self):
        if file := self.hparams["datamodule_config"].get("train_explicit_file"):
            return self.movielens.explicit_feedback_scipy_csr(file)

    @property
    def val_explicit(self):
        if file := self.hparams["datamodule_config"].get("val_explicit_file"):
            return self.movielens.explicit_feedback_scipy_csr(file)

    @property
    def test_explicit(self):
        if file := self.hparams["datamodule_config"].get("test_explicit_file"):
            return self.movielens.explicit_feedback_scipy_csr(file)


class MovieLensNonGradientRecommender(
    NonGradientRecommenderMixin, MovieLensRecommender
):
    @property
    def module_candidates(self):
        return super().module_candidates + [als, baseline, movielens_callbacks]


class MovieLensPMFRecommender(mf.MFRecommender, MovieLensRecommender):
    pass


class MovieLensSLIMRecommender(slim.SLIMRecommender, MovieLensRecommender):
    pass
