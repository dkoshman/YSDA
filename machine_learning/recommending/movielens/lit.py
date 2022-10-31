from ..models import als, baseline, cat, mf, slim
from ..lit import LitRecommenderBase, NonGradientRecommenderMixin
from .data import MovieLens100k, MovieLens25m
from . import callbacks as movielens_callbacks
from . import cat as movielens_cat


class MovieLensRecommender(LitRecommenderBase):
    @property
    def movielens(self):
        return MovieLens100k(self.hparams["datamodule_config"]["directory"])

    def train_explicit(self):
        if file := self.hparams["datamodule_config"].get("train_explicit_file"):
            return self.movielens.explicit_feedback_scipy_csr(file)

    def val_explicit(self):
        if file := self.hparams["datamodule_config"].get("val_explicit_file"):
            return self.movielens.explicit_feedback_scipy_csr(file)

    def test_explicit(self):
        if file := self.hparams["datamodule_config"].get("test_explicit_file"):
            return self.movielens.explicit_feedback_scipy_csr(file)


class MovieLens25mRecommender(LitRecommenderBase):
    @property
    def movielens(self):
        return MovieLens25m(self.hparams["datamodule_config"]["directory"])

    def train_explicit(self):
        return self.movielens.explicit_feedback_scipy_csr("train_ratings")

    def val_explicit(self):
        return self.movielens.explicit_feedback_scipy_csr("test_ratings")

    def test_explicit(self):
        return self.movielens.explicit_feedback_scipy_csr("test_ratings")


class MovieLensNonGradientRecommenderMixin(NonGradientRecommenderMixin):
    @property
    def module_candidates(self):
        return super().module_candidates + [
            als,
            baseline,
            cat,
            movielens_cat,
            movielens_callbacks,
        ]


class MovieLensNonGradientRecommender(
    MovieLensNonGradientRecommenderMixin, MovieLensRecommender
):
    pass


class MovieLensMFRecommender(mf.MFRecommender, MovieLensRecommender):
    pass


class MovieLensSLIMRecommender(slim.SLIMRecommender, MovieLensRecommender):
    pass


class MovieLensMyMFRecommender(mf.MyMFRecommender, MovieLensRecommender):
    pass


class MovieLens25mNonGradientRecommender(
    MovieLensNonGradientRecommenderMixin, MovieLens25mRecommender
):
    pass


class MovieLens25mMFRecommender(mf.MFRecommender, MovieLens25mRecommender):
    pass


class MovieLens25mSLIMRecommender(slim.SLIMRecommender, MovieLens25mRecommender):
    pass


class MovieLens25mMyMFRecommender(mf.MyMFRecommender, MovieLens25mRecommender):
    pass
