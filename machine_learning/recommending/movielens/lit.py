from ..models import als, baseline, cat, mf, slim
from ..lit import LitRecommenderBase, NonGradientRecommenderMixin
from .data import MovieLens100k, MovieLens25m
from . import cat as movielens_cat


class MovieLensRecommender(LitRecommenderBase):
    @property
    def movielens(self):
        return MovieLens100k(self.hparams["datamodule"]["directory"])

    def common_explicit(self, filename):
        if file := self.hparams["datamodule"].get(filename):
            try:
                return self.movielens.explicit_feedback_scipy_csr(file)
            except FileNotFoundError:
                return

    def train_explicit(self):
        return self.common_explicit(filename="train_explicit_file")

    def val_explicit(self):
        return self.common_explicit(filename="val_explicit_file")

    def test_explicit(self):
        return self.common_explicit(filename="test_explicit_file")


class MovieLens25mRecommender(LitRecommenderBase):
    @property
    def movielens(self):
        return MovieLens25m(self.hparams["datamodule"]["directory"])

    def common_explicit(self, filename):
        try:
            return self.movielens.explicit_feedback_scipy_csr(filename)
        except FileNotFoundError:
            return

    def train_explicit(self):
        return self.common_explicit("train_ratings")

    def val_explicit(self):
        return self.common_explicit("test_ratings")

    def test_explicit(self):
        return self.common_explicit("test_ratings")


class MovieLensNonGradientRecommenderMixin(NonGradientRecommenderMixin):
    @property
    def module_candidates(self):
        return super().module_candidates + [
            als,
            baseline,
            cat,
            movielens_cat,
        ]


class MovieLensNonGradientRecommender(
    MovieLensNonGradientRecommenderMixin, MovieLensRecommender
):
    pass


class MovieLensMFRecommender(mf.MFRecommender, MovieLensRecommender):
    pass


class MovieLensSLIMRecommender(slim.SLIMRecommender, MovieLensRecommender):
    pass


class MovieLens25mNonGradientRecommender(
    MovieLensNonGradientRecommenderMixin, MovieLens25mRecommender
):
    pass


class MovieLens25mMFRecommender(mf.MFRecommender, MovieLens25mRecommender):
    pass


class MovieLens25mSLIMRecommender(slim.SLIMRecommender, MovieLens25mRecommender):
    pass
