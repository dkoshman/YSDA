from .data import MovieLensMixin
from ..models import als, baseline, cat, mf, slim
from ..lit import NonGradientRecommenderMixin, LitRecommenderBase
from . import cat as movielens_cat


class MovieLensNonGradientRecommenderMixin(NonGradientRecommenderMixin):
    @property
    def module_candidates(self):
        return super().module_candidates + [
            als,
            baseline,
            cat,
            movielens_cat,
        ]


class MovieLensRecommender(MovieLensMixin, LitRecommenderBase):
    pass


class MovieLensNonGradientRecommender(
    MovieLensNonGradientRecommenderMixin, MovieLensRecommender
):
    pass


class MovieLensMFRecommender(mf.MFRecommender, MovieLensRecommender):
    pass


class MovieLensSLIMRecommender(slim.SLIMRecommender, MovieLensRecommender):
    pass
