from als import ALSRecommender
from bpmf import BPMFRecommender
from entrypoints import BaselineRecommender
from movielens import MovieLensDataModuleMixin, MovieLensDispatcher
from pmf import PMFRecommender
from slim import SLIMRecommender

from my_tools.entrypoints import ConfigDispenser


class MovieLensALSRecommender(ALSRecommender, MovieLensDataModuleMixin):
    pass


class MovieLensBPMFRecommender(BPMFRecommender, MovieLensDataModuleMixin):
    pass


class MovieLensPMFRecommender(PMFRecommender, MovieLensDataModuleMixin):
    pass


class MovieLensBaselineRecommender(BaselineRecommender, MovieLensDataModuleMixin):
    pass


class MovieLensSLIMRecommender(SLIMRecommender, MovieLensDataModuleMixin):
    pass


@ConfigDispenser
def main(config):
    MovieLensDispatcher(
        config=config,
        class_candidates=[
            MovieLensALSRecommender,
            MovieLensBaselineRecommender,
            MovieLensBPMFRecommender,
            MovieLensPMFRecommender,
            MovieLensSLIMRecommender,
        ],
    ).dispatch()


if __name__ == "__main__":
    main()
