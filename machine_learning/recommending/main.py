from my_tools.entrypoints import ConfigDispenser

from recommending.cat import MovieLensCatBoostRecommender
from recommending.movielens import MovieLensDispatcher


class AllDispatcher(MovieLensDispatcher):
    def build_class(self, class_candidates=(), **kwargs):
        return super().build_class(
            class_candidates=list(class_candidates) + [MovieLensCatBoostRecommender],
            **kwargs,
        )


@ConfigDispenser
def main(config):
    AllDispatcher(config=config).dispatch()


if __name__ == "__main__":
    main()
