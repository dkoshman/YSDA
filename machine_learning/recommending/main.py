from my_tools.entrypoints import ConfigDispenser

from recommending.movielens.models import MovieLensCatBoostRecommender
from recommending.movielens.entrypoints import MovieLensDispatcher


class AllDispatcher(MovieLensDispatcher):
    @property
    def class_candidates(self):
        return super().class_candidates + [MovieLensCatBoostRecommender]


@ConfigDispenser
def main(config):
    AllDispatcher(config=config).dispatch()


if __name__ == "__main__":
    main()
