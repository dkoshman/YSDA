import pathlib
import pickle

from dataset import get_data, RankingDataset, split_data, train_vectorizer


def prepare_text_data(max_features):
    train, queries, orgs = get_data()

    train_relevance, val_relevance, test_relevance = split_data(
        train[["query_id", "org_id", "relevance"]],
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
    )

    queries, orgs, vectorizer = train_vectorizer(
        queries=queries,
        orgs=orgs,
        train_relevance=train_relevance,
        max_features=max_features,
    )

    train_dataset = RankingDataset(queries, orgs, train_relevance)
    val_dataset = RankingDataset(queries, orgs, val_relevance)
    test_dataset = RankingDataset(queries, orgs, test_relevance)

    pathlib.Path("data").mkdir(exist_ok=True)
    data_name = f"max_features_{max_features}"
    pickle.dump(
        {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "test_dataset": test_dataset,
            "vectorizer": vectorizer,
        },
        open(f"data/{data_name}.pickle", "wb"),
    )
    print(f"Data {data_name} ready")


def main():
    max_features_candidates = [None, 4096, 1024, 256]
    for max_features in max_features_candidates:
        prepare_text_data(max_features)


if __name__ == "__main__":
    main()
