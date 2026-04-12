from __future__ import annotations

import random

import pandas as pd

try:
    from src.retrieve import BM25Retriever
except ImportError:
    from retrieve import BM25Retriever


random.seed(42)


QUERY_SPECS = [
    {"query": "gaming laptop", "positives": [1, 2, 3], "semi_relevant": [4, 17, 18]},
    {"query": "student laptop", "positives": [4], "semi_relevant": [19, 1, 2]},
    {"query": "office chair", "positives": [5, 6, 7], "semi_relevant": [8]},
    {"query": "ergonomic chair", "positives": [5, 6], "semi_relevant": [7]},
    {"query": "wireless earbuds", "positives": [9, 10, 11], "semi_relevant": [12]},
    {"query": "noise cancelling earbuds", "positives": [9, 11], "semi_relevant": [10, 12]},
    {"query": "running shoes", "positives": [13, 14, 15], "semi_relevant": [16, 20]},
    {"query": "trail running shoes", "positives": [14], "semi_relevant": [13, 15]},
    {"query": "gaming keyboard", "positives": [18], "semi_relevant": [17]},
    {"query": "gaming accessories", "positives": [17, 18], "semi_relevant": [1, 2, 3]},
    {"query": "laptop charger", "positives": [19], "semi_relevant": [4, 1, 2, 3]},
    {"query": "socks", "positives": [20], "semi_relevant": [13, 14, 15]},
]


def build_training_data(
    catalog_path: str = "data/raw/catalog.csv",
    output_path: str = "data/raw/search_data_expanded.csv",
    retrieve_top_k: int = 10,
    num_random_negatives: int = 2,
) -> None:
    catalog_df = pd.read_csv(catalog_path)
    retriever = BM25Retriever(catalog_path)

    all_doc_ids = set(catalog_df["doc_id"].tolist())
    rows: list[dict[str, object]] = []

    for spec in QUERY_SPECS:
        query = spec["query"]
        positive_ids = set(spec["positives"])
        semi_ids = set(spec["semi_relevant"])

        for doc_id in positive_ids:
            doc_row = catalog_df[catalog_df["doc_id"] == doc_id].iloc[0]
            rows.append(
                {
                    "query": query,
                    "doc_id": int(doc_id),
                    "document": doc_row["document"],
                    "relevance": 3,
                }
            )

        for doc_id in semi_ids:
            doc_row = catalog_df[catalog_df["doc_id"] == doc_id].iloc[0]
            rows.append(
                {
                    "query": query,
                    "doc_id": int(doc_id),
                    "document": doc_row["document"],
                    "relevance": 1,
                }
            )

        retrieved = retriever.retrieve(query, top_k=retrieve_top_k)
        retrieved_ids = retrieved["doc_id"].tolist()

        hard_negative_ids = [
            doc_id for doc_id in retrieved_ids if doc_id not in positive_ids and doc_id not in semi_ids
        ]

        for doc_id in hard_negative_ids[:3]:
            doc_row = catalog_df[catalog_df["doc_id"] == doc_id].iloc[0]
            rows.append(
                {
                    "query": query,
                    "doc_id": int(doc_id),
                    "document": doc_row["document"],
                    "relevance": 0,
                }
            )

        used_ids = positive_ids.union(semi_ids).union(set(hard_negative_ids[:3]))
        candidate_random_negatives = list(all_doc_ids - used_ids)

        sampled_random_negatives = random.sample(
            candidate_random_negatives,
            k=min(num_random_negatives, len(candidate_random_negatives)),
        )

        for doc_id in sampled_random_negatives:
            doc_row = catalog_df[catalog_df["doc_id"] == doc_id].iloc[0]
            rows.append(
                {
                    "query": query,
                    "doc_id": int(doc_id),
                    "document": doc_row["document"],
                    "relevance": 0,
                }
            )

    training_df = pd.DataFrame(rows)

    training_df = (
        training_df.sort_values(["query", "doc_id", "relevance"], ascending=[True, True, False])
        .drop_duplicates(subset=["query", "doc_id"], keep="first")
        .reset_index(drop=True)
    )

    training_df.to_csv(output_path, index=False)

    print("Saved expanded training data to:", output_path)
    print("Shape:", training_df.shape)
    print(training_df.head(20))


if __name__ == "__main__":
    build_training_data()
