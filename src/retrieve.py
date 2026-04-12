from __future__ import annotations

import re

import pandas as pd
from rank_bm25 import BM25Okapi


def clean_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9\s]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


class BM25Retriever:
    def __init__(self, catalog_path: str):
        self.catalog_df = pd.read_csv(catalog_path).copy()
        self.catalog_df["document_clean"] = self.catalog_df["document"].apply(clean_text)

        self.tokenized_corpus = [
            doc.split() for doc in self.catalog_df["document_clean"].tolist()
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_clean = clean_text(query)
        tokenized_query = query_clean.split()

        scores = self.bm25.get_scores(tokenized_query)

        results = self.catalog_df.copy()
        results["bm25_score"] = scores

        ranked = (
            results.sort_values("bm25_score", ascending=False)
            .head(top_k)
            .reset_index(drop=True)
        )
        return ranked[["doc_id", "document", "category", "bm25_score"]]


if __name__ == "__main__":
    retriever = BM25Retriever("data/raw/catalog.csv")
    query = "gaming laptop"
    print(retriever.retrieve(query, top_k=5))
