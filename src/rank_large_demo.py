from __future__ import annotations

import argparse

import joblib
import pandas as pd
from datasets import load_dataset
from rank_bm25 import BM25Okapi

from features import build_features, clean_text

FEATURE_COLUMNS_LEXICAL = [
    "query_len",
    "doc_len",
    "overlap_count",
    "overlap_ratio",
    "exact_match",
    "tfidf_cosine_sim",
    "bm25_score",
]


def run_demo(query_override: str | None = None, retrieve_top_k: int = 10) -> pd.DataFrame:
    model = joblib.load("models/lgbm_ranker_large.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer_large.pkl")

    ds = load_dataset("ms_marco", "v1.1", split="validation")

    sample = ds[0]
    query = query_override or sample["query"]
    documents = sample["passages"]["passage_text"]

    tokenized_docs = [clean_text(doc).split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)

    query_tokens = clean_text(query).split()
    bm25_scores = bm25.get_scores(query_tokens)

    candidates = pd.DataFrame(
        {
            "query": [query] * len(documents),
            "document": documents,
            "relevance": [0] * len(documents),
            "bm25_score": bm25_scores,
        }
    ).sort_values("bm25_score", ascending=False).head(retrieve_top_k).reset_index(drop=True)

    featured = build_features(
        candidates[["query", "document", "relevance"]],
        vectorizer=vectorizer,
        add_bm25_feature=False,
        add_semantic_feature=False,
    )
    featured["bm25_score"] = candidates["bm25_score"].to_numpy()

    featured["model_score"] = model.predict(featured[FEATURE_COLUMNS_LEXICAL])

    out = candidates.copy()
    out["model_score"] = featured["model_score"].to_numpy()
    out = out.sort_values("model_score", ascending=False).reset_index(drop=True)

    print(f"Query: {query}")
    print(out[["document", "bm25_score", "model_score"]].head(10))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-stage ranking demo with large model")
    parser.add_argument("--query", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    run_demo(query_override=args.query, retrieve_top_k=args.top_k)
