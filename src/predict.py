from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
    from src.features import build_features, get_sentence_model
    from src.retrieve import BM25Retriever
except ImportError:
    from features import build_features, get_sentence_model
    from retrieve import BM25Retriever


FEATURE_COLUMNS = [
    "query_len",
    "doc_len",
    "overlap_count",
    "overlap_ratio",
    "exact_match",
    "tfidf_cosine_sim",
    "bm25_score",
    "semantic_cosine_sim",
]

def load_model(model_path: str = "models/lgbm_ranker.pkl"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run training first.")
    return joblib.load(path)


def load_vectorizer(vectorizer_path: str = "models/tfidf_vectorizer.pkl"):
    path = Path(vectorizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {path}. Run training first.")
    return joblib.load(path)


def rank_documents(
    query: str,
    documents: list[str],
    model_path: str = "models/lgbm_ranker.pkl",
    vectorizer_path: str = "models/tfidf_vectorizer.pkl",
) -> pd.DataFrame:
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)

    inference_df = pd.DataFrame(
        {
            "query": [query] * len(documents),
            "document": documents,
            "relevance": [0] * len(documents),
        }
    )

    featured = build_features(inference_df, vectorizer=vectorizer)
    featured["model_score"] = model.predict(featured[FEATURE_COLUMNS])

    ranked = featured.sort_values("model_score", ascending=False).reset_index(drop=True)
    return ranked[["query", "document", "model_score"]]


def rerank_candidates(
    query: str,
    candidates_df: pd.DataFrame,
    model_path: str = "models/lgbm_ranker.pkl",
    vectorizer_path: str = "models/tfidf_vectorizer.pkl",
) -> pd.DataFrame:
    model = load_model(model_path)
    vectorizer = load_vectorizer(vectorizer_path)
    sentence_model = get_sentence_model()

    infer_df = pd.DataFrame(
        {
            "query": [query] * len(candidates_df),
            "document": candidates_df["document"].tolist(),
            "relevance": [0] * len(candidates_df),
        }
    )

    infer_df = build_features(
        infer_df,
        vectorizer=vectorizer,
        add_bm25_feature=False,
        add_semantic_feature=True,
        sentence_model=sentence_model,
    )
    infer_df["bm25_score"] = candidates_df["bm25_score"].to_numpy()
    infer_df["model_score"] = model.predict(infer_df[FEATURE_COLUMNS])

    output_df = candidates_df.copy()
    output_df["model_score"] = infer_df["model_score"].to_numpy()

    return output_df.sort_values("model_score", ascending=False).reset_index(drop=True)


def search(query: str, retrieve_top_k: int = 10, catalog_path: str = "data/raw/catalog.csv") -> pd.DataFrame:
    retriever = BM25Retriever(catalog_path)
    candidates_df = retriever.retrieve(query, top_k=retrieve_top_k)

    return rerank_candidates(query, candidates_df)


def rank_candidates(query: str, candidates: list[str], model_path: str = "models/lgbm_ranker.pkl") -> pd.DataFrame:
    return rank_documents(query=query, documents=candidates, model_path=model_path)


if __name__ == "__main__":
    query = "gaming laptop"
    results = search(query, retrieve_top_k=8)
    print(results[["doc_id", "document", "bm25_score", "model_score"]])
