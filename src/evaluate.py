import math
from typing import Any

import joblib
import pandas as pd

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


def dcg_at_k(relevances: list[int], k: int) -> float:
    cutoff_rels = relevances[:k]
    score = 0.0

    for rank, rel in enumerate(cutoff_rels, start=1):
        score += (2**rel - 1) / math.log2(rank + 1)

    return score


def ndcg_at_k(y_true: list[int], y_score: list[float], k: int = 10) -> float:
    paired = list(zip(y_true, y_score))
    ranked_by_pred = sorted(paired, key=lambda x: x[1], reverse=True)
    ranked_true = [rel for rel, _ in ranked_by_pred]

    ideal_rels = sorted(y_true, reverse=True)

    dcg = dcg_at_k(ranked_true, k)
    idcg = dcg_at_k(ideal_rels, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def reciprocal_rank(y_true: list[int], y_score: list[float], threshold: int = 1) -> float:
    paired = list(zip(y_true, y_score))
    ranked_by_pred = sorted(paired, key=lambda x: x[1], reverse=True)

    for rank, (rel, _) in enumerate(ranked_by_pred, start=1):
        if rel >= threshold:
            return 1.0 / rank

    return 0.0


def average_precision(y_true: list[int], y_score: list[float], threshold: int = 1) -> float:
    paired = list(zip(y_true, y_score))
    ranked_by_pred = sorted(paired, key=lambda x: x[1], reverse=True)

    num_relevant = sum(1 for rel in y_true if rel >= threshold)
    if num_relevant == 0:
        return 0.0

    precisions = []
    hits = 0

    for rank, (rel, _) in enumerate(ranked_by_pred, start=1):
        if rel >= threshold:
            hits += 1
            precisions.append(hits / rank)

    return sum(precisions) / num_relevant


def evaluate_per_query(df: pd.DataFrame, score_column: str, k: int = 10) -> dict[str, float]:
    ndcg_scores = []
    mrr_scores = []
    map_scores = []

    for _, group in df.groupby("query"):
        y_true = group["relevance"].tolist()
        y_score = group[score_column].tolist()

        ndcg_scores.append(ndcg_at_k(y_true, y_score, k=k))
        mrr_scores.append(reciprocal_rank(y_true, y_score))
        map_scores.append(average_precision(y_true, y_score))

    if not ndcg_scores:
        return {f"NDCG@{k}": 0.0, "MRR": 0.0, "MAP": 0.0}

    return {
        f"NDCG@{k}": sum(ndcg_scores) / len(ndcg_scores),
        "MRR": sum(mrr_scores) / len(mrr_scores),
        "MAP": sum(map_scores) / len(map_scores),
    }


def show_failures(
    df: pd.DataFrame,
    score_column: str,
    top_n: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    rows = []

    for query, group in df.groupby("query"):
        ranked = group.sort_values(score_column, ascending=False).reset_index(drop=True)
        ideal = group.sort_values("relevance", ascending=False).reset_index(drop=True)

        if ranked.empty or ideal.empty:
            continue

        top_pred_doc = ranked.loc[0, "document"]
        top_pred_rel = int(ranked.loc[0, "relevance"])

        top_ideal_doc = ideal.loc[0, "document"]
        top_ideal_rel = int(ideal.loc[0, "relevance"])

        if top_pred_doc != top_ideal_doc:
            rows.append(
                {
                    "query": query,
                    "predicted_top_doc": top_pred_doc,
                    "predicted_top_rel": top_pred_rel,
                    "ideal_top_doc": top_ideal_doc,
                    "ideal_top_rel": top_ideal_rel,
                }
            )

    failure_df = pd.DataFrame(rows)
    if verbose:
        print("\nFailure cases:")
        if failure_df.empty:
            print("No top-1 mismatches found on this test split.")
        else:
            print(failure_df.head(top_n))

    return failure_df


def evaluate_model(
    feature_columns: list[str] | None = None,
    test_data_path: str = "data/processed/test_featured.csv",
    model_path: str = "models/lgbm_ranker.pkl",
    k: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    test_df = pd.read_csv(test_data_path)
    model = joblib.load(model_path)

    selected_features = feature_columns or FEATURE_COLUMNS

    missing_features = [feature for feature in selected_features if feature not in test_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in test dataframe: {missing_features}")

    test_df["model_score"] = model.predict(test_df[selected_features])

    bm25_results = evaluate_per_query(test_df, score_column="bm25_score", k=k)
    model_results = evaluate_per_query(test_df, score_column="model_score", k=k)

    failure_df = show_failures(test_df, "model_score", top_n=20, verbose=False)

    if verbose:
        print("\nBM25 baseline results:")
        for metric, value in bm25_results.items():
            print(f"{metric}: {value:.4f}")

        print("\nReranker results:")
        for metric, value in model_results.items():
            print(f"{metric}: {value:.4f}")

        print("\nDetailed test ranking:")
        for query, group in test_df.groupby("query"):
            ranked = group.sort_values("model_score", ascending=False)
            print(f"\nQuery: {query}")
            print(ranked[["document", "relevance", "bm25_score", "model_score"]])

        print("\nFailure analysis for model:")
        if failure_df.empty:
            print("No top-1 mismatches found on this test split.")
        else:
            print(failure_df.head(20))

    return {
        "bm25": bm25_results,
        "model": model_results,
        "failures": failure_df,
        "features": selected_features,
    }


if __name__ == "__main__":
    evaluate_model()
