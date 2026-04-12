from __future__ import annotations

import pandas as pd

from evaluate import evaluate_model
from experiment_logger import log_experiment
from train import train_ranker


def run_ablation() -> pd.DataFrame:
    configs: list[tuple[str, list[str], str]] = [
        ("bm25_only", ["bm25_score"], "Lexical retrieval signal only."),
        (
            "bm25_tfidf",
            ["bm25_score", "tfidf_cosine_sim"],
            "Add TF-IDF similarity to lexical baseline.",
        ),
        (
            "bm25_tfidf_overlap",
            ["bm25_score", "tfidf_cosine_sim", "overlap_count", "overlap_ratio"],
            "Add token-overlap features.",
        ),
        (
            "hybrid_full",
            [
                "query_len",
                "doc_len",
                "overlap_count",
                "overlap_ratio",
                "exact_match",
                "tfidf_cosine_sim",
                "bm25_score",
                "semantic_cosine_sim",
            ],
            "Full lexical + semantic reranker.",
        ),
    ]

    rows: list[dict[str, object]] = []

    for idx, (name, features, notes) in enumerate(configs, start=1):
        print(f"\nRunning config {idx}/{len(configs)}: {name}")
        train_ranker(feature_columns=features)
        result = evaluate_model(feature_columns=features, verbose=False)

        model_metrics = result["model"]
        log_experiment(name, features, model_metrics, notes=notes)

        row = {
            "experiment": name,
            "features": ", ".join(features),
            "NDCG@5": model_metrics.get("NDCG@5", 0.0),
            "MRR": model_metrics.get("MRR", 0.0),
            "MAP": model_metrics.get("MAP", 0.0),
        }
        rows.append(row)

        print(
            f"{name}: NDCG@5={row['NDCG@5']:.4f}, "
            f"MRR={row['MRR']:.4f}, MAP={row['MAP']:.4f}"
        )

    summary_df = pd.DataFrame(rows).sort_values("NDCG@5", ascending=False).reset_index(drop=True)

    print("\nAblation Summary:")
    print(summary_df)
    return summary_df


if __name__ == "__main__":
    run_ablation()
