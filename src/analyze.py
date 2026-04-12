from __future__ import annotations

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


def show_feature_importance() -> pd.DataFrame:
    model = joblib.load("models/lgbm_ranker.pkl")

    gain_importance = model.booster_.feature_importance(importance_type="gain")
    split_importance = model.booster_.feature_importance(importance_type="split")

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain_importance": gain_importance,
            "split_importance": split_importance,
        }
    )

    importance_df = (
        importance_df.sort_values("gain_importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\nFeature Importance:")
    print(importance_df)
    return importance_df


if __name__ == "__main__":
    show_feature_importance()
