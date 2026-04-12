from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupShuffleSplit

try:
    from src.features import build_features, fit_vectorizer
except ImportError:
    from features import build_features, fit_vectorizer


LARGE_FEATURE_COLUMNS = [
    "query_len",
    "doc_len",
    "overlap_count",
    "overlap_ratio",
    "exact_match",
    "tfidf_cosine_sim",
    "bm25_score",
]


def split_by_query(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df["query"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy().sort_values("query").reset_index(drop=True)
    test_df = df.iloc[test_idx].copy().sort_values("query").reset_index(drop=True)
    return train_df, test_df


def prepare_groups(df: pd.DataFrame) -> list[int]:
    return df.groupby("query").size().to_list()


def train_large(
    pairs_path: str,
    max_train_queries: int | None,
    model_path: str,
    vectorizer_path: str,
    train_output: str,
    test_output: str,
) -> None:
    print(f"Loading pairs from: {pairs_path}")
    raw_df = pd.read_parquet(pairs_path)

    if max_train_queries is not None:
        selected_queries = raw_df["query"].drop_duplicates().head(max_train_queries)
        raw_df = raw_df[raw_df["query"].isin(selected_queries)].reset_index(drop=True)
        print(f"Filtered to first {max_train_queries} queries. Rows: {len(raw_df)}")

    raw_df = raw_df[["query", "document", "relevance", "bm25_score"]].dropna(subset=["query", "document"])

    train_raw, test_raw = split_by_query(raw_df)

    vectorizer = fit_vectorizer(train_raw, save_path=vectorizer_path)

    train_df = build_features(
        train_raw,
        vectorizer=vectorizer,
        add_bm25_feature=False,
        add_semantic_feature=False,
    )
    test_df = build_features(
        test_raw,
        vectorizer=vectorizer,
        add_bm25_feature=False,
        add_semantic_feature=False,
    )

    train_df["bm25_score"] = train_raw["bm25_score"].to_numpy()
    test_df["bm25_score"] = test_raw["bm25_score"].to_numpy()

    x_train = train_df[LARGE_FEATURE_COLUMNS]
    y_train = train_df["relevance"].astype(int)
    group_train = prepare_groups(train_df)

    x_test = test_df[LARGE_FEATURE_COLUMNS]
    y_test = test_df["relevance"].astype(int)
    group_test = prepare_groups(test_df)

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        importance_type="gain",
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_test, y_test)],
        eval_group=[group_test],
        eval_at=[10],
    )

    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path_obj)

    train_out = Path(train_output)
    test_out = Path(test_output)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)

    print("Large training completed.")
    print(f"Train rows: {len(train_df):,}, Test rows: {len(test_df):,}")
    print(f"Model saved: {model_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LambdaRank model on large MS MARCO-derived pairs.")
    parser.add_argument("--pairs-path", default="data/raw/msmarco_ltr_pairs.parquet")
    parser.add_argument("--max-train-queries", type=int, default=None)
    parser.add_argument("--model-path", default="models/lgbm_ranker_large.pkl")
    parser.add_argument("--vectorizer-path", default="models/tfidf_vectorizer_large.pkl")
    parser.add_argument("--train-output", default="data/processed/train_featured_large.parquet")
    parser.add_argument("--test-output", default="data/processed/test_featured_large.parquet")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_large(
        pairs_path=args.pairs_path,
        max_train_queries=args.max_train_queries,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        train_output=args.train_output,
        test_output=args.test_output,
    )
