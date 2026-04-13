from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import pyarrow.dataset as ds
from lightgbm import LGBMRanker
from sklearn.model_selection import GroupShuffleSplit

from evaluate import evaluate_per_query
from features import build_features, fit_vectorizer, get_sentence_model

DEFAULT_DATA_DIR = "data/large/msmarco_hf"


def load_large_dataset(data_dir: str, max_rows: int | None = 1_500_000) -> pd.DataFrame:
    parquet_files = sorted(Path(data_dir).glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet shards found in {data_dir}")

    dataset = ds.dataset([str(path) for path in parquet_files], format="parquet")
    scanner = dataset.scanner(columns=["query_id", "query", "document", "relevance"], batch_size=50_000)

    chunks: list[pd.DataFrame] = []
    rows_loaded = 0

    for batch in scanner.to_batches():
        pdf = batch.to_pandas()
        chunks.append(pdf)
        rows_loaded += len(pdf)

        if max_rows is not None and rows_loaded >= max_rows:
            break

    if not chunks:
        raise RuntimeError(f"No rows found in {data_dir}")

    df = pd.concat(chunks, ignore_index=True)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()

    return df


def train_large_ranker(
    data_dir: str = DEFAULT_DATA_DIR,
    max_rows: int | None = 1_500_000,
    use_semantic: bool = False,
    random_state: int = 42,
) -> dict[str, float]:
    df = load_large_dataset(data_dir, max_rows=max_rows)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["query_id"]))

    raw_train = df.iloc[train_idx].copy().sort_values("query_id").reset_index(drop=True)
    raw_test = df.iloc[test_idx].copy().sort_values("query_id").reset_index(drop=True)

    vectorizer = fit_vectorizer(raw_train, save_path="models/tfidf_vectorizer_large.pkl")
    sentence_model = get_sentence_model() if use_semantic else None

    train_df = build_features(
        raw_train,
        vectorizer=vectorizer,
        add_bm25_feature=True,
        add_semantic_feature=use_semantic,
        sentence_model=sentence_model,
    )
    test_df = build_features(
        raw_test,
        vectorizer=vectorizer,
        add_bm25_feature=True,
        add_semantic_feature=use_semantic,
        sentence_model=sentence_model,
    )

    feature_columns = [
        "query_len",
        "doc_len",
        "overlap_count",
        "overlap_ratio",
        "exact_match",
        "tfidf_cosine_sim",
        "bm25_score",
    ]
    if use_semantic:
        feature_columns.append("semantic_cosine_sim")

    x_train = train_df[feature_columns]
    y_train = train_df["relevance"].astype(int)
    group_train = train_df.groupby("query_id").size().to_list()

    x_test = test_df[feature_columns]
    y_test = test_df["relevance"].astype(int)
    group_test = test_df.groupby("query_id").size().to_list()

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        importance_type="gain",
        min_data_in_leaf=20,
        min_data_in_bin=20,
        verbosity=-1,
        random_state=random_state,
    )

    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_test, y_test)],
        eval_group=[group_test],
        eval_at=[1, 3, 5, 10],
    )

    test_df["model_score"] = model.predict(x_test)

    bm25_metrics = evaluate_per_query(test_df, "bm25_score", k=10)
    model_metrics = evaluate_per_query(test_df, "model_score", k=10)

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, "models/lgbm_ranker_large.pkl")
    train_df.to_parquet("data/processed/train_featured_large.parquet", index=False)
    test_df.to_parquet("data/processed/test_featured_large.parquet", index=False)

    report = {
        "rows_train": float(len(train_df)),
        "rows_test": float(len(test_df)),
        "bm25_NDCG@10": float(bm25_metrics.get("NDCG@10", 0.0)),
        "model_NDCG@10": float(model_metrics.get("NDCG@10", 0.0)),
        "bm25_MRR": float(bm25_metrics.get("MRR", 0.0)),
        "model_MRR": float(model_metrics.get("MRR", 0.0)),
        "bm25_MAP": float(bm25_metrics.get("MAP", 0.0)),
        "model_MAP": float(model_metrics.get("MAP", 0.0)),
    }

    Path("experiments").mkdir(parents=True, exist_ok=True)
    Path("experiments/large_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train large ranker on online dataset shards")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--max-rows", type=int, default=1_500_000)
    parser.add_argument("--use-semantic", action="store_true")

    args = parser.parse_args()

    train_large_ranker(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        use_semantic=args.use_semantic,
    )
