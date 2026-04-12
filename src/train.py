from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMRanker

from features import build_features, fit_vectorizer, get_sentence_model
from split import split_data_by_query


DATA_PATH = "data/raw/search_data_expanded.csv"


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


def prepare_groups(df: pd.DataFrame) -> list[int]:
    return df.groupby("query").size().to_list()


def train_ranker(
    data_path: str = DATA_PATH,
    feature_columns: list[str] | None = None,
    random_state: int = 42,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
) -> dict[str, Any]:
    raw_train_df, raw_test_df = split_data_by_query(data_path)

    vectorizer = fit_vectorizer(raw_train_df, save_path="models/tfidf_vectorizer.pkl")
    sentence_model = get_sentence_model()

    train_df = build_features(
        raw_train_df,
        vectorizer=vectorizer,
        add_bm25_feature=True,
        add_semantic_feature=True,
        sentence_model=sentence_model,
    )
    test_df = build_features(
        raw_test_df,
        vectorizer=vectorizer,
        add_bm25_feature=True,
        add_semantic_feature=True,
        sentence_model=sentence_model,
    )

    train_df = train_df.sort_values("query").reset_index(drop=True)
    test_df = test_df.sort_values("query").reset_index(drop=True)

    selected_features = feature_columns or FEATURE_COLUMNS

    missing_features = [feature for feature in selected_features if feature not in train_df.columns]
    if missing_features:
        raise ValueError(f"Missing features in training dataframe: {missing_features}")

    x_train = train_df[selected_features]
    y_train = train_df["relevance"].astype(int)
    group_train = prepare_groups(train_df)

    x_test = test_df[selected_features]
    y_test = test_df["relevance"].astype(int)
    group_test = prepare_groups(test_df)

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        importance_type="gain",
        min_data_in_leaf=1,
        min_data_in_bin=1,
        verbosity=-1,
        random_state=random_state,
    )

    model.fit(
        x_train,
        y_train,
        group=group_train,
        eval_set=[(x_test, y_test)],
        eval_group=[group_test],
        eval_at=[1, 3, 5],
    )

    models_dir = Path("models")
    processed_dir = Path("data/processed")
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "lgbm_ranker.pkl")
    train_df.to_csv(processed_dir / "train_featured.csv", index=False)
    test_df.to_csv(processed_dir / "test_featured.csv", index=False)

    print("Training finished.")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("Features used:", selected_features)
    print("Train groups:", group_train)
    print("Test groups:", group_test)

    return {
        "feature_columns": selected_features,
        "train_shape": train_df.shape,
        "test_shape": test_df.shape,
        "train_groups": group_train,
        "test_groups": group_test,
    }


if __name__ == "__main__":
    train_ranker()
