import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from data_loader import load_data


def split_data_by_query(
    file_path: str,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_data(file_path)

    groups = df["query"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df = train_df.sort_values("query").reset_index(drop=True)
    test_df = test_df.sort_values("query").reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = split_data_by_query("data/raw/search_data_expanded.csv")

    print("Train queries:", train_df["query"].unique())
    print("Test queries:", test_df["query"].unique())
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
