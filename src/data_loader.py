from pathlib import Path

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    csv_path = Path(file_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    df = load_data("data/raw/search_data.csv")
    print(df.head())
    print(df.shape)
