from data_loader import load_data
from features import build_features


def main() -> None:
    df = load_data("data/raw/search_data.csv")
    feature_df = build_features(df)

    print(feature_df.head())
    print(feature_df.columns.tolist())


if __name__ == "__main__":
    main()
