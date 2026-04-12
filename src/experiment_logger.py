from __future__ import annotations

import os

import pandas as pd

LOG_PATH = "experiments/experiments.csv"


def log_experiment(
    name: str,
    features: list[str],
    metrics: dict[str, float],
    notes: str = "",
    log_path: str = LOG_PATH,
) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    row = {
        "experiment_name": name,
        "features": ",".join(features),
        "ndcg@5": metrics.get("NDCG@5", 0.0),
        "mrr": metrics.get("MRR", 0.0),
        "map": metrics.get("MAP", 0.0),
        "notes": notes,
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(log_path, index=False)
    print(f"Logged experiment: {name}")
