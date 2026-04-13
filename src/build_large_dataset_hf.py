from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset


def _write_shard(rows: list[dict[str, object]], out_dir: Path, shard_idx: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"train_shard_{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    return path.stat().st_size


def build_large_dataset(
    output_dir: str = "data/large/msmarco_hf",
    target_gb: float = 12.0,
    shard_rows: int = 250_000,
    max_negatives_per_query: int = 4,
    max_records: int | None = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(target_gb * (1024**3))

    stream = load_dataset("ms_marco", "v1.1", split="train", streaming=True)

    rows: list[dict[str, object]] = []
    shard_idx = 0
    bytes_written = 0
    queries_seen = 0
    rows_written = 0

    for item in stream:
        query = item["query"]
        query_id = int(item["query_id"])

        passages = item["passages"]["passage_text"]
        is_selected = item["passages"]["is_selected"]

        positives = [p for p, sel in zip(passages, is_selected) if sel == 1]
        negatives = [p for p, sel in zip(passages, is_selected) if sel == 0]

        if not positives:
            continue

        for pos_doc in positives:
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "document": pos_doc,
                    "relevance": 3,
                }
            )

        for neg_doc in negatives[:max_negatives_per_query]:
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "document": neg_doc,
                    "relevance": 0,
                }
            )

        queries_seen += 1

        if len(rows) >= shard_rows:
            shard_bytes = _write_shard(rows, out_dir, shard_idx)
            bytes_written += shard_bytes
            rows_written += len(rows)
            rows = []
            shard_idx += 1

            print(
                f"Wrote shard {shard_idx:05d} | "
                f"rows={rows_written:,} | size={bytes_written / (1024**3):.2f} GB"
            )

        if bytes_written >= target_bytes:
            break

        if max_records is not None and queries_seen >= max_records:
            break

    if rows:
        shard_bytes = _write_shard(rows, out_dir, shard_idx)
        bytes_written += shard_bytes
        rows_written += len(rows)
        shard_idx += 1

    metadata = {
        "dataset": "ms_marco v1.1 train (streaming)",
        "target_gb": target_gb,
        "bytes_written": bytes_written,
        "written_gb": bytes_written / (1024**3),
        "rows_written": rows_written,
        "queries_seen": queries_seen,
        "shards": shard_idx,
        "max_negatives_per_query": max_negatives_per_query,
    }

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nBuild complete")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build large LTR dataset from online MS MARCO")
    parser.add_argument("--output-dir", default="data/large/msmarco_hf")
    parser.add_argument("--target-gb", type=float, default=12.0)
    parser.add_argument("--shard-rows", type=int, default=250_000)
    parser.add_argument("--max-negatives-per-query", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)

    args = parser.parse_args()

    build_large_dataset(
        output_dir=args.output_dir,
        target_gb=args.target_gb,
        shard_rows=args.shard_rows,
        max_negatives_per_query=args.max_negatives_per_query,
        max_records=args.max_records,
    )
