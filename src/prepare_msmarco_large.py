from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import ir_datasets
import pyarrow as pa
import pyarrow.parquet as pq


def _doc_text(doc) -> str:
    for attr in ("text", "body", "contents"):
        value = getattr(doc, attr, None)
        if value:
            return str(value)
    return ""


def build_large_ltr_dataset(
    dataset_id: str,
    output_parquet: str,
    output_stats: str,
    max_queries: int,
    hard_negatives_per_query: int,
    min_positive_per_query: int,
    flush_every: int,
) -> None:
    dataset = ir_datasets.load(dataset_id)

    print(f"Dataset: {dataset}")
    print(f"Docs: {dataset.docs_count()}, Queries: {dataset.queries_count()}, Qrels: {dataset.qrels_count()}")

    selected_queries: dict[str, str] = {}
    for query in dataset.queries_iter():
        selected_queries[str(query.query_id)] = str(query.text)
        if len(selected_queries) >= max_queries:
            break

    selected_qids = set(selected_queries.keys())
    print(f"Selected queries: {len(selected_qids)}")

    positives_by_qid: dict[str, set[str]] = defaultdict(set)
    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        if qid not in selected_qids:
            continue
        if int(qrel.relevance) > 0:
            positives_by_qid[qid].add(str(qrel.doc_id))

    negatives_by_qid: dict[str, list[tuple[str, float]]] = defaultdict(list)
    bm25_by_qid_doc: dict[str, dict[str, float]] = defaultdict(dict)

    for scored in dataset.scoreddocs_iter():
        qid = str(scored.query_id)
        if qid not in selected_qids:
            continue

        doc_id = str(scored.doc_id)
        score = float(scored.score) if scored.score is not None else 0.0
        bm25_by_qid_doc[qid][doc_id] = score

        if doc_id in positives_by_qid[qid]:
            continue

        if len(negatives_by_qid[qid]) < hard_negatives_per_query:
            negatives_by_qid[qid].append((doc_id, score))

    usable_qids = [
        qid
        for qid in selected_qids
        if len(positives_by_qid.get(qid, set())) >= min_positive_per_query
        and len(negatives_by_qid.get(qid, [])) > 0
    ]

    print(f"Usable queries after filtering: {len(usable_qids)}")

    needed_doc_ids: set[str] = set()
    for qid in usable_qids:
        needed_doc_ids.update(positives_by_qid[qid])
        needed_doc_ids.update([doc_id for doc_id, _ in negatives_by_qid[qid]])

    print(f"Unique docs required: {len(needed_doc_ids)}")

    doc_text_by_id: dict[str, str] = {}
    found = 0
    for doc in dataset.docs_iter():
        doc_id = str(doc.doc_id)
        if doc_id in needed_doc_ids:
            text = _doc_text(doc)
            if text:
                doc_text_by_id[doc_id] = text
                found += 1
                if found % 100000 == 0:
                    print(f"Collected doc texts: {found}")

        if found >= len(needed_doc_ids):
            break

    print(f"Collected required docs: {len(doc_text_by_id)}")

    output_parquet_path = Path(output_parquet)
    output_parquet_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("query_id", pa.string()),
            ("query", pa.string()),
            ("doc_id", pa.string()),
            ("document", pa.string()),
            ("relevance", pa.int32()),
            ("bm25_score", pa.float32()),
        ]
    )

    writer = pq.ParquetWriter(output_parquet_path, schema=schema, compression="snappy")

    rows: list[dict[str, object]] = []
    total_rows = 0
    positive_rows = 0
    negative_rows = 0

    for idx, qid in enumerate(usable_qids, start=1):
        query_text = selected_queries[qid]

        for doc_id in positives_by_qid[qid]:
            text = doc_text_by_id.get(doc_id)
            if not text:
                continue
            rows.append(
                {
                    "query_id": qid,
                    "query": query_text,
                    "doc_id": doc_id,
                    "document": text,
                    "relevance": 3,
                    "bm25_score": float(bm25_by_qid_doc[qid].get(doc_id, 0.0)),
                }
            )
            positive_rows += 1

        for doc_id, score in negatives_by_qid[qid]:
            text = doc_text_by_id.get(doc_id)
            if not text:
                continue
            rows.append(
                {
                    "query_id": qid,
                    "query": query_text,
                    "doc_id": doc_id,
                    "document": text,
                    "relevance": 0,
                    "bm25_score": float(score),
                }
            )
            negative_rows += 1

        if len(rows) >= flush_every:
            table = pa.Table.from_pylist(rows, schema=schema)
            writer.write_table(table)
            total_rows += len(rows)
            rows = []
            print(f"Written rows: {total_rows}")

        if idx % 10000 == 0:
            print(f"Processed queries: {idx}/{len(usable_qids)}")

    if rows:
        table = pa.Table.from_pylist(rows, schema=schema)
        writer.write_table(table)
        total_rows += len(rows)

    writer.close()

    stats = {
        "dataset_id": dataset_id,
        "selected_queries": len(selected_qids),
        "usable_queries": len(usable_qids),
        "unique_docs": len(doc_text_by_id),
        "rows_total": total_rows,
        "rows_positive": positive_rows,
        "rows_negative": negative_rows,
        "output_parquet": str(output_parquet_path),
        "output_size_mb": round(output_parquet_path.stat().st_size / (1024 * 1024), 2),
    }

    output_stats_path = Path(output_stats)
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Large dataset build complete.")
    print(json.dumps(stats, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build large LTR training pairs from MS MARCO.")
    parser.add_argument("--dataset-id", default="msmarco-passage/train")
    parser.add_argument("--output-parquet", default="data/raw/msmarco_ltr_pairs.parquet")
    parser.add_argument("--output-stats", default="data/raw/msmarco_ltr_stats.json")
    parser.add_argument("--max-queries", type=int, default=800000)
    parser.add_argument("--hard-negatives", type=int, default=30)
    parser.add_argument("--min-positives", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=100000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_large_ltr_dataset(
        dataset_id=args.dataset_id,
        output_parquet=args.output_parquet,
        output_stats=args.output_stats,
        max_queries=args.max_queries,
        hard_negatives_per_query=args.hard_negatives,
        min_positive_per_query=args.min_positives,
        flush_every=args.flush_every,
    )
