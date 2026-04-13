# Two-Stage Search Ranking System

<p align="center">
   <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1200&color=22C55E&center=true&vCenter=true&width=900&lines=BM25+Retrieval+%E2%86%92+ML+Reranking;Lexical+%2B+Semantic+Signals;Built+for+Portfolio+%2B+Interview+Defense" alt="Typing animation" />
</p>

<p align="center">
   <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
   <img src="https://img.shields.io/badge/LightGBM-LambdaRank-02569B?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM" />
   <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
   <img src="https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
   <img src="https://img.shields.io/badge/Docker-Deployment-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</p>

Production-minded search intelligence pipeline where a user query is first matched lexically (BM25) and then reranked by a learning-to-rank model using lexical + semantic features.

## 30-Second Interview Pitch

I built a production-minded two-stage search ranking system where BM25 first retrieves top-K lexical candidates, then a LightGBM LambdaRank reranker reorders them using lexical and semantic features. The project includes hard-negative data generation, offline ranking evaluation with NDCG, MRR, MAP, experiment tracking, a FastAPI serving layer, a Streamlit demo UI, and Docker packaging.

## Table Of Contents

- [30-Second Interview Pitch](#30-second-interview-pitch)
- [Why This Project](#why-this-project)
- [Visual Architecture Gallery](#visual-architecture-gallery)
- [Demo Preview](#demo-preview)
- [Demo Recording Guide](#demo-recording-guide)
- [System Architecture](#system-architecture)
- [Request Lifecycle](#request-lifecycle)
- [Feature Stack](#feature-stack)
- [Results Snapshot](#results-snapshot)
- [Interactive Runbook](#interactive-runbook)
- [Large Dataset Pipeline (10-20 GB)](#large-dataset-pipeline-10-20-gb)
- [API Example](#api-example)
- [Experiment Tracking](#experiment-tracking)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

## Why This Project

- Solves a real ranking problem using a scalable two-stage architecture.
- Uses hard negatives to improve model discrimination.
- Combines lexical and semantic relevance features.
- Includes evaluation, experiment logging, API serving, UI demo, and Docker packaging.

## Visual Architecture Gallery

![Architecture dark](docs/assets/architecture-dark.svg)

## Demo Preview

![Demo preview placeholder](docs/assets/demo-placeholder.svg)

Replace this placeholder with a short GIF once you record UI interaction.

## Demo Recording Guide

- Full checklist: [docs/demo-recording-checklist.md](docs/demo-recording-checklist.md)
- One-command launcher (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-demo.ps1
```

- Auto-convert recorded MP4 to optimized GIF:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/convert-demo-to-gif.ps1 -InputMp4 docs/assets/demo.mp4 -OutputGif docs/assets/demo.gif
```

- After recording, export GIF as `docs/assets/demo.gif` and replace the placeholder path in this README.

## System Architecture

```mermaid
flowchart LR
      U[User Query] --> API[FastAPI /search]
      API --> RET[BM25 Retriever]
      RET --> TOP[Top-K Candidates]
      TOP --> FE[Feature Builder]
      FE --> RANK[LightGBM LambdaRank]
      RANK --> OUT[Final Ranked Results]
```

## Request Lifecycle

```mermaid
sequenceDiagram
      participant User
      participant UI as Streamlit UI
      participant API as FastAPI
      participant RET as BM25 Retriever
      participant LTR as LightGBM Reranker

      User->>UI: Enter query
      UI->>API: POST /search
      API->>RET: retrieve(query, top_k)
      RET-->>API: candidates + bm25_score
      API->>LTR: build features + predict
      LTR-->>API: model_score
      API-->>UI: ranked results
      UI-->>User: Render final list
```

## Feature Stack

| Group | Features |
|---|---|
| Lexical | `bm25_score`, `tfidf_cosine_sim`, `overlap_count`, `overlap_ratio`, `exact_match` |
| Structural | `query_len`, `doc_len` |
| Semantic | `semantic_cosine_sim` (Sentence Transformers) |

## Results Snapshot

Latest observed offline metrics from the held-out query split:

| Model | NDCG@5 | MRR | MAP |
|---|---:|---:|---:|
| BM25 Baseline | 0.8170 | 0.8333 | 0.7931 |
| BM25 + Reranker (Hybrid Features) | 0.9905 | 1.0000 | 0.9347 |

## Interactive Runbook

<details>
<summary><strong>1) Setup Environment</strong></summary>

```bash
pip install -r requirements.txt
```

</details>

<details>
<summary><strong>2) Generate Expanded Labeled Data (with hard negatives)</strong></summary>

```bash
python src/generate_training_data.py
```

</details>

<details>
<summary><strong>3) Train + Evaluate + Analyze</strong></summary>

```bash
python src/train.py
python src/evaluate.py
python src/analyze.py
```

</details>

<details>
<summary><strong>4) Run Inference Locally</strong></summary>

```bash
python src/predict.py
```

</details>

<details>
<summary><strong>5) Launch API</strong></summary>

```bash
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000/docs`

</details>

<details>
<summary><strong>6) Launch UI Demo</strong></summary>

```bash
streamlit run app/ui.py
```

</details>

<details>
<summary><strong>7) Docker Run</strong></summary>

```bash
docker build -t search-ranker .
docker run -p 8000:8000 search-ranker
```

</details>

## Large Dataset Pipeline (10-20 GB)

This project now includes a large-data training path using MS MARCO from online sources.

1. Build large training shards from online dataset (target size configurable):

```bash
python src/build_large_dataset_hf.py --target-gb 12 --output-dir data/large/msmarco_hf
```

2. Train a large ranker on generated shards:

```bash
python src/train_large_ranker.py --data-dir data/large/msmarco_hf --max-rows 1500000
```

3. Run two-stage ranking demo with large model:

```bash
python src/rank_large_demo.py --top-k 10
```

Outputs:

- `models/lgbm_ranker_large.pkl`
- `models/tfidf_vectorizer_large.pkl`
- `data/processed/train_featured_large.parquet`
- `data/processed/test_featured_large.parquet`
- `experiments/large_training_report.json`

## API Example

Request:

```json
{
   "query": "gaming laptop",
   "retrieve_top_k": 10
}
```

Response shape:

```json
{
   "query": "gaming laptop",
   "results": [
      {
         "doc_id": 3,
         "document": "lenovo legion gaming laptop",
         "category": "laptop",
         "bm25_score": 2.4467,
         "model_score": 7.9735
      }
   ]
}
```

## Experiment Tracking

- Experiment log: [experiments/experiments.csv](experiments/experiments.csv)
- Logger helper: [src/experiment_logger.py](src/experiment_logger.py)
- Ablation runner: [src/ablation.py](src/ablation.py)

Offline experiment pipeline:

```mermaid
flowchart TD
      A[catalog.csv + query specs] --> B[generate_training_data.py]
      B --> C[search_data_expanded.csv]
      C --> D[split.py GroupSplit by query]
      D --> E[features.py]
      E --> F[train.py]
      F --> G[evaluate.py]
      G --> H[experiment_logger.py]
      H --> I[experiments/experiments.csv]
```

## Project Structure

```text
search-ranking-model/
├── app/
│   ├── main.py
│   └── ui.py
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
│   └── experiments.csv
├── models/
├── src/
│   ├── generate_training_data.py
│   ├── retrieve.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── analyze.py
│   ├── ablation.py
│   ├── experiment_logger.py
│   └── predict.py
├── Dockerfile
└── requirements.txt
```

## Roadmap

- Real user behavior labels (clicks, add-to-cart, purchase)
- Hard-negative mining from larger corpus
- Embedding cache for lower latency
- Experiment tracking with run IDs and config hashes
- CI pipeline for regression checks
