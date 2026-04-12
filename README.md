# Two-Stage Search Ranking System (BM25 + ML Reranker)

Portfolio-grade search intelligence project that simulates real-world product search:

1. Retrieve candidate documents with BM25
2. Rerank top-K candidates with LightGBM LambdaRank using lexical + semantic features

## What This Project Does

- User query -> BM25 retriever -> top-K documents -> feature engineering -> reranker -> final ranking
- Demonstrates end-to-end ML system design: data generation, hard negative mining, ranking model training, evaluation, API serving, UI demo, and Docker packaging

## Architecture

```
User Query
   ↓
API Layer (FastAPI)
   ↓
BM25 Retriever
   ↓
Top-K Candidate Documents
   ↓
Feature Engine
   ↓
LightGBM Ranker (LambdaRank)
   ↓
Final Ranked Results
```

## Features

- `bm25_score`
- `tfidf_cosine_sim`
- `overlap_count`, `overlap_ratio`
- `exact_match`
- `semantic_cosine_sim` (Sentence Transformers)

## Experiment Evidence

- Experiment tracker: [experiments/experiments.csv](experiments/experiments.csv)
- Logger utility: [src/experiment_logger.py](src/experiment_logger.py)
- Ablation runner: [src/ablation.py](src/ablation.py)

## Repository Structure

- `data/raw/`: catalog and supervised training datasets
- `data/processed/`: generated feature tables
- `src/`: retrieval, feature engineering, training, evaluation, ablation, analysis, inference
- `app/`: FastAPI service and Streamlit demo UI
- `experiments/`: experiment logs
- `models/`: serialized model artifacts

## Quickstart

1. Create and activate virtual environment
2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Generate expanded labeled data with hard negatives

```bash
python src/generate_training_data.py
```

4. Train, evaluate, analyze, and run prediction

```bash
python src/train.py
python src/evaluate.py
python src/analyze.py
python src/predict.py
```

## API Demo

```bash
uvicorn app.main:app --reload
```

- Swagger: `http://127.0.0.1:8000/docs`

## UI Demo (Streamlit)

```bash
streamlit run app/ui.py
```

## Docker

```bash
docker build -t search-ranker .
docker run -p 8000:8000 search-ranker
```

## Outputs

- `models/lgbm_ranker.pkl`
- `models/tfidf_vectorizer.pkl`
- `data/processed/train_featured.csv`
- `data/processed/test_featured.csv`
