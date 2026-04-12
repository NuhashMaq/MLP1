from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.predict import search


class SearchRequest(BaseModel):
    query: str
    retrieve_top_k: int = 10


app = FastAPI(title="Search Ranking Model API")


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Two-stage Search Ranking API is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search")
def run_search(request: SearchRequest) -> dict[str, object]:
    results = search(query=request.query, retrieve_top_k=request.retrieve_top_k)
    return {
        "query": request.query,
        "results": results.to_dict(orient="records"),
    }
