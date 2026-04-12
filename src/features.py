import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_SENTENCE_MODEL: SentenceTransformer | None = None


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_overlap(query: str, document: str) -> int:
    query_words = set(query.split())
    doc_words = set(document.split())
    return len(query_words.intersection(doc_words))


def exact_match(query: str, document: str) -> int:
    return int(query in document)


def get_sentence_model() -> SentenceTransformer:
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer(SEMANTIC_MODEL_NAME)
    return _SENTENCE_MODEL


def fit_vectorizer(df: pd.DataFrame, save_path: str = "models/tfidf_vectorizer.pkl") -> TfidfVectorizer:
    data = df.copy()
    data["query_clean"] = data["query"].apply(clean_text)
    data["document_clean"] = data["document"].apply(clean_text)

    combined_text = pd.concat([data["query_clean"], data["document_clean"]], axis=0)

    vectorizer = TfidfVectorizer()
    vectorizer.fit(combined_text)

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_path)

    return vectorizer


def add_bm25_scores(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    bm25_series = pd.Series(index=data.index, dtype=float)

    for query, group in data.groupby("query", sort=False):
        docs = group["document_clean"].tolist()
        tokenized_docs = [doc.split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)

        tokenized_query = clean_text(query).split()
        scores = bm25.get_scores(tokenized_query)

        bm25_series.loc[group.index] = scores

    data["bm25_score"] = bm25_series
    return data


def _encode_texts(sentence_model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    unique_texts = list(dict.fromkeys(texts))
    unique_embeddings = sentence_model.encode(
        unique_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedding_map = {text: embedding for text, embedding in zip(unique_texts, unique_embeddings)}
    return np.asarray([embedding_map[text] for text in texts])


def add_semantic_similarity(
    df: pd.DataFrame,
    sentence_model: SentenceTransformer | None = None,
) -> pd.DataFrame:
    data = df.copy()

    if sentence_model is None:
        sentence_model = get_sentence_model()

    query_embeddings = _encode_texts(sentence_model, data["query_clean"].tolist())
    doc_embeddings = _encode_texts(sentence_model, data["document_clean"].tolist())

    semantic_scores = np.sum(query_embeddings * doc_embeddings, axis=1)
    data["semantic_cosine_sim"] = semantic_scores
    return data


def build_features(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer | None = None,
    add_bm25_feature: bool = True,
    add_semantic_feature: bool = True,
    sentence_model: SentenceTransformer | None = None,
) -> pd.DataFrame:
    data = df.copy()

    data["query_clean"] = data["query"].apply(clean_text)
    data["document_clean"] = data["document"].apply(clean_text)

    data["query_len"] = data["query_clean"].apply(lambda x: len(x.split()))
    data["doc_len"] = data["document_clean"].apply(lambda x: len(x.split()))

    data["overlap_count"] = data.apply(
        lambda row: word_overlap(row["query_clean"], row["document_clean"]), axis=1
    )
    data["overlap_ratio"] = data["overlap_count"] / data["query_len"].replace(0, 1)

    data["exact_match"] = data.apply(
        lambda row: exact_match(row["query_clean"], row["document_clean"]), axis=1
    )

    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        combined_text = pd.concat([data["query_clean"], data["document_clean"]], axis=0)
        vectorizer.fit(combined_text)

    query_vectors = vectorizer.transform(data["query_clean"])
    doc_vectors = vectorizer.transform(data["document_clean"])

    similarities = []
    for idx in range(data.shape[0]):
        similarity = cosine_similarity(query_vectors[idx], doc_vectors[idx])[0][0]
        similarities.append(similarity)

    data["tfidf_cosine_sim"] = similarities

    if add_bm25_feature:
        data = add_bm25_scores(data)

    if add_semantic_feature:
        data = add_semantic_similarity(data, sentence_model=sentence_model)

    return data
