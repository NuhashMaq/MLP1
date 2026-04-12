from __future__ import annotations

import requests
import streamlit as st


st.set_page_config(page_title="Search Ranking Demo", page_icon="🔎", layout="centered")
st.title("Search Ranking Demo")
st.caption("Two-stage ranking: BM25 retrieval + LightGBM reranking")

api_url = st.text_input("API URL", value="http://127.0.0.1:8000/search")
query = st.text_input("Enter your search query", value="gaming laptop")
retrieve_top_k = st.slider("Retrieve top-k", min_value=3, max_value=20, value=10)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            response = requests.post(
                api_url,
                json={"query": query, "retrieve_top_k": retrieve_top_k},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results", [])

            st.subheader(f"Results ({len(results)})")
            for idx, item in enumerate(results, start=1):
                st.markdown(f"**{idx}. {item.get('document', '')}**")
                st.caption(
                    " | ".join(
                        [
                            f"category: {item.get('category', 'n/a')}",
                            f"bm25: {item.get('bm25_score', 0):.3f}",
                            f"model: {item.get('model_score', 0):.3f}",
                        ]
                    )
                )
                st.divider()
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
            st.info("Start API first: uvicorn app.main:app --reload")
