"""
Streamlit UI for the GM Meetings RAG bot.

Local:
  streamlit run streamlit_app.py

Streamlit Community Cloud:
  Main file: streamlit_app.py
  Secrets: GROQ_API_KEY (see secrets.toml.example / DEPLOYMENT.md)
  Commit `chroma_db/` (or set CHROMA_PERSIST_DIR) so the index exists at runtime.

Dynamic behaviour:
  - Year and document-type filters are populated from the live ChromaDB index,
    so they update automatically when new meeting data is ingested.
  - Use the "Refresh index" button in the sidebar after running
    `python ingest.py --update` to reload without restarting the server.
"""

from __future__ import annotations

import streamlit as st


@st.cache_resource
def get_engine():
    """Load embeddings, Chroma, and LLM once per process (not on every widget interaction)."""
    from rag import GMRagEngine

    return GMRagEngine()


def render_response(resp) -> None:
    st.markdown("### Answer")
    st.markdown(resp.answer)

    st.markdown("### Key points")
    for pt in resp.key_points:
        st.markdown(f"- {pt}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Relevant years**")
        st.write(", ".join(resp.relevant_years) or "—")
    with c2:
        st.markdown("**Document types**")
        st.write(", ".join(resp.document_types) or "—")
    with c3:
        conf = resp.confidence.upper()
        if resp.confidence == "high":
            st.success(f"**Confidence:** {conf}")
        elif resp.confidence == "medium":
            st.warning(f"**Confidence:** {conf}")
        else:
            st.error(f"**Confidence:** {conf}")

    st.markdown("### Sources")
    if resp.sources:
        for src in resp.sources:
            st.markdown(f"- `{src}`")
    else:
        st.caption("No source filenames listed.")


def main() -> None:
    st.set_page_config(
        page_title="GM Meetings RAG",
        layout="wide",
    )
    st.title("GM Meetings RAG")
    st.caption("Query OEKO-TEX GM meeting documents (2022–2025) via retrieval + Llama on Groq.")

    from env_config import get_secret

    if not get_secret("GROQ_API_KEY"):
        st.error(
            "Missing **GROQ_API_KEY**. On Streamlit Cloud, add it under **App settings → Secrets**. "
            "Locally, set it in `.env` or `.streamlit/secrets.toml`."
        )
        st.stop()

    engine = get_engine()

    with st.sidebar:
        st.header("Filters")

        # Year filter — populated from the live index so new meeting years
        # appear automatically after an --update ingest.
        live_years = engine.get_years()
        year_options = ["All years"] + live_years
        year_choice = st.selectbox("Year", year_options, index=0)
        year_filter = None if year_choice == "All years" else year_choice

        # Doc-type filter — populated from the live index as well.
        try:
            stats = engine.collection_stats()
            dtype_keys = sorted(stats.get("by_doc_type", {}).keys())
        except Exception:
            dtype_keys = []
        doc_options = ["All types"] + dtype_keys
        doc_choice = st.selectbox("Document type", doc_options, index=0)
        doc_type_filter = None if doc_choice == "All types" else doc_choice

        st.divider()

        if st.button("Show index stats"):
            try:
                s = engine.collection_stats()
                st.metric("Total chunks", s["total_chunks"])
                st.metric("Chart-OCR chunks", s["chart_ocr_enriched"])
                st.json({
                    "by_year":        s["by_year"],
                    "by_doc_type":    s["by_doc_type"],
                    "year_locations": s.get("year_locations", {}),
                })
            except Exception as e:
                st.error(str(e))

        st.divider()

        st.caption("After ingesting new data with `python ingest.py --update`, "
                   "click below to reload the index without restarting the server.")
        if st.button("Refresh index", type="secondary"):
            get_engine.clear()
            st.success("Index cache cleared — reloading on next query.")
            st.rerun()

    question = st.text_area(
        "Your question",
        placeholder="e.g. What were the main sustainability themes in the 2023 Budapest meeting?",
        height=120,
    )
    ask = st.button("Ask", type="primary", use_container_width=True)

    if ask:
        q = (question or "").strip()
        if not q:
            st.warning("Enter a question first.")
        else:
            with st.spinner("Retrieving context and generating answer…"):
                try:
                    resp = engine.query(
                        q,
                        year_filter=year_filter,
                        doc_type_filter=doc_type_filter,
                    )
                except Exception as e:
                    st.error(f"Query failed: {e}")
                    return
            render_response(resp)


if __name__ == "__main__":
    main()
