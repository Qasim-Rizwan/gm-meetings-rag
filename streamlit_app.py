"""
GM Meetings RAG — Chat Page
============================
Main entry point.  Admin portal is available as a separate page (see sidebar).

Local:
  streamlit run streamlit_app.py

Streamlit Community Cloud:
  Main file : streamlit_app.py
  Secrets   : GROQ_API_KEY  (see secrets.toml.example / DEPLOYMENT.md)
"""

from __future__ import annotations

import streamlit as st
from app_utils import get_engine, get_available_years, get_available_doc_types
from streamlit_theme import apply_theme, hero_header, sidebar_brand

st.set_page_config(
    page_title="GM Meetings RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme("chat")


# ── Response renderer ──────────────────────────────────────────────────────────
def render_response(resp) -> None:
    with st.container(border=True):
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── GROQ key guard ─────────────────────────────────────────────────────────
    from env_config import get_secret

    if not get_secret("GROQ_API_KEY"):
        st.error(
            "Missing **GROQ_API_KEY**. On Streamlit Cloud, add it under "
            "**App settings → Secrets**. Locally, set it in `.env`."
        )
        st.stop()

    # ── Sidebar — filters ──────────────────────────────────────────────────────
    with st.sidebar:
        sidebar_brand(subtitle="Knowledge assistant")
        st.markdown('<p style="font-size:0.72rem;color:#94a3b8;margin:0.5rem 0 0.25rem 0;text-transform:uppercase;letter-spacing:0.06em;">Navigate</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Filters**")

        # Year filter — built from actual collection metadata (fully dynamic)
        available_years = get_available_years()
        year_options    = ["All years"] + available_years
        year_choice     = st.selectbox(
            "Year",
            year_options,
            index=0,
            help="Years are loaded from the documents in the knowledge base.",
        )
        year_filter = None if year_choice == "All years" else year_choice

        # Doc-type filter — also fully dynamic
        available_types = get_available_doc_types()
        doc_options     = ["All types"] + available_types
        doc_choice      = st.selectbox("Document type", doc_options, index=0)
        doc_type_filter = None if doc_choice == "All types" else doc_choice

        st.markdown("---")
        if st.button("Show index stats", use_container_width=True):
            try:
                s = get_engine().collection_stats()
                st.metric("Total chunks", s["total_chunks"])
                st.metric("Chart-OCR chunks", s["chart_ocr_enriched"])
                st.json({"by_year": s["by_year"], "by_doc_type": s["by_doc_type"]})
            except Exception as e:
                st.error(str(e))

    # ── Page header ────────────────────────────────────────────────────────────
    hero_header(
        "GM Meetings RAG",
        "Ask questions in natural language — answers are grounded in your document "
        "knowledge base. Switch to **Admin** in the sidebar to upload PDFs or ZIPs "
        "and edit the system prompt.",
    )

    # ── Chat interface ─────────────────────────────────────────────────────────
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
                    resp = get_engine().query(
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
