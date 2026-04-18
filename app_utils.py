"""
Shared utilities for the Streamlit multi-page app.

Centralising get_engine() here ensures that both the chat page and the admin
page use the SAME @st.cache_resource cache entry — preventing the engine from
being loaded twice when a user navigates between pages.
"""

from __future__ import annotations

import streamlit as st

# ── Admin credentials ──────────────────────────────────────────────────────────
ADMIN_USERNAME = "fahad"
ADMIN_PASSWORD = "fahad09@"


# ── Shared cached engine ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG engine…")
def get_engine():
    """
    Load embeddings, ChromaDB, and Groq LLM once per server process.
    Cached at the process level so navigating between pages never re-loads it.
    Call  st.cache_resource.clear()  to force a reload (e.g. after ingestion).
    """
    from rag import GMRagEngine
    return GMRagEngine()


# ── Dynamic filter helpers ─────────────────────────────────────────────────────
def get_available_years() -> list[str]:
    """
    Return sorted list of years that actually exist in the vector store.
    Falls back to an empty list if the engine / collection is not ready.
    """
    try:
        stats = get_engine().collection_stats()
        return sorted(stats.get("by_year", {}).keys())
    except Exception:
        return []


def get_available_doc_types() -> list[str]:
    """
    Return sorted list of doc_type values that actually exist in the vector store.
    """
    try:
        stats = get_engine().collection_stats()
        return sorted(stats.get("by_doc_type", {}).keys())
    except Exception:
        return []
