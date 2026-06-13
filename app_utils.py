
from __future__ import annotations

import streamlit as st

# ── Admin credentials ──────────────────────────────────────────────────────────
ADMIN_USERNAME = "fahad"
ADMIN_PASSWORD = "fahad09@"


# ── Shared cached engine ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG engine…")
def get_engine():
   
    from rag import GMRagEngine
    return GMRagEngine()


# ── Dynamic filter helpers ─────────────────────────────────────────────────────
def get_available_years() -> list[str]:
    
    try:
        stats = get_engine().collection_stats()
        return sorted(stats.get("by_year", {}).keys())
    except Exception:
        return []


def get_available_doc_types() -> list[str]:
    
    try:
        stats = get_engine().collection_stats()
        return sorted(stats.get("by_doc_type", {}).keys())
    except Exception:
        return []
