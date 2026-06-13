

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _from_streamlit_secrets(key: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore

        if key not in st.secrets:
            return None
        raw = st.secrets[key]
        if raw is None:
            return None
        s = str(raw).strip()
        return s if s else None
    except Exception:
        return None


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Prefer Streamlit Cloud secrets, then process environment."""
    s = _from_streamlit_secrets(key)
    if s is not None:
        return s
    v = os.getenv(key)
    if v is not None and str(v).strip():
        return str(v).strip()
    return default


def get_chroma_persist_dir() -> str:
    """Persisted Chroma path: secret/env CHROMA_PERSIST_DIR, else ./chroma_db."""
    override = get_secret("CHROMA_PERSIST_DIR")
    if override:
        p = Path(override).expanduser()
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        return str(p)
    return str(BASE_DIR / "chroma_db")


def hf_offline_enabled() -> bool:
    v = get_secret("HF_OFFLINE")
    return str(v or "").lower() in ("1", "true", "yes")


def apply_hf_hub_env() -> None:
    """Allow HF downloads by default (Cloud); set HF_OFFLINE=1 locally if needed."""
    if hf_offline_enabled():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
