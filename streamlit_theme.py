"""
Polished visual theme for the Streamlit multi-page RAG app.

Injected via unsafe_allow_html CSS — safe here because the stylesheet is static
(no user input).  Call apply_theme() once per page, immediately after
st.set_page_config().
"""

from __future__ import annotations

import html
import streamlit as st

# ── Accent: chat = indigo–teal, admin = amber–rose (subtle distinction) ────────
_VARIANTS = {
    "chat": {
        "accent1": "#6366f1",
        "accent2": "#14b8a6",
        "accent_soft": "rgba(99, 102, 241, 0.14)",
        "glow": "rgba(99, 102, 241, 0.28)",
    },
    "admin": {
        "accent1": "#d97706",
        "accent2": "#e11d48",
        "accent_soft": "rgba(217, 119, 6, 0.14)",
        "glow": "rgba(217, 119, 6, 0.25)",
    },
}


def apply_theme(variant: str = "chat") -> None:
    """Inject global CSS + optional per-page accent."""
    v = _VARIANTS.get(variant, _VARIANTS["chat"])
    accent1 = v["accent1"]
    accent2 = v["accent2"]
    accent_soft = v["accent_soft"]
    glow = v["glow"]
    main_max = "1120px" if variant == "admin" else "920px"

    css = f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Outfit:wght@500;600;700&display=swap" rel="stylesheet">

    <style>
      /* ── Root & app shell ─────────────────────────────────────────────── */
      html, body, [class*="css"] {{
        font-family: 'DM Sans', system-ui, sans-serif !important;
      }}
      .stApp {{
        background: linear-gradient(165deg, #f0f4ff 0%, #f8fafc 38%, #ecfeff 100%) !important;
        background-attachment: fixed !important;
      }}
      .main .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: {main_max} !important;
      }}
      [data-testid="stAppViewContainer"] > .main {{
        background: transparent !important;
      }}

      /* ── Sidebar ─────────────────────────────────────────────────────── */
      [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        border-right: 1px solid rgba(15, 23, 42, 0.08) !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.06) !important;
      }}
      [data-testid="stSidebar"] .block-container {{
        padding-top: 1.25rem !important;
      }}

      /* ── Multipage nav — pill-style links ─────────────────────────────── */
      [data-testid="stSidebarNav"] {{
        padding-top: 0.5rem !important;
      }}
      [data-testid="stSidebarNav"] ul {{
        gap: 0.35rem !important;
        display: flex !important;
        flex-direction: column !important;
      }}
      [data-testid="stSidebarNav"] li {{
        margin: 0 !important;
        list-style: none !important;
      }}
      [data-testid="stSidebarNav"] a {{
        display: block !important;
        padding: 0.55rem 0.9rem !important;
        border-radius: 12px !important;
        text-decoration: none !important;
        color: #334155 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em !important;
        transition: background 0.2s ease, color 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease !important;
        border: 1px solid transparent !important;
      }}
      [data-testid="stSidebarNav"] a:hover {{
        background: {accent_soft} !important;
        color: #0f172a !important;
        transform: translateX(2px) !important;
      }}
      [data-testid="stSidebarNav"] a[aria-current="page"],
      [data-testid="stSidebarNav"] a:focus-visible {{
        background: linear-gradient(135deg, {accent_soft} 0%, rgba(255, 255, 255, 0.92) 100%) !important;
        color: #0f172a !important;
        border-color: rgba(15, 23, 42, 0.12) !important;
        border-left: 3px solid {accent1} !important;
        box-shadow: 0 2px 12px {glow} !important;
        font-weight: 600 !important;
      }}
      /* Streamlit sometimes wraps the active state differently */
      [data-testid="stSidebarNav"] span[data-testid="stMarkdownContainer"] p {{
        margin: 0 !important;
      }}

      /* ── Typography ───────────────────────────────────────────────────── */
      h1, h2, h3, .rag-brand-title {{
        font-family: 'Outfit', 'DM Sans', sans-serif !important;
        letter-spacing: -0.02em !important;
        color: #0f172a !important;
      }}
      h1 {{ font-weight: 700 !important; }}

      /* ── Hero card (main chat) ─────────────────────────────────────────── */
      .rag-hero {{
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 20px;
        padding: 1.75rem 2rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow:
          0 4px 6px -1px rgba(15, 23, 42, 0.06),
          0 24px 48px -12px rgba(99, 102, 241, 0.12);
      }}
      .rag-hero h1 {{
        margin: 0 0 0.35rem 0 !important;
        font-size: 2rem !important;
        background: linear-gradient(120deg, {accent1} 0%, {accent2} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }}
      .rag-hero .rag-sub {{
        color: #64748b !important;
        font-size: 1.02rem !important;
        line-height: 1.55 !important;
        margin: 0 !important;
      }}

      /* ── Sidebar brand strip ──────────────────────────────────────────── */
      .rag-sidebar-brand {{
        padding: 0.5rem 0 1.25rem 0;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(15, 23, 42, 0.06);
      }}
      .rag-sidebar-brand .rag-logo {{
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
      }}
      .rag-sidebar-brand .rag-logo span {{
        font-size: 1.35rem;
        line-height: 1;
      }}
      .rag-sidebar-brand .rag-tagline {{
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.35rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }}

      /* ── Primary buttons ──────────────────────────────────────────────── */
      .stButton > button[kind="primary"],
      div[data-testid="stBaseButton-primary"] button {{
        background: linear-gradient(135deg, {accent1} 0%, {accent2} 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 4px 14px {glow} !important;
        transition: transform 0.15s ease, box-shadow 0.2s ease !important;
      }}
      .stButton > button[kind="primary"]:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px {glow} !important;
      }}

      /* ── Secondary / default buttons ───────────────────────────────────── */
      .stButton > button[kind="secondary"] {{
        border-radius: 12px !important;
        border: 1px solid rgba(15, 23, 42, 0.12) !important;
        font-weight: 500 !important;
      }}

      /* ── Inputs ───────────────────────────────────────────────────────── */
      .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div {{
        border-radius: 12px !important;
        border-color: rgba(15, 23, 42, 0.12) !important;
      }}
      .stTextArea textarea:focus, .stTextInput input:focus {{
        border-color: {accent1} !important;
        box-shadow: 0 0 0 3px {accent_soft} !important;
      }}

      /* ── Tabs ────────────────────────────────────────────────────────── */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: rgba(255,255,255,0.6);
        padding: 0.35rem;
        border-radius: 14px;
        border: 1px solid rgba(15, 23, 42, 0.08);
      }}
      .stTabs [data-baseweb="tab"] {{
        border-radius: 10px !important;
        padding: 0.5rem 1.1rem !important;
        font-weight: 600 !important;
      }}
      .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, {accent_soft} 0%, rgba(255,255,255,0.9) 100%) !important;
        color: #0f172a !important;
      }}

      /* ── Metrics & expanders ─────────────────────────────────────────── */
      [data-testid="stMetricValue"] {{
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
      }}

      /* ── Bordered containers (login card, etc.) ─────────────────────────── */
      [data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 22px !important;
        border: 1px solid rgba(15, 23, 42, 0.08) !important;
        background: linear-gradient(165deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.95) 100%) !important;
        box-shadow:
          0 20px 40px -16px rgba(15, 23, 42, 0.12),
          0 0 0 1px rgba(255,255,255,0.75) inset !important;
        padding: 0.35rem !important;
      }}

      /* ── Answer section polish ───────────────────────────────────────── */
      div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stMarkdownContainer"]) h3 {{
        color: #334155 !important;
        font-size: 1.05rem !important;
        margin-top: 1.25rem !important;
      }}

      /* ── Hide Streamlit menu / footer noise (optional, subtle) ─────────── */
      #MainMenu {{visibility: hidden;}}
      footer {{visibility: hidden;}}
      header[data-testid="stHeader"] {{
        background: transparent !important;
      }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _book_svg(icon_color: str) -> str:
    c = html.escape(icon_color, quote=True)
    return (
        f'<svg width="22" height="22" viewBox="0 0 24 24" fill="none" '
        f'xmlns="http://www.w3.org/2000/svg" style="vertical-align:middle;'
        f'flex-shrink:0;color:{c}"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round"/>'
        f'<path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>'
    )


def sidebar_brand(
    *, subtitle: str = "Knowledge assistant", icon_color: str = "#6366f1"
) -> None:
    """Render the branded header block at the top of the sidebar."""
    sub_safe = html.escape(subtitle)
    st.markdown(
        f"""
        <div class="rag-sidebar-brand">
          <div class="rag-logo">{_book_svg(icon_color)} RAG Assistant</div>
          <div class="rag-tagline">{sub_safe}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero_header(title: str, subtitle: str) -> None:
    """Large gradient title + subtitle for the main chat page."""
    t = html.escape(title)
    s = html.escape(subtitle)
    st.markdown(
        f'<div class="rag-hero"><h1>{t}</h1><p class="rag-sub">{s}</p></div>',
        unsafe_allow_html=True,
    )
