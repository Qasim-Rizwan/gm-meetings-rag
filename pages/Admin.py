"""
GM Meetings RAG — Admin Portal
================================
Accessible via the sidebar navigation.
Shows a login gate; once authenticated reveals the admin dashboard.

Credentials:
  Username : fahad
  Password : fahad09@
"""

from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from app_utils import ADMIN_USERNAME, ADMIN_PASSWORD, get_engine
from streamlit_theme import apply_theme, hero_header, sidebar_brand

st.set_page_config(
    page_title="Admin — GM Meetings RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme("admin")

# ── Session state defaults ─────────────────────────────────────────────────────
st.session_state.setdefault("admin_logged_in", False)

# ── Sidebar (this page loads its own script — keep branding consistent) ──────
with st.sidebar:
    sidebar_brand(subtitle="Admin console", icon_color="#d97706")
    st.markdown(
        '<p style="font-size:0.72rem;color:#94a3b8;margin:0;text-transform:uppercase;letter-spacing:0.06em;">Navigate</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Use the sidebar links above to return to the main chat.")


# ══════════════════════════════════════════════════════════════════════════════
# LOGIN GATE
# ══════════════════════════════════════════════════════════════════════════════

def render_login() -> None:
    hero_header(
        "Admin Portal",
        "Secure access to document ingestion and the LLM system prompt. "
        "Sign in below to continue.",
    )
    col_left, col_center, col_right = st.columns([1, 1.35, 1])
    with col_center:
        with st.container(border=True):
            st.markdown("### Sign in")
            st.caption("Administrator credentials required.")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            st.markdown("")  # spacer
            if st.button("Sign in", type="primary", use_container_width=True):
                if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                    st.session_state["admin_logged_in"] = True
                    st.rerun()
                else:
                    st.error("Incorrect username or password. Please try again.")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — ZIP extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_pdfs_from_zip(zip_bytes: bytes, dest_dir: Path) -> List[Path]:
    """
    Extract a ZIP archive to *dest_dir* and return every .pdf path found,
    including PDFs inside sub-folders.  Skips macOS __MACOSX junk entries.
    """
    zip_tmp = dest_dir / "_upload.zip"
    zip_tmp.write_bytes(zip_bytes)

    pdf_paths: List[Path] = []
    with zipfile.ZipFile(zip_tmp, "r") as zf:
        for member in zf.infolist():
            # Skip directories, hidden macOS metadata, and non-PDF files
            name_lower = member.filename.lower()
            if (
                member.is_dir()
                or "__macosx" in name_lower
                or not name_lower.endswith(".pdf")
            ):
                continue
            # Flatten into dest_dir — use just the basename to avoid path traversal
            safe_name = Path(member.filename).name
            target = dest_dir / safe_name
            # Handle duplicate names by appending a counter
            counter = 1
            while target.exists():
                stem = Path(safe_name).stem
                target = dest_dir / f"{stem}_{counter}.pdf"
                counter += 1
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            pdf_paths.append(target)

    zip_tmp.unlink(missing_ok=True)
    return pdf_paths


def _collect_pdf_paths(uploaded_files, tmp_dir: Path) -> List[Tuple[str, Path]]:
    """
    Given the raw Streamlit uploaded-file objects (PDF or ZIP), return a flat
    list of (display_name, pdf_path) tuples with every PDF ready to ingest.
    ZIP files are automatically extracted; nested sub-folders are walked.
    """
    results: List[Tuple[str, Path]] = []

    for uf in uploaded_files:
        name_lower = uf.name.lower()

        if name_lower.endswith(".zip"):
            # Extract ZIP to a sub-folder so names don't collide across ZIPs
            zip_dir = tmp_dir / Path(uf.name).stem
            zip_dir.mkdir(parents=True, exist_ok=True)
            try:
                pdfs = _extract_pdfs_from_zip(uf.getbuffer(), zip_dir)
            except Exception as exc:
                st.error(f"**{uf.name}** — could not extract ZIP: {exc}")
                continue
            if not pdfs:
                st.warning(f"**{uf.name}** — no PDF files found inside the ZIP.")
            for pdf_path in pdfs:
                results.append((f"{uf.name}/{pdf_path.name}", pdf_path))

        elif name_lower.endswith(".pdf"):
            pdf_path = tmp_dir / uf.name
            # Handle duplicate names (unlikely but safe)
            counter = 1
            while pdf_path.exists():
                pdf_path = tmp_dir / f"{Path(uf.name).stem}_{counter}.pdf"
                counter += 1
            pdf_path.write_bytes(uf.getbuffer())
            results.append((uf.name, pdf_path))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD — DOCUMENTS TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_documents_tab() -> None:
    st.markdown("### Upload Documents to Knowledge Base")
    st.caption(
        "Upload individual **PDF** files **or a ZIP archive** containing PDFs "
        "(including nested folders inside the ZIP). "
        "All PDFs are chunked and embedded into the ChromaDB vector store. "
        "The RAG engine reloads automatically after a successful upload."
    )

    with st.form("upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Drag & drop PDF or ZIP files here, or click Browse",
            type=["pdf", "zip"],
            accept_multiple_files=True,
            help="You can mix PDFs and ZIPs in a single upload.",
        )

        st.markdown("#### Document Metadata")
        st.caption(
            "These values are applied to **all** documents in this batch. "
            "Run separate uploads if files belong to different years."
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            year = st.text_input(
                "Year",
                value="2025",
                help="The year this document belongs to (e.g. 2025, 2026 …).",
            )

        with col2:
            _known_locations = {
                "2022": "Rome",
                "2023": "Budapest",
                "2024": "Alcoy",
                "2025": "Copenhagen",
            }
            location = st.text_input(
                "Location / City",
                value=_known_locations.get(year.strip(), ""),
                help="Meeting location — type freely (e.g. Berlin, Tokyo …).",
            )

        with col3:
            doc_type_options = [
                "auto-detect",
                "meeting_minutes",
                "presentation",
                "finance_report",
                "marketing_report",
                "quality_report",
                "sustainability_report",
                "technical_update",
                "legal_document",
                "schedule",
                "document",
            ]
            doc_type_choice = st.selectbox(
                "Document type",
                doc_type_options,
                index=0,
                help="Leave as 'auto-detect' to infer from filename.",
            )

        submitted = st.form_submit_button(
            "Process & Ingest", type="primary", use_container_width=True
        )

    if not submitted:
        return

    if not uploaded_files:
        st.warning("Please select at least one PDF or ZIP file before submitting.")
        return

    year_val = year.strip()
    if not year_val:
        st.error("Year cannot be empty.")
        return

    location_val = location.strip() or year_val
    doc_type = None if doc_type_choice == "auto-detect" else doc_type_choice

    # ── Resolve every uploaded file into flat PDF paths ────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="rag_upload_"))
    try:
        with st.spinner("Extracting files…"):
            pdf_jobs = _collect_pdf_paths(uploaded_files, tmp_dir)

        if not pdf_jobs:
            st.warning("No PDF files found in the uploaded files.")
            return

        zip_count  = sum(1 for uf in uploaded_files if uf.name.lower().endswith(".zip"))
        file_count = sum(1 for uf in uploaded_files if uf.name.lower().endswith(".pdf"))
        total_pdfs = len(pdf_jobs)

        summary_parts = []
        if file_count:
            summary_parts.append(f"{file_count} PDF file(s)")
        if zip_count:
            summary_parts.append(f"{zip_count} ZIP archive(s)")
        st.info(
            f"Found **{total_pdfs} PDF(s)** to ingest "
            f"from {' + '.join(summary_parts)}."
        )

        # ── Ingest each PDF ────────────────────────────────────────────────────
        from ingest import ingest_uploaded_file

        success_count = 0
        progress = st.progress(0, text="Starting ingestion…")

        for i, (display_name, pdf_path) in enumerate(pdf_jobs, 1):
            progress.progress(
                (i - 1) / total_pdfs,
                text=f"Ingesting {display_name}  ({i}/{total_pdfs})…",
            )
            try:
                n_chunks = ingest_uploaded_file(
                    file_path=str(pdf_path),
                    year=year_val,
                    location=location_val,
                    doc_type=doc_type,
                    chart_mode="none",
                )
                success_count += 1
                st.success(
                    f"**{display_name}** — {n_chunks} chunks added."
                )
            except Exception as exc:
                st.error(f"**{display_name}** — failed: {exc}")

        progress.progress(1.0, text="Done.")

    finally:
        # Always clean up temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if success_count > 0:
        st.cache_resource.clear()
        st.info(
            f"{success_count} of {total_pdfs} PDF(s) ingested successfully. "
            "The RAG engine will reload on the next query."
        )

    # ── Collection statistics ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Current Knowledge Base Statistics")
    try:
        s = get_engine().collection_stats()
        m1, m2 = st.columns(2)
        m1.metric("Total chunks", s["total_chunks"])
        m2.metric("Chart-OCR chunks", s["chart_ocr_enriched"])
        st.json({"by_year": s["by_year"], "by_doc_type": s["by_doc_type"]})
    except Exception as e:
        st.warning(f"Could not load stats: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD — PROMPT EDITOR TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_prompt_editor_tab() -> None:
    from rag import SYSTEM_PROMPT, get_system_prompt, save_system_prompt, reset_system_prompt

    st.markdown("### System Prompt Editor")
    st.caption(
        "Modify the system prompt that is sent to the LLM with every query. "
        "Saving a new prompt reloads the engine automatically so changes take "
        "effect on the very next question."
    )

    active_prompt = get_system_prompt()
    is_custom = active_prompt.strip() != SYSTEM_PROMPT.strip()

    if is_custom:
        st.info("A **custom prompt** is currently active.", icon="✏️")
    else:
        st.success("Using the **built-in default prompt**.", icon="✅")

    edited = st.text_area(
        "System prompt",
        value=active_prompt,
        height=450,
        label_visibility="collapsed",
    )

    col_save, col_reset, _ = st.columns([1, 1, 2])
    with col_save:
        if st.button("Save Prompt", type="primary", use_container_width=True):
            if not edited.strip():
                st.error("Prompt cannot be empty.")
            else:
                save_system_prompt(edited)
                st.cache_resource.clear()
                st.success("Prompt saved. Engine reloading on next query.")
                st.rerun()
    with col_reset:
        if st.button("Reset to Default", use_container_width=True):
            reset_system_prompt()
            st.cache_resource.clear()
            st.success("Reverted to built-in default prompt.")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard() -> None:
    row_spacer, row_logout = st.columns([5, 1])
    with row_logout:
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state["admin_logged_in"] = False
            st.rerun()

    hero_header(
        "Admin Dashboard",
        "Upload PDFs or ZIP archives, tune the system prompt, and inspect "
        "knowledge-base statistics — all in one place.",
    )

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_docs, tab_prompt = st.tabs(["Documents", "Prompt Editor"])
    with tab_docs:
        render_documents_tab()
    with tab_prompt:
        render_prompt_editor_tab()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state["admin_logged_in"]:
    render_dashboard()
else:
    render_login()
