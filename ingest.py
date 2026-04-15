"""
GM Meetings RAG - Ingestion Pipeline (v4 — Local OCR for charts)
================================================================
Loads all PDF documents from the yearly GM Meeting folders,
detects chart/image-heavy pages, extracts visible text from those
slides with **local Tesseract OCR** (no cloud API, no rate limits),
and persists everything to a ChromaDB vector store.

Optional: `--groq-vision` uses Groq Llama 4 Scout instead of OCR(if you have quota). Use `GROQ_VISION_DELAY` seconds between calls
to reduce429 errors.

Run:
    python ingest.py --rebuild              # default: Tesseract OCR on charts
    python ingest.py --rebuild --no-chart-ocr   # skip chart enrichment entirely
    python ingest.py --rebuild --groq-vision    # cloud vision (rate-limited)

Requires for OCR: Tesseract installed on the system
  macOS: brew install tesseract
  Ubuntu: sudo apt install tesseract-ocr
"""

import sys
import time
import io
import base64
import warnings
import logging
from pathlib import Path
from typing import List, Optional, Literal
from functools import partial

print = partial(print, flush=True)

warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pymupdf").setLevel(logging.ERROR)

from env_config import apply_hf_hub_env, get_chroma_persist_dir, get_secret

apply_hf_hub_env()

import fitz
from PIL import Image, ImageEnhance
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).resolve().parent
DATA_DIR        = BASE_DIR / "GM 2022 - External Folder"
CHROMA_DIR      = get_chroma_persist_dir()
COLLECTION      = "gm_meetings"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

PPT_MAX_CHARS   = 2000
TEXT_CHUNK_SIZE = 1000
TEXT_CHUNK_OVL  = 200

CHART_TEXT_THRESHOLD = 200

OCR_DPI         = 300
# Tesseract: OEM 3 = default LSTM; PSM 6 = uniform text block (works well on slides)
TESSERACT_CONFIG = "--oem 3 --psm 6"

VISION_MODEL    = "meta-llama/llama-4-scout-17b-16e-instruct"
VISION_DPI      = 200

CHART_KEYWORDS = (
    "per institute", "round robin", "certificate origin",
    "failed certificate", "control testing", "sources",
    "development", "top 10", "failure", "stores and sources",
    "on-site visit", "overdue", "label origin",
)

VISION_PROMPT = (
    "This is a slide from an OEKO-TEX quality management report or "
    "presentation. It contains a chart, graph, or table with important "
    "numerical data. Extract ALL data visible in this image:\n"
    "- Every entity name (institute, country, parameter, product, etc.)\n"
    "- Every numerical value (percentage, count, rate)\n"
    "- Every colour code (green/yellow/red) and what it means\n"
    "- Year labels and comparison data\n"
    "- Overall summary statistics shown\n"
    "Format the output as a clean structured list. Be exhaustive — every "
    "single number matters for analysis."
)

FOLDER_META = {
    "GM 2022 - E":                      {"year": "2022", "location": "Rome"},
    "GM Meeting 2023_external folder":  {"year": "2023", "location": "Budapest"},
    "GM_Alcoy_24_external folder":      {"year": "2024", "location": "Alcoy"},
    "GM_Copenhagen_25_external folder": {"year": "2025", "location": "Copenhagen"},
}

PPT_KEYWORDS = ("ppt", "1_1 speed")

ChartMode = Literal["none", "local", "groq"]

_groq_client = None


def parse_cli() -> tuple[bool, ChartMode, float]:
    argv = set(sys.argv[1:])
    force = "--rebuild" in argv
    try:
        delay = float(get_secret("GROQ_VISION_DELAY") or "2.5")
    except (TypeError, ValueError):
        delay = 2.5
    if "--no-chart-ocr" in argv:
        mode: ChartMode = "none"
    elif "--groq-vision" in argv:
        mode = "groq"
    else:
        mode = "local"
    return force, mode, delay


def _tesseract_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_chart_data_local_ocr(page: fitz.Page) -> Optional[str]:
    """Render page and OCR with local Tesseract — no network, no API quota."""
    try:
        import pytesseract
    except ImportError:
        print("         !! pytesseract not installed: pip install pytesseract")
        return None

    try:
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        img = ImageEnhance.Contrast(img).enhance(1.35)
        text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        text = text.strip()
        if len(text) < 15:
            return None
        return text
    except Exception as exc:
        print(f"         !! OCR failed: {exc}")
        return None


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            print("\n[ERROR] GROQ_API_KEY not set — required for --groq-vision.\n")
            sys.exit(1)
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def extract_chart_data_groq(page: fitz.Page) -> Optional[str]:
    """Optional cloud vision — may hit rate limits on large corpora."""
    try:
        pix = page.get_pixmap(dpi=VISION_DPI)
        img_b64 = base64.b64encode(pix.tobytes("png")).decode()
        client = get_groq_client()
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": VISION_PROMPT},
                ]
            }],
            temperature=0.1,
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"         !! Groq vision failed: {exc}")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_ppt_pdf(filename: str) -> bool:
    return any(k in filename.lower() for k in PPT_KEYWORDS)


def is_chart_page(text: str, page) -> bool:
    if len(text) >= CHART_TEXT_THRESHOLD:
        return False
    images = page.get_images(full=True)
    if not images:
        return False
    text_lower = text.lower()
    return any(kw in text_lower for kw in CHART_KEYWORDS)


def infer_doc_type(filename: str) -> str:
    f = filename.lower()
    if "minutes" in f:
        return "meeting_minutes"
    if "ppt" in f or "presentation" in f or "annual" in f:
        return "presentation"
    if "finance" in f:
        return "finance_report"
    if "marketing" in f:
        return "marketing_report"
    if "quality" in f:
        return "quality_report"
    if "legal" in f or "green claims" in f or "directive" in f:
        return "legal_document"
    if "microfibre" in f:
        return "technical_update"
    if "quantis" in f:
        return "sustainability_report"
    if "jd" in f or "secretary" in f:
        return "job_description"
    if "schedule" in f or "speed meeting" in f:
        return "schedule"
    return "document"


def collect_pdfs() -> List[dict]:
    records = []
    for folder_name, meta in FOLDER_META.items():
        folder_path = DATA_DIR / folder_name
        if not folder_path.exists():
            print(f"  [WARN] Folder not found: {folder_path}")
            continue
        for pdf_path in sorted(folder_path.rglob("*.pdf")):
            records.append({
                "path":     pdf_path,
                "year":     meta["year"],
                "location": meta["location"],
                "folder":   folder_name,
                "filename": pdf_path.name,
                "doc_type": infer_doc_type(pdf_path.name),
                "is_ppt":   is_ppt_pdf(pdf_path.name),
            })
    return records


def load_and_enrich_documents(
    records: List[dict],
    chart_mode: ChartMode,
    groq_delay: float,
) -> List[Document]:
    all_docs: List[Document] = []
    total = len(records)
    chart_pages = 0
    enriched = 0

    if chart_mode == "local" and not _tesseract_available():
        print(
            "\n  [WARN] Tesseract not found or not working. Chart OCR disabled.\n"
            "         Install: macOS: brew install tesseract\n"
            "                   Ubuntu: sudo apt install tesseract-ocr\n"
            "         Or run with --no-chart-ocr, or --groq-vision if you have API quota.\n"
        )
        chart_mode = "none"

    extract_fn = None
    label = ""
    data_tag = "OCR"
    if chart_mode == "local":
        extract_fn = extract_chart_data_local_ocr
        label = "OCR"
        data_tag = "OCR"
        print(f"      Chart enrichment: local Tesseract (dpi={OCR_DPI})\n")
    elif chart_mode == "groq":
        extract_fn = extract_chart_data_groq
        label = "Groq vision"
        data_tag = "vision"
        print(f"      Chart enrichment: {label} (delay {groq_delay}s between pages)\n")

    for idx, rec in enumerate(records, 1):
        tag = "[PPT]" if rec["is_ppt"] else "     "
        print(f"  [{idx:02d}/{total}] {tag} {rec['filename']}")

        try:
            loader = PyMuPDFLoader(str(rec["path"]))
            pages = loader.load()
        except Exception as exc:
            print(f"         !! Failed to load: {exc}")
            continue

        fitz_doc = fitz.open(str(rec["path"]))

        for page_doc in pages:
            page_num = page_doc.metadata.get("page", 0)
            text = page_doc.page_content.strip()

            page_doc.metadata.update({
                "year":      rec["year"],
                "location":  rec["location"],
                "folder":    rec["folder"],
                "filename":  rec["filename"],
                "doc_type":  rec["doc_type"],
                "is_ppt":    str(rec["is_ppt"]),
                "source":    str(rec["path"]),
            })

            if extract_fn and page_num < len(fitz_doc):
                fitz_page = fitz_doc[page_num]
                if is_chart_page(text, fitz_page):
                    chart_pages += 1
                    extracted = extract_fn(fitz_page)
                    if chart_mode == "groq" and groq_delay > 0:
                        time.sleep(groq_delay)
                    if extracted:
                        enriched += 1
                        page_doc.page_content = (
                            f"{text}\n\n"
                            f"[EXTRACTED CHART DATA ({data_tag}) — {rec['year']} {rec['location']}]\n"
                            f"{extracted}"
                        )
                        page_doc.metadata["has_chart_ocr"] = "True"
                        print(f"         + {label}: pg {page_num + 1}")
                    else:
                        page_doc.metadata["has_chart_ocr"] = "False"
                else:
                    page_doc.metadata["has_chart_ocr"] = "False"
            else:
                page_doc.metadata["has_chart_ocr"] = "False"

            all_docs.append(page_doc)

        fitz_doc.close()

    print(f"\n      Chart pages detected: {chart_pages}")
    print(f"      Chart pages enriched: {enriched}")
    return all_docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=TEXT_CHUNK_SIZE,
        chunk_overlap=TEXT_CHUNK_OVL,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Document] = []
    kept_whole = 0
    split_count = 0

    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue

        has_ocr = doc.metadata.get("has_chart_ocr") == "True"
        is_ppt = doc.metadata.get("is_ppt") == "True"

        if has_ocr or is_ppt or len(text) <= PPT_MAX_CHARS:
            chunks.append(Document(page_content=text, metadata=doc.metadata))
            kept_whole += 1
        else:
            sub = splitter.split_documents([doc])
            sub = [c for c in sub if c.page_content.strip()]
            chunks.extend(sub)
            split_count += len(sub)

    print(f"      Pages kept whole  : {kept_whole}")
    print(f"      Pages split into  : {split_count} chunks")
    return chunks


def ingest(force_rebuild: bool = False, chart_mode: ChartMode = "local", groq_delay: float = 2.5) -> None:
    chroma_path = Path(CHROMA_DIR)

    if chroma_path.exists() and not force_rebuild:
        print(
            f"\n[INFO] ChromaDB already exists at '{CHROMA_DIR}'.\n"
            "       Pass --rebuild to force a fresh index.\n"
        )
        return

    print("\n" + "=" * 64)
    print("  GM Meetings RAG — Index Builder (v4)")
    print("  Charts: local Tesseract OCR by default (no API limits)")
    print("=" * 64)

    print("\n[1/4] Discovering PDF documents …")
    records = collect_pdfs()
    ppt_count = sum(1 for r in records if r["is_ppt"])
    print(f"      Found {len(records)} PDFs  "
          f"({ppt_count} PPT-slides, {len(records) - ppt_count} text docs)\n")

    print("[2/4] Loading pages + chart enrichment …")
    print(f"      Chart mode: {chart_mode}")
    print(f"      Chart keywords + <{CHART_TEXT_THRESHOLD} chars + embedded images\n")
    docs = load_and_enrich_documents(records, chart_mode, groq_delay)
    print(f"      Loaded {len(docs)} pages total.\n")

    print("[3/4] Chunking …")
    chunks = chunk_documents(docs)
    print(f"\n      Total chunks: {len(chunks)}\n")

    from collections import Counter
    year_counts = Counter(c.metadata["year"] for c in chunks)
    ocr_chunks = sum(1 for c in chunks if c.metadata.get("has_chart_ocr") == "True")
    for yr in sorted(year_counts):
        print(f"      {yr}: {year_counts[yr]} chunks")
    print(f"      OCR-enriched chunks: {ocr_chunks}\n")

    print("[4/4] Embedding + persisting to ChromaDB …")
    print(f"      Model     : {EMBED_MODEL}")
    print(f"      Directory : {CHROMA_DIR}\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if chroma_path.exists() and force_rebuild:
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("      Removed old ChromaDB.\n")

    t0 = time.time()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    elapsed = time.time() - t0

    print(f"\n[DONE] Index built in {elapsed:.1f}s")
    print(f"       '{COLLECTION}' holds {vectorstore._collection.count()} vectors.")
    print(f"       Persisted to: {CHROMA_DIR}\n")


if __name__ == "__main__":
    force, mode, delay = parse_cli()
    ingest(force_rebuild=force, chart_mode=mode, groq_delay=delay)
