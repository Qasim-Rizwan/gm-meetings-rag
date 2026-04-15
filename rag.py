"""
GM Meetings RAG — Query Engine
==============================
Retrieves relevant document chunks from ChromaDB (including
chart text from local OCR when you ran `ingest.py --rebuild`),
using Sentence Transformer embeddings, then generates a structured
response via Llama on Groq.

Usage (interactive CLI):
    python rag.py

Usage (as module):
    from rag import GMRagEngine
    engine = GMRagEngine()
    result = engine.query("What were the key decisions in 2023?")
    print(result)
"""

import sys
import warnings
import logging
from pathlib import Path
from typing import List, Optional, Literal

from pydantic import BaseModel, Field

from env_config import apply_hf_hub_env, get_chroma_persist_dir, get_secret

apply_hf_hub_env()
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent
CHROMA_DIR    = get_chroma_persist_dir()
COLLECTION    = "gm_meetings"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.3-70b-versatile"
TOP_K         = 12

YEARS         = ["2022", "2023", "2024", "2025"]

# ── Structured Response Schema ────────────────────────────────────────────────

class GMResponse(BaseModel):
    """Structured answer produced by the RAG pipeline."""
    answer: str = Field(
        description="Comprehensive, well-structured answer to the question."
    )
    key_points: List[str] = Field(
        description="3–6 concise bullet points summarising the most important findings."
    )
    relevant_years: List[str] = Field(
        description="List of GM meeting years that the answer draws from (e.g. ['2022','2023'])."
    )
    document_types: List[str] = Field(
        description="Types of source documents referenced (e.g. meeting_minutes, presentation)."
    )
    sources: List[str] = Field(
        description="Filenames of the specific documents used to form the answer."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high   = answer is well-supported by multiple relevant chunks; "
            "medium = partial support; "
            "low    = limited or indirect evidence."
        )
    )

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert analyst of OEKO-TEX General Manager (GM) Meeting documents \
spanning 2022–2025 (locations: Rome 2022, Budapest 2023, Alcoy 2024, Copenhagen 2025).

Your task is to answer the user's question using ONLY the provided context chunks. \
Do not fabricate information. If the context is insufficient, say so clearly.

Return your answer as a valid JSON object that strictly follows this schema:
{schema}

Guidelines:
- "answer"         : Write 2–4 paragraphs. Be specific, cite years and meeting locations.
- "key_points"     : Extract 3–6 concrete takeaways. Start each with a verb.
- "relevant_years" : Only include years actually referenced in your answer.
- "document_types" : Include all distinct document types referenced.
- "sources"        : List unique filenames (no paths). Include every file you drew from.
- "confidence"     : Assess how well the context supports your answer.

IMPORTANT — Numerical & Statistical Data:
- Some context chunks contain [EXTRACTED CHART DATA (OCR)] or [(vision)] sections. \
These are text read from chart slides (local Tesseract OCR, or optional cloud vision \
during ingestion). Treat them as the best available machine-readable representation \
of the slide — OCR may have minor errors; cross-check numbers when possible.
- When the question asks for a rate, percentage, count, or ranking, prioritize data \
from [EXTRACTED CHART DATA] sections. Cite specific numbers.
- If raw numbers are present, calculate derived values (e.g. success rate, failure \
rate) and show your working in the answer.
- When comparing data across years, note which year each data point comes from.

Context chunks (each prefixed with its metadata):
{context}
"""

HUMAN_PROMPT = "Question: {question}"

# ── Engine ────────────────────────────────────────────────────────────────────

class GMRagEngine:
    def __init__(self) -> None:
        self._check_index()
        self._load_components()

    def _check_index(self) -> None:
        if not Path(CHROMA_DIR).exists():
            print(
                "\n[ERROR] ChromaDB index not found.\n"
                "        Run  python ingest.py  first to build the index.\n"
            )
            sys.exit(1)

    def _load_components(self) -> None:
        print("[RAG] Loading embedding model …")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("[RAG] Connecting to ChromaDB …")
        self.vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DIR,
        )

        api_key = get_secret("GROQ_API_KEY")
        if not api_key:
            print(
                "\n[ERROR] GROQ_API_KEY not set.\n"
                "        Local: set GROQ_API_KEY in .env\n"
                "        Streamlit Cloud: App settings → Secrets → GROQ_API_KEY\n"
                "        Get a key: https://console.groq.com/keys\n"
            )
            sys.exit(1)

        print(f"[RAG] Connecting to Groq / Llama ({GROQ_MODEL}) …")
        self.llm = ChatGroq(
            model=GROQ_MODEL,
            groq_api_key=api_key,
            temperature=0.2,
        )

        self.parser = JsonOutputParser(pydantic_object=GMResponse)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])
        print("[RAG] Ready.\n")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        year_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        k: int = TOP_K,
    ) -> List[Document]:
        where: dict = {}
        if year_filter and doc_type_filter:
            where = {"$and": [{"year": year_filter}, {"doc_type": doc_type_filter}]}
        elif year_filter:
            where = {"year": year_filter}
        elif doc_type_filter:
            where = {"doc_type": doc_type_filter}

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, **({"filter": where} if where else {})},
        )
        return retriever.invoke(question)

    # ── Format context ────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(docs: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            m = doc.metadata
            vision_tag = " [CHART-OCR]" if m.get("has_chart_ocr") == "True" else ""
            header = (
                f"[Chunk {i}{vision_tag}] "
                f"Year={m.get('year','?')} | "
                f"Location={m.get('location','?')} | "
                f"Type={m.get('doc_type','?')} | "
                f"File={m.get('filename','?')} | "
                f"Page={m.get('page','?')}"
            )
            parts.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        year_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> GMResponse:
        """Full RAG pipeline: retrieve → format → Llama (Groq) → structured response."""
        docs = self.retrieve(question, year_filter=year_filter, doc_type_filter=doc_type_filter)

        if not docs:
            return GMResponse(
                answer="No relevant documents were found for your query.",
                key_points=["No matching content in the knowledge base."],
                relevant_years=[],
                document_types=[],
                sources=[],
                confidence="low",
            )

        context = self._format_context(docs)
        schema  = self.parser.get_format_instructions()

        chain  = self.prompt | self.llm | self.parser
        result = chain.invoke({
            "schema":   schema,
            "context":  context,
            "question": question,
        })

        if isinstance(result, dict):
            return GMResponse(**result)
        return result

    # ── Convenience helpers ───────────────────────────────────────────────────

    def query_year(self, question: str, year: str) -> GMResponse:
        if year not in YEARS:
            raise ValueError(f"year must be one of {YEARS}")
        return self.query(question, year_filter=year)

    def collection_stats(self) -> dict:
        col = self.vectorstore._collection
        count = col.count()
        meta_sample = col.get(limit=count, include=["metadatas"])["metadatas"]
        from collections import Counter
        years = Counter(m.get("year", "?") for m in meta_sample)
        dtypes = Counter(m.get("doc_type", "?") for m in meta_sample)
        chart_ocr = sum(1 for m in meta_sample if m.get("has_chart_ocr") == "True")
        return {
            "total_chunks":         count,
            "chart_ocr_enriched":   chart_ocr,
            "by_year":              dict(sorted(years.items())),
            "by_doc_type":          dict(sorted(dtypes.items())),
        }


# ── Pretty Printer ────────────────────────────────────────────────────────────

def print_response(resp: GMResponse, question: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  QUERY: {question}")
    print("=" * width)

    print(f"\n📋 ANSWER\n{'-' * width}")
    print(resp.answer)

    print(f"\n✅ KEY POINTS\n{'-' * width}")
    for pt in resp.key_points:
        print(f"  • {pt}")

    print(f"\n📅 RELEVANT YEARS : {', '.join(resp.relevant_years) or 'N/A'}")
    print(f"📄 DOCUMENT TYPES : {', '.join(resp.document_types) or 'N/A'}")
    print(f"🔗 SOURCES\n{'-' * width}")
    for src in resp.sources:
        print(f"  - {src}")

    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(resp.confidence, "⚪")
    print(f"\n{conf_icon} CONFIDENCE : {resp.confidence.upper()}")
    print("=" * width + "\n")


# ── Interactive CLI ───────────────────────────────────────────────────────────

HELP_TEXT = """
Commands:
  <question>          — Ask anything about the GM meetings
  year <YYYY>         — Set a year filter (2022 / 2023 / 2024 / 2025)
  year clear          — Clear year filter
  stats               — Show collection statistics
  help                — Show this help message
  exit / quit         — Exit the program
"""

def run_cli(engine: GMRagEngine) -> None:
    print("\n" + "=" * 70)
    print("  GM Meetings RAG — Interactive Query Interface")
    print("  Data: OEKO-TEX GM Meetings 2022–2025 (chart OCR when indexed)")
    print("=" * 70)
    print(HELP_TEXT)

    year_filter: Optional[str] = None

    while True:
        try:
            prefix = f"[{year_filter}] " if year_filter else ""
            raw = input(f"{prefix}Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not raw:
            continue

        low = raw.lower()

        if low in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if low == "help":
            print(HELP_TEXT)
            continue

        if low == "stats":
            stats = engine.collection_stats()
            print(f"\n  Total chunks      : {stats['total_chunks']}")
            print(f"  Chart-OCR chunks  : {stats['chart_ocr_enriched']}")
            print(f"  By year           : {stats['by_year']}")
            print(f"  By doc type       : {stats['by_doc_type']}\n")
            continue

        if low.startswith("year "):
            arg = raw[5:].strip()
            if arg.lower() == "clear":
                year_filter = None
                print("  Year filter cleared.\n")
            elif arg in YEARS:
                year_filter = arg
                print(f"  Year filter set to {year_filter}.\n")
            else:
                print(f"  Unknown year '{arg}'. Choose from {YEARS} or 'clear'.\n")
            continue

        print("  Thinking …\n")
        try:
            resp = engine.query(raw, year_filter=year_filter)
            print_response(resp, raw)
        except Exception as exc:
            print(f"  [ERROR] {exc}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = GMRagEngine()
    run_cli(engine)
