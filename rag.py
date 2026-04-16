"""
GM Meetings RAG — Production-Grade Query Engine (v2)
====================================================
Architecture:
  QueryAnalyzer  →  HybridRetriever (Dense + BM25 + RRF + MultiQuery)
  →  CrossEncoder Reranker  →  ContextBuilder  →  Groq Llama  →  GMResponse

Usage (CLI):
    python rag.py

Usage (module):
    from rag import GMRagEngine
    engine = GMRagEngine()
    result = engine.query("What were the key decisions in 2023?")
    result = engine.query_with_filters("quality failures", year="2024")
"""

import re
import sys
import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple

from pydantic import BaseModel, Field

from env_config import apply_hf_hub_env, get_chroma_persist_dir, get_secret

apply_hf_hub_env()
warnings.filterwarnings("ignore")

logging.basicConfig(format="[%(levelname)s] %(name)s: %(message)s", level=logging.WARNING)
logger = logging.getLogger("gm_rag")
logger.setLevel(logging.INFO)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser

# ── Configuration ──────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).resolve().parent
CHROMA_DIR     = get_chroma_persist_dir()
COLLECTION     = "gm_meetings"
EMBED_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL     = "llama-3.3-70b-versatile"

# Cross-encoder: ms-marco-MiniLM-L-6-v2 is fast (6-layer); swap to
# BAAI/bge-reranker-base for ~10% accuracy gain at ~2× cost.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

YEARS          = ["2022", "2023", "2024", "2025"]
YEAR_LOCATIONS = {"2022": "Rome", "2023": "Budapest", "2024": "Alcoy", "2025": "Copenhagen"}

# Retrieval pool sizes before reranking
DENSE_TOP_K  = 20
BM25_TOP_K   = 20
RRF_K        = 60   # RRF smoothing constant — higher = flatter score distribution
RERANK_TOP_N = 7    # chunks actually sent to the LLM

# Multi-query expansion — rule-based rephrasings to widen recall; set 0 to disable
MULTI_QUERY_N = 2


# ── Structured Response Schema ─────────────────────────────────────────────────

class GMResponse(BaseModel):
    """Structured answer produced by the full RAG pipeline."""
    answer: str = Field(
        description="Comprehensive answer grounded strictly in the provided context."
    )
    key_points: List[str] = Field(
        description="3–6 bullet points, each starting with an action verb."
    )
    relevant_years: List[str] = Field(
        description="Years actually referenced in the answer (e.g. ['2023','2024'])."
    )
    document_types: List[str] = Field(
        description="Distinct source document types referenced."
    )
    sources: List[str] = Field(
        description="Unique filenames used to form the answer (no paths, no duplicates)."
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description=(
            "high   = 3+ strongly relevant chunks support the answer; "
            "medium = 1–2 relevant chunks, partial coverage; "
            "low    = weak, indirect, or single-chunk evidence."
        )
    )


# ── Query Understanding ────────────────────────────────────────────────────────

# Maps intent names to trigger keywords
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "comparison":     ["compare", "versus", "vs", "difference", "between", "trend", "evolution", "change"],
    "numeric":        ["how many", "percentage", "rate", "count", "number", "statistics", "data", "figure"],
    "decision":       ["decision", "decided", "agreed", "approved", "voted", "resolution", "concluded"],
    "summary":        ["summary", "overview", "main", "key", "highlight", "takeaway", "summarize"],
    "roadmap":        ["plan", "roadmap", "future", "strategy", "initiative", "goal", "objective", "priority"],
    "issue":          ["issue", "problem", "challenge", "concern", "risk", "failure", "mistake"],
    "financial":      ["budget", "cost", "revenue", "financial", "finance", "spend", "investment", "profit"],
    "marketing":      ["marketing", "campaign", "brand", "communication", "promotion", "awareness"],
    "quality":        ["quality", "audit", "control", "certificate", "compliance", "round robin", "testing"],
    "sustainability": ["sustainability", "environment", "climate", "carbon", "green", "emission"],
}

# Maps doc_type values to trigger keywords for auto-detection
DOC_TYPE_KEYWORDS: Dict[str, List[str]] = {
    "meeting_minutes":      ["minutes", "discussed", "agenda", "action item"],
    "presentation":         ["presentation", "slide", "ppt", "annual report"],
    "finance_report":       ["finance", "budget", "revenue", "cost"],
    "marketing_report":     ["marketing", "brand", "campaign"],
    "quality_report":       ["quality", "audit", "certificate", "round robin"],
    "legal_document":       ["legal", "directive", "green claims", "compliance", "regulation"],
    "sustainability_report": ["sustainability", "environment", "quantis", "carbon"],
}


@dataclass
class QueryContext:
    original_question:  str
    rewritten_query:    str                      # enriched query used for retrieval
    expanded_queries:   List[str] = field(default_factory=list)  # multi-query variants
    detected_years:     List[str] = field(default_factory=list)
    detected_doc_type:  Optional[str] = None
    intent:             str = "general"


class QueryAnalyzer:
    """
    Extracts metadata signals from the raw question:
      - year detection (numeric + location name)
      - intent classification
      - doc_type hint
    Then rewrites the query for better retrieval and optionally generates
    expanded query variants for multi-query retrieval.
    """

    _YEAR_RE    = re.compile(r"\b(202[2-5])\b")
    _YEAR_WORDS = {"rome": "2022", "budapest": "2023", "alcoy": "2024", "copenhagen": "2025"}

    def analyze(self, question: str) -> QueryContext:
        q_lower = question.lower()

        # ── Year detection ────────────────────────────────────────────────────
        years: List[str] = self._YEAR_RE.findall(question)
        for loc_word, yr in self._YEAR_WORDS.items():
            if loc_word in q_lower and yr not in years:
                years.append(yr)
        # Stable deduplicate
        seen: set = set()
        detected_years = [y for y in years if not (y in seen or seen.add(y))]  # type: ignore[func-returns-value]

        # ── Intent detection (first match wins) ───────────────────────────────
        intent = "general"
        for intent_name, kws in INTENT_KEYWORDS.items():
            if any(kw in q_lower for kw in kws):
                intent = intent_name
                break

        # ── Doc type hint ─────────────────────────────────────────────────────
        detected_doc_type: Optional[str] = None
        for dtype, kws in DOC_TYPE_KEYWORDS.items():
            if any(kw in q_lower for kw in kws):
                detected_doc_type = dtype
                break

        # ── Query rewriting ───────────────────────────────────────────────────
        year_ctx = f"GM meeting {' '.join(detected_years)}" if detected_years else "GM meeting"
        rewritten = f"{year_ctx} OEKO-TEX {question.rstrip('?')}"

        # ── Multi-query expansion ─────────────────────────────────────────────
        expanded = self._expand(question, detected_years, intent)

        return QueryContext(
            original_question=question,
            rewritten_query=rewritten,
            expanded_queries=expanded,
            detected_years=detected_years,
            detected_doc_type=detected_doc_type,
            intent=intent,
        )

    def _expand(
        self, question: str, years: List[str], intent: str
    ) -> List[str]:
        """Rule-based query variants to improve recall via multi-query retrieval."""
        q = question.strip().rstrip("?")
        variants: List[str] = [
            f"OEKO-TEX {q} discussion summary",
        ]
        if years:
            variants.append(f"{' and '.join(years)} GM meeting {q}")
        else:
            variants.append(f"GM meeting {q} details")
        return variants[:MULTI_QUERY_N]


# ── Hybrid Retriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Two-stage hybrid retrieval:
      1. Dense search   — ChromaDB cosine similarity (with metadata filter)
      2. Sparse search  — BM25Okapi over all indexed chunks (with same filter)
    Then merge both ranked lists via Reciprocal Rank Fusion (RRF).
    Supports multi-query: queries from expanded_queries are also searched and fused.
    """

    def __init__(self, vectorstore: Chroma) -> None:
        self.vectorstore = vectorstore
        self._all_docs: List[Document] = []
        self._bm25 = None
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        """Load all chunks from ChromaDB once and build a BM25 index in memory."""
        try:
            from rank_bm25 import BM25Okapi  # type: ignore
        except ImportError:
            logger.warning(
                "[Retriever] rank_bm25 not installed — BM25 disabled. "
                "Run: pip install rank-bm25"
            )
            return

        logger.info("[Retriever] Building BM25 index from ChromaDB …")
        col   = self.vectorstore._collection
        total = col.count()
        if total == 0:
            logger.warning("[Retriever] Collection is empty — BM25 index skipped.")
            return

        result    = col.get(limit=total, include=["documents", "metadatas"])
        texts     = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        self._all_docs = [
            Document(page_content=txt, metadata=meta)
            for txt, meta in zip(texts, metadatas)
            if txt and txt.strip()
        ]
        tokenized  = [d.page_content.lower().split() for d in self._all_docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"[Retriever] BM25 index ready ({len(self._all_docs)} chunks).")

    # ── Filter builder ────────────────────────────────────────────────────────

    @staticmethod
    def _build_chroma_filter(
        years: List[str], doc_type: Optional[str]
    ) -> Optional[dict]:
        conditions: List[dict] = []
        if years:
            conditions.append(
                {"year": years[0]}
                if len(years) == 1
                else {"$or": [{"year": y} for y in years]}
            )
        if doc_type:
            conditions.append({"doc_type": doc_type})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ── Dense search ──────────────────────────────────────────────────────────

    def _dense_search(
        self, query: str, chroma_filter: Optional[dict], k: int
    ) -> List[Document]:
        try:
            kwargs: dict = {"k": k}
            if chroma_filter:
                kwargs["filter"] = chroma_filter
            return self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs=kwargs
            ).invoke(query)
        except Exception as exc:
            logger.warning(f"[Dense] search error: {exc}")
            return []

    # ── BM25 search ───────────────────────────────────────────────────────────

    def _bm25_search(
        self,
        query: str,
        years: List[str],
        doc_type: Optional[str],
        k: int,
    ) -> List[Document]:
        if self._bm25 is None or not self._all_docs:
            return []
        try:
            scores = self._bm25.get_scores(query.lower().split())
            filtered: List[Tuple[float, Document]] = []
            for score, doc in zip(scores, self._all_docs):
                meta = doc.metadata
                if years and meta.get("year") not in years:
                    continue
                if doc_type and meta.get("doc_type") != doc_type:
                    continue
                filtered.append((score, doc))
            filtered.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in filtered[:k]]
        except Exception as exc:
            logger.warning(f"[BM25] search error: {exc}")
            return []

    # ── RRF merge ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(*ranked_lists: List[Document], k: int = RRF_K) -> List[Document]:
        """Reciprocal Rank Fusion: score = Σ 1/(k + rank) across all lists."""
        scores:  Dict[str, float]    = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = doc.page_content[:150]  # content fingerprint for dedup
                scores[key] += 1.0 / (k + rank)
                doc_map[key] = doc

        return [doc_map[key] for key in sorted(doc_map, key=lambda x: scores[x], reverse=True)]

    # ── Dedup helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe(docs: List[Document]) -> List[Document]:
        seen: set = set()
        out: List[Document] = []
        for d in docs:
            key = d.page_content[:150]
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    # ── Main retrieve ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_ctx: QueryContext,
        dense_k: int = DENSE_TOP_K,
        bm25_k: int  = BM25_TOP_K,
    ) -> List[Document]:
        chroma_filter = self._build_chroma_filter(
            query_ctx.detected_years, query_ctx.detected_doc_type
        )

        logger.info(
            f"[Retriever] '{query_ctx.rewritten_query[:70]}…' | "
            f"years={query_ctx.detected_years or 'all'} | "
            f"type={query_ctx.detected_doc_type or 'all'} | "
            f"intent={query_ctx.intent}"
        )

        # Execute all query variants (main + expansions)
        all_queries = [query_ctx.rewritten_query] + query_ctx.expanded_queries
        dense_pool:  List[Document] = []
        bm25_pool:   List[Document] = []

        for q in all_queries:
            dense_pool.extend(self._dense_search(q, chroma_filter, dense_k))
            bm25_pool.extend(
                self._bm25_search(q, query_ctx.detected_years, query_ctx.detected_doc_type, bm25_k)
            )

        dense_deduped = self._dedupe(dense_pool)
        bm25_deduped  = self._dedupe(bm25_pool)
        merged        = self._rrf_merge(dense_deduped, bm25_deduped)

        logger.info(
            f"[Retriever] Dense={len(dense_deduped)} | "
            f"BM25={len(bm25_deduped)} | "
            f"After RRF={len(merged)}"
        )
        return merged


# ── Cross-Encoder Reranker ─────────────────────────────────────────────────────

class Reranker:
    """
    Scores every (question, chunk) pair with a cross-encoder and returns
    only the top-n most relevant chunks.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, 6-layer).
    For higher accuracy: BAAI/bge-reranker-base (~2× slower, ~+8% MRR).
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model = None
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            logger.info(f"[Reranker] Loading cross-encoder: {model_name} …")
            self.model = CrossEncoder(model_name)
            logger.info("[Reranker] Ready.")
        except ImportError:
            logger.warning(
                "[Reranker] sentence-transformers not installed — reranking disabled. "
                "pip install sentence-transformers"
            )

    def rerank(
        self, question: str, docs: List[Document], top_n: int = RERANK_TOP_N
    ) -> List[Document]:
        if not docs:
            return docs
        if self.model is None:
            logger.warning("[Reranker] Skipped — returning top-n by RRF rank.")
            return docs[:top_n]

        pairs  = [(question, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        top = scored[:top_n]
        logger.info(
            "[Reranker] Scores: " + " | ".join(
                f"{s:.3f} [{d.metadata.get('year')}/{d.metadata.get('doc_type')}]"
                for s, d in top
            )
        )
        return [doc for _, doc in top]


# ── Context Builder ────────────────────────────────────────────────────────────

class ContextBuilder:
    """
    Formats reranked chunks into a numbered context block for the LLM.
    Includes all key metadata as a header so the model can cite sources correctly.
    """

    @staticmethod
    def build(docs: List[Document]) -> str:
        parts: List[str] = []
        for i, doc in enumerate(docs, 1):
            m       = doc.metadata
            ocr_tag = " [CHART-OCR]" if m.get("has_chart_ocr") == "True" else ""
            header  = (
                f"[Chunk {i}{ocr_tag}] "
                f"Year={m.get('year', '?')} | "
                f"Location={m.get('location', '?')} | "
                f"Type={m.get('doc_type', '?')} | "
                f"File={m.get('filename', '?')} | "
                f"Page={m.get('page', '?')}"
            )
            parts.append(f"{header}\n{doc.page_content.strip()}")
        return "\n\n---\n\n".join(parts)


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise analyst of OEKO-TEX General Manager (GM) Meeting documents \
covering 2022–2025 (Rome 2022, Budapest 2023, Alcoy 2024, Copenhagen 2025).

━━ STRICT ANTI-HALLUCINATION RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Answer EXCLUSIVELY from the provided context chunks.
   Do NOT use any external or background knowledge.
2. If the answer is not present in the context, respond with exactly:
   "This information is not found in the provided documents."
3. Do NOT infer, extrapolate, or generalise beyond what the documents state.
4. Do NOT mix up years — every factual claim must be tagged to its exact year.
5. If context partially answers the question, cover the supported part and
   explicitly note: "Information about [X] was not found in the documents."
6. Prefer saying "not found" over guessing.

━━ NUMERICAL / CHART DATA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Chunks tagged [CHART-OCR] contain machine-read chart/table data.
  OCR may have minor recognition errors — cross-check figures when possible.
- For rate/percentage/count/ranking questions, prioritise [CHART-OCR] chunks.
- Quote exact numbers and show any derived calculations step-by-step.
- Explicitly label which year each data point belongs to.

━━ CONFIDENCE CALIBRATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- high   : 3+ strongly relevant chunks provide consistent support.
- medium : 1–2 relevant chunks, or partial coverage only.
- low    : Single weak chunk, indirect evidence, or inferred from context.
  → When confidence = low, state the uncertainty explicitly in "answer".

━━ REQUIRED JSON OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Return ONLY a valid JSON object matching this schema exactly:
{schema}

Field guidelines:
  "answer"         — 2–4 paragraphs. Cite year + location for every claim.
  "key_points"     — 3–6 bullets. Each must start with an action verb.
  "relevant_years" — Only years explicitly mentioned in your answer.
  "document_types" — All distinct source doc types you drew from.
  "sources"        — All filenames cited (no paths, no duplicates).
  "confidence"     — Calibrated per rules above; note uncertainty in answer
                     if low.

━━ CONTEXT (pre-filtered + reranked for relevance) ━━━━━━━━━━━━━━━━━━━━━━━━━━
{context}
"""

HUMAN_PROMPT = "Question: {question}"


# ── Generator ──────────────────────────────────────────────────────────────────

class Generator:
    """Invokes Groq Llama with the formatted context and parses the structured response."""

    def __init__(self, llm: ChatGroq) -> None:
        self.llm    = llm
        self.parser = JsonOutputParser(pydantic_object=GMResponse)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human",  HUMAN_PROMPT),
        ])

    def generate(self, question: str, context: str) -> GMResponse:
        schema = self.parser.get_format_instructions()
        chain  = self.prompt | self.llm | self.parser
        result = chain.invoke({
            "schema":   schema,
            "context":  context,
            "question": question,
        })
        if isinstance(result, dict):
            return GMResponse(**result)
        return result  # type: ignore[return-value]


# ── Main Engine ────────────────────────────────────────────────────────────────

class GMRagEngine:
    """
    Production-grade RAG engine.

    Full pipeline:
      QueryAnalyzer
        → HybridRetriever  (Dense ChromaDB + BM25 + RRF + multi-query expansion)
        → CrossEncoder Reranker
        → ContextBuilder
        → Groq Llama Generator
        → GMResponse

    Public API:
      engine.query(question)
      engine.query_with_filters(question, year, doc_type)
      engine.query_year(question, year)
      engine.collection_stats()
    """

    def __init__(self) -> None:
        self._check_index()
        self._load_components()

    def _check_index(self) -> None:
        if not Path(CHROMA_DIR).exists():
            print(
                "\n[ERROR] ChromaDB index not found.\n"
                "        Run  python ingest.py --rebuild  first.\n"
            )
            sys.exit(1)

    def _load_components(self) -> None:
        print("[RAG] Loading embedding model …")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("[RAG] Connecting to ChromaDB …")
        self.vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

        print("[RAG] Building hybrid retriever (dense + BM25) …")
        self.query_analyzer = QueryAnalyzer()
        self.retriever      = HybridRetriever(self.vectorstore)

        print("[RAG] Loading cross-encoder reranker …")
        self.reranker    = Reranker()
        self.ctx_builder = ContextBuilder()

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
        llm = ChatGroq(model=GROQ_MODEL, groq_api_key=api_key, temperature=0.1)
        self.generator = Generator(llm)
        print("[RAG] Ready.\n")

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        year_filter:     Optional[str] = None,
        doc_type_filter: Optional[str] = None,
    ) -> GMResponse:
        """
        Full pipeline query.
        year_filter / doc_type_filter override the auto-detected values
        (useful when called from the Streamlit UI).
        """
        query_ctx = self.query_analyzer.analyze(question)

        # Explicit UI overrides win over auto-detected values
        if year_filter:
            query_ctx.detected_years    = [year_filter]
        if doc_type_filter:
            query_ctx.detected_doc_type = doc_type_filter

        # ── Retrieve ──────────────────────────────────────────────────────────
        candidates = self.retriever.retrieve(query_ctx)

        if not candidates:
            logger.warning("[Engine] No documents retrieved.")
            return GMResponse(
                answer="This information is not found in the provided documents.",
                key_points=["No matching content found in the knowledge base."],
                relevant_years=[],
                document_types=[],
                sources=[],
                confidence="low",
            )

        # ── Rerank ────────────────────────────────────────────────────────────
        top_docs = self.reranker.rerank(question, candidates, top_n=RERANK_TOP_N)

        logger.info(f"[Engine] {len(top_docs)} chunks → LLM")
        for i, d in enumerate(top_docs, 1):
            logger.info(
                f"  [{i}] year={d.metadata.get('year')} | "
                f"type={d.metadata.get('doc_type')} | "
                f"file={d.metadata.get('filename')} | "
                f"pg={d.metadata.get('page')} | "
                f"ocr={d.metadata.get('has_chart_ocr')}"
            )

        # ── Build context + generate ──────────────────────────────────────────
        context  = self.ctx_builder.build(top_docs)
        response = self.generator.generate(question, context)
        return response

    def query_with_filters(
        self,
        question: str,
        year:     Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> GMResponse:
        """Explicit-filter variant — matches the documented public API."""
        return self.query(question, year_filter=year, doc_type_filter=doc_type)

    def query_year(self, question: str, year: str) -> GMResponse:
        if year not in YEARS:
            raise ValueError(f"year must be one of {YEARS}")
        return self.query(question, year_filter=year)

    def collection_stats(self) -> dict:
        from collections import Counter
        col   = self.vectorstore._collection
        count = col.count()
        meta  = col.get(limit=count, include=["metadatas"])["metadatas"]
        years  = Counter(m.get("year",     "?") for m in meta)
        dtypes = Counter(m.get("doc_type", "?") for m in meta)
        ocr    = sum(1 for m in meta if m.get("has_chart_ocr") == "True")
        return {
            "total_chunks":       count,
            "chart_ocr_enriched": ocr,
            "by_year":            dict(sorted(years.items())),
            "by_doc_type":        dict(sorted(dtypes.items())),
        }


# ── Pretty Printer ─────────────────────────────────────────────────────────────

def print_response(resp: GMResponse, question: str) -> None:
    W = 70
    print("\n" + "=" * W)
    print(f"  QUERY: {question}")
    print("=" * W)
    print(f"\n📋 ANSWER\n{'-' * W}")
    print(resp.answer)
    print(f"\n✅ KEY POINTS\n{'-' * W}")
    for pt in resp.key_points:
        print(f"  • {pt}")
    print(f"\n📅 RELEVANT YEARS : {', '.join(resp.relevant_years) or 'N/A'}")
    print(f"📄 DOCUMENT TYPES : {', '.join(resp.document_types) or 'N/A'}")
    print(f"🔗 SOURCES\n{'-' * W}")
    for src in resp.sources:
        print(f"  - {src}")
    icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(resp.confidence, "⚪")
    print(f"\n{icon} CONFIDENCE : {resp.confidence.upper()}")
    print("=" * W + "\n")


# ── Interactive CLI ────────────────────────────────────────────────────────────

HELP_TEXT = """
Commands:
  <question>          — Ask anything about the GM meetings
  year <YYYY>         — Set a year filter (2022–2025)
  year clear          — Clear year filter
  stats               — Show collection statistics
  help                — Show this help
  exit / quit         — Exit
"""


def run_cli(engine: GMRagEngine) -> None:
    print("\n" + "=" * 70)
    print("  GM Meetings RAG v2 — Production | Hybrid + Reranked")
    print("  Data: OEKO-TEX GM Meetings 2022–2025")
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
            s = engine.collection_stats()
            print(f"\n  Total chunks     : {s['total_chunks']}")
            print(f"  Chart-OCR chunks : {s['chart_ocr_enriched']}")
            print(f"  By year          : {s['by_year']}")
            print(f"  By doc type      : {s['by_doc_type']}\n")
            continue
        if low.startswith("year "):
            arg = raw[5:].strip()
            if arg.lower() == "clear":
                year_filter = None
                print("  Year filter cleared.\n")
            elif arg in YEARS:
                year_filter = arg
                print(f"  Year filter → {year_filter}\n")
            else:
                print(f"  Unknown year '{arg}'. Choose from {YEARS} or 'clear'.\n")
            continue

        print("  Thinking …\n")
        try:
            resp = engine.query(raw, year_filter=year_filter)
            print_response(resp, raw)
        except Exception as exc:
            print(f"  [ERROR] {exc}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = GMRagEngine()
    run_cli(engine)
