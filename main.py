# NOTE: pysqlite3 override is applied in init_engine() AFTER packages are installed,
# not here at module load time (pysqlite3-binary hasn't been pip-installed yet at this point).

import os
import sys
import logging
import asyncio
import subprocess
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
)
logger = logging.getLogger("fastapi_app")

# ── Global State ──────────────────────────────────────────────────────────────
engine = None
startup_status = "starting"   # starting | installing | loading | healthy | error
startup_error = ""            # human-readable error message if status == error

# ── Persistent pip download cache ─────────────────────────────────────────────
# /home/site is persistent Azure storage — cached .whl files survive restarts.
# This avoids re-downloading ~700MB of wheels every cold start.
PIP_CACHE_DIR = "/home/site/pip-cache"


def _run_pip(step_name: str, *args):
    """Run a pip install command, logging stdout/stderr on failure."""
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--cache-dir", PIP_CACHE_DIR,
        "--prefer-binary",
        "--quiet",
    ] + list(args)
    logger.info(f"[{step_name}] pip install {' '.join(str(a) for a in args[:3])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        err = result.stderr[-3000:] if result.stderr else result.stdout[-3000:]
        logger.error(f"[{step_name}] FAILED:\n{err}")
        raise RuntimeError(f"pip install failed at step '{step_name}': {err[-200:]}")
    logger.info(f"[{step_name}] done.")


def _already_installed() -> bool:
    """Fast check: are the heavy packages already importable in this session?"""
    try:
        import torch          # noqa: F401
        import chromadb       # noqa: F401
        import langchain      # noqa: F401
        return True
    except ImportError:
        return False


def ensure_packages():
    """
    Install all ML packages needed by the RAG engine.
    - Uses persistent pip DOWNLOAD cache (/home/site/pip-cache) so wheels are
      only downloaded once and reused on subsequent restarts.
    - Packages themselves install to the system site-packages of the current
      container session (ephemeral, but fast from cache after first boot).
    - Pinning opentelemetry prevents pip from backtracking through 30+ versions.
    """
    global startup_status, startup_error

    if _already_installed():
        logger.info("Heavy packages already importable — skipping installation.")
        return

    logger.info("=== Installing ML packages (first boot: ~7 min, restarts: ~3 min) ===")
    logger.info("Uvicorn is running — health checks return 200 throughout.")
    startup_status = "installing"
    os.makedirs(PIP_CACHE_DIR, exist_ok=True)

    # ── Step 1: CPU-only PyTorch ──────────────────────────────────────────────
    # Must use --extra-index-url so pip finds the +cpu variant on PyTorch CDN.
    _run_pip("torch-cpu",
             "torch==2.3.1+cpu",
             "--extra-index-url", "https://download.pytorch.org/whl/cpu")

    # ── Step 2: Core NLP stack ────────────────────────────────────────────────
    _run_pip("nlp-stack",
             "sentence-transformers>=2.7.0",
             "transformers>=4.40.0",
             "huggingface_hub>=0.23.0",
             "numpy<2")

    # ── Step 3: Pin opentelemetry BEFORE chromadb ─────────────────────────────
    # chromadb depends on opentelemetry-exporter-otlp-proto-grpc without a tight
    # upper bound, causing pip to backtrack through 20+ releases for hours.
    # Pre-pinning a known-good set stops the backtracking immediately.
    _run_pip("opentelemetry-pins",
             "opentelemetry-api==1.27.0",
             "opentelemetry-sdk==1.27.0",
             "opentelemetry-exporter-otlp-proto-grpc==1.27.0",
             "opentelemetry-exporter-otlp-proto-common==1.27.0",
             "opentelemetry-proto==1.27.0",
             "opentelemetry-semantic-conventions==0.48b0")

    # ── Step 4: Vector DB ─────────────────────────────────────────────────────
    _run_pip("chromadb",
             "chromadb>=1.0.0,<2",
             "pysqlite3-binary>=0.5.2",
             "protobuf>=3.20.3,<5.0.0")

    # ── Step 5: LangChain stack ───────────────────────────────────────────────
    _run_pip("langchain",
             "langchain>=1.2.0,<2",
             "langchain-community>=0.4.0",
             "langchain-core>=1.2.0,<2",
             "langchain-text-splitters>=1.1.0,<2",
             "langchain-huggingface>=1.2.0,<2",
             "langchain-chroma>=1.1.0,<2",
             "langchain-groq>=1.1.0,<2")

    # ── Step 6: Retrieval + LLM + config ─────────────────────────────────────
    _run_pip("extras",
             "rank-bm25>=0.2.2",
             "groq>=0.37.0",
             "python-dotenv>=1.0.0")

    logger.info("=== ALL PACKAGES INSTALLED SUCCESSFULLY ===")


def init_engine():
    """Load the RAG engine. Called only after ensure_packages() succeeds."""
    global engine, startup_status
    startup_status = "loading"
    logger.info("Loading GM Meetings RAG Engine...")

    # Apply pysqlite3 override NOW — pysqlite3-binary is installed at this point.
    # ChromaDB requires sqlite3 >= 3.35.0 but Azure's system sqlite3 is older.
    # This replaces the built-in sqlite3 module with the newer pysqlite3 version.
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        logger.info("pysqlite3 override applied successfully.")
    except ImportError:
        logger.warning("pysqlite3 not available — ChromaDB may fail with old sqlite3.")

    try:
        from rag import GMRagEngine
        engine = GMRagEngine()
        startup_status = "healthy"
        logger.info("=== RAG ENGINE READY — accepting queries ===")
    except Exception as e:
        logger.error(f"Engine init failed: {e}", exc_info=True)
        raise


def background_bootstrap():
    """Run in background thread: install packages then load engine."""
    global startup_status, startup_error
    try:
        ensure_packages()
        init_engine()
    except Exception as e:
        startup_status = "error"
        startup_error = str(e)
        logger.error(f"Bootstrap failed: {e}", exc_info=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Uvicorn starts immediately so Azure health probes get 200 from second one.
    All heavy work runs in a background thread.
    """
    logger.info("Uvicorn ready. Launching background bootstrap thread...")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, background_bootstrap)
    yield
    logger.info("Shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GM Meetings RAG API",
    description="FastAPI backend for MS Copilot Studio integration.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
security_bearer = HTTPBearer(auto_error=False)


def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    auth: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
):
    token = os.getenv("API_ACCESS_TOKEN")
    if not token:
        logger.warning("API_ACCESS_TOKEN not set — running UNSECURED.")
        return
    provided = x_api_key or (auth.credentials if auth else None)
    if not provided or provided != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token."
        )


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., example="What were the key decisions in the 2024 GM meeting?")


class ChatResponse(BaseModel):
    response: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    ALWAYS returns HTTP 200 — Azure uses this to decide if the container is
    alive. The 'status' field tells you what the server is currently doing.
    """
    return {
        "status": startup_status,
        "engine": "loaded" if engine is not None else "not_loaded",
        "error": startup_error if startup_error else None,
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)],
)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for MS Copilot Studio."""
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty."
        )
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server not ready (status: {startup_status}). Try again in a moment.",
        )
    try:
        logger.info(f"Query: {request.message[:80]}...")
        result = engine.query(request.message)
        if not result or not result.answer:
            raise HTTPException(status_code=500, detail="Engine returned empty response.")
        return ChatResponse(response=result.answer)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
