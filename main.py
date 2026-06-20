# System SQLite3 override for ChromaDB (must be executed before any other import)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

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

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("fastapi_app")

# ── Global State ─────────────────────────────────────────────────────────────
engine = None
startup_status = "starting"   # starting | installing | loading | healthy | error

# ── Persistent Package Cache ──────────────────────────────────────────────────
# /home/site is a persistent Azure volume — packages survive container restarts.
PACKAGES_DIR = "/home/site/packages"
# Change this version string whenever you change the package list below.
# This forces a reinstall when dependencies change.
INSTALL_FLAG = "/home/site/packages/.installed_v5"


def ensure_packages():
    """
    Install all heavy ML packages to persistent /home/site/packages/ storage.
    On first boot this takes ~5 minutes. On subsequent restarts it's instant.
    """
    global startup_status

    # If already installed (from a previous container run), just add to path
    if os.path.exists(INSTALL_FLAG):
        logger.info("Packages already cached — adding /home/site/packages to sys.path")
        if PACKAGES_DIR not in sys.path:
            sys.path.insert(0, PACKAGES_DIR)
        return

    logger.info("=== FIRST BOOT: Installing packages to /home/site/packages/ ===")
    logger.info("This takes ~5 minutes. Health checks will return 200 throughout.")
    startup_status = "installing"
    os.makedirs(PACKAGES_DIR, exist_ok=True)

    def run_pip(*args):
        cmd = [sys.executable, "-m", "pip", "install",
               "--target", PACKAGES_DIR,
               "--no-cache-dir", "--quiet", "--prefer-binary"] + list(args)
        logger.info(f"pip install: {' '.join(args[:3])}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"pip install failed:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"pip install failed for: {args}")
        logger.info(f"Installed: {args[0]}")

    try:
        # ── Step 1: CPU-only PyTorch (avoids 2GB CUDA bloat) ─────────────────
        run_pip(
            "torch==2.3.1+cpu",
            "--extra-index-url", "https://download.pytorch.org/whl/cpu"
        )

        # ── Step 2: Core ML / NLP stack ──────────────────────────────────────
        run_pip(
            "sentence-transformers>=2.7.0",
            "transformers>=4.40.0",
            "huggingface_hub>=0.23.0",
            "numpy<2",
        )

        # ── Step 3: Vector DB (pin opentelemetry to avoid pip backtracking) ──
        run_pip(
            "opentelemetry-api==1.27.0",
            "opentelemetry-sdk==1.27.0",
            "opentelemetry-exporter-otlp-proto-grpc==1.27.0",
            "opentelemetry-exporter-otlp-proto-common==1.27.0",
            "opentelemetry-proto==1.27.0",
            "opentelemetry-semantic-conventions==0.48b0",
        )
        run_pip(
            "chromadb>=1.0.0,<2",
            "pysqlite3-binary>=0.5.2",
            "protobuf>=3.20.3,<4.0.0",
        )

        # ── Step 4: LangChain stack ───────────────────────────────────────────
        run_pip(
            "langchain>=1.2.0,<2",
            "langchain-community>=0.4.0",
            "langchain-core>=1.2.0,<2",
            "langchain-text-splitters>=1.1.0,<2",
            "langchain-huggingface>=1.2.0,<2",
            "langchain-chroma>=1.1.0,<2",
            "langchain-groq>=1.1.0,<2",
        )

        # ── Step 5: Retrieval + LLM ───────────────────────────────────────────
        run_pip(
            "rank-bm25>=0.2.2",
            "groq>=0.37.0",
            "python-dotenv>=1.0.0",
        )

        # Mark install as done
        with open(INSTALL_FLAG, "w") as f:
            f.write("ok")

        # Add to sys.path for this session
        if PACKAGES_DIR not in sys.path:
            sys.path.insert(0, PACKAGES_DIR)

        logger.info("=== ALL PACKAGES INSTALLED SUCCESSFULLY ===")

    except Exception as e:
        logger.error(f"CRITICAL: Package installation failed: {e}")
        startup_status = "error"
        raise


def init_engine():
    """Load the RAG engine after packages are confirmed installed."""
    global engine, startup_status
    startup_status = "loading"
    logger.info("Loading GM Meetings RAG Engine...")
    try:
        from rag import GMRagEngine
        engine = GMRagEngine()
        startup_status = "healthy"
        logger.info("=== RAG ENGINE READY ===")
    except Exception as e:
        logger.error(f"CRITICAL: Engine init failed: {e}", exc_info=True)
        startup_status = "error"


def background_bootstrap():
    """Run in a ThreadPoolExecutor: install packages, then load engine."""
    try:
        ensure_packages()
        init_engine()
    except Exception as e:
        logger.error(f"Bootstrap failed: {e}", exc_info=True)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Starts uvicorn immediately so health checks pass from the first second.
    Heavy work (package install + engine init) runs in a background thread.
    """
    logger.info("Uvicorn started. Launching background bootstrap...")
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, background_bootstrap)
    yield
    logger.info("Shutting down.")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GM Meetings RAG API",
    description="FastAPI backend for Microsoft Copilot Studio integration.",
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
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API token.")


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., example="What were the key decisions in the 2024 GM meeting?")


class ChatResponse(BaseModel):
    response: str


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Always returns 200 OK — Azure uses this to decide if container is alive.
    The 'status' field shows what the server is currently doing.
    """
    return {
        "status": startup_status,
        "engine": "loaded" if engine is not None else "not_loaded",
    }


@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK,
          dependencies=[Depends(verify_api_key)])
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint for MS Copilot Studio."""
    if not request.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Message cannot be empty.")
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server is not ready yet (status: {startup_status}). Try again in a minute.",
        )
    try:
        logger.info(f"Query: {request.message[:80]}...")
        result = engine.query(request.message)
        if not result or not result.answer:
            raise HTTPException(status_code=500, detail="Engine returned empty response.")
        return ChatResponse(response=result.answer)
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
