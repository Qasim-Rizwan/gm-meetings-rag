# System SQLite3 override for ChromaDB (must be executed before any other import)
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Security, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("fastapi_app")

# Global engine reference
engine = None

def init_engine():
    """Initializes the RAG engine in a background thread to prevent blocking Uvicorn startup."""
    global engine
    logger.info("Initializing GM Meetings RAG Engine in background thread...")
    try:
        from rag import GMRagEngine
        engine = GMRagEngine()
        logger.info("RAG Engine successfully initialized.")
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize RAG Engine: {e}", exc_info=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Initializes the RAG Engine in the background so the server can accept connections immediately.
    """
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, init_engine)
    yield
    logger.info("Shutting down application...")

# Initialize FastAPI application
app = FastAPI(
    title="GM Meetings RAG Web API",
    description="FastAPI Web API wrapping the RAG Engine for Microsoft Copilot Studio and Power Automate integration.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict this in production to match your Power Automate / Copilot Studio tenants if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication Helpers ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
security_bearer = HTTPBearer(auto_error=False)

def verify_api_key(
    x_api_key: Optional[str] = Security(api_key_header),
    auth: Optional[HTTPAuthorizationCredentials] = Security(security_bearer)
):
    """
    Dependency to secure the API endpoint.
    If the API_ACCESS_TOKEN environment variable is set, incoming requests must provide
    a matching key via 'X-API-Key' header or standard 'Authorization: Bearer <TOKEN>' header.
    If API_ACCESS_TOKEN is not set, it logs a warning but allows the request (unsecured mode for local testing).
    """
    token = os.getenv("API_ACCESS_TOKEN")
    if not token:
        logger.warning("API_ACCESS_TOKEN environment variable is not set. API is running in UNSECURED mode.")
        return

    provided_key = None
    if x_api_key:
        provided_key = x_api_key
    elif auth and auth.credentials:
        provided_key = auth.credentials

    if not provided_key or provided_key != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Access Token (X-API-Key or Bearer Token required)"
        )

# --- Request / Response Schemas ---
class ChatRequest(BaseModel):
    message: str = Field(
        ..., 
        example="What were the main decisions in the 2024 Alcoy meeting?",
        description="The user question string to query the knowledge base."
    )

class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description="The RAG generated response answer grounded in the documents."
    )

# --- API Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Azure App Service readiness and liveness probe.
    IMPORTANT: Always returns 200 OK so Azure does NOT kill the container
    during cold start (model loading takes ~60s on B1).
    The 'engine' field indicates actual readiness.
    """
    if engine is None:
        return {"status": "starting", "engine": "loading"}
    return {"status": "healthy", "engine": "loaded"}

@app.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(verify_api_key)]
)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for MS Copilot Studio.
    Accepts: {"message": "..."}
    Returns: {"response": "..."}
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message field cannot be empty."
        )

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG Engine is not initialized."
        )

    try:
        logger.info(f"Processing query: {request.message[:100]}...")
        # Invoke the existing, untouched GMRagEngine query method
        result = engine.query(request.message)
        
        # Check if we got a fallback response or empty answer
        if not result or not result.answer:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Engine failed to generate a response."
            )
            
        return ChatResponse(response=result.answer)
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the request: {str(e)}"
        )
