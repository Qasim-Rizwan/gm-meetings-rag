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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Debug Lifespan Context Manager.
    Engine loading is disabled to verify server deployment.
    """
    logger.info("Application starting up in debug mode (Engine loading disabled)...")
    yield
    logger.info("Shutting down application...")

# Initialize FastAPI application
app = FastAPI(
    title="GM Meetings RAG Web API (Debug)",
    description="Debug version to verify server deployment on Azure.",
    version="1.0.0-debug",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request / Response Schemas ---
class ChatRequest(BaseModel):
    message: str = Field(..., example="Hello", description="Test message")

class ChatResponse(BaseModel):
    response: str

# --- API Endpoints ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check."""
    return {"status": "healthy", "mode": "debug"}

@app.post("/api/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """Simple echo endpoint to verify connectivity."""
    logger.info(f"Received test message: {request.message}")
    return ChatResponse(response=f"Echo: {request.message}")
