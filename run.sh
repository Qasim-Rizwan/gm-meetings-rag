#!/bin/bash
# Azure App Service Python Startup Script
# Oryx build is DISABLED (SCM_DO_BUILD_DURING_DEPLOYMENT=false)
# This script installs packages and starts the server directly.

set -e

echo "=== Azure App Service: Starting setup script ==="
echo "Date: $(date)"
echo "Python: $(python --version 2>&1)"

# ── 1. Persistent Hugging Face model cache (survives restarts) ───────────────
export HF_HOME="/home/site/hf_cache"
export TRANSFORMERS_CACHE="/home/site/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="/home/site/hf_cache/sentence_transformers"
mkdir -p "$HF_HOME"

# ── 2. Install Python dependencies directly (no Oryx cache issues) ──────────
echo "Installing Python dependencies..."
pip install --quiet --no-cache-dir \
    "fastapi>=0.100.0,<1.0.0" \
    "uvicorn[standard]>=0.22.0,<1.0.0" \
    "pydantic>=2.0.0,<3"
echo "Dependencies installed successfully."

# ── 3. Resolve port (Azure sets $PORT dynamically; default 8000 for local) ──
APP_PORT="${PORT:-8000}"
echo "Configuring application to run on port: $APP_PORT"

# ── 4. Boot Uvicorn with exec to forward OS signals cleanly ─────────────────
echo "Launching FastAPI application via Uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port "$APP_PORT" --timeout-keep-alive 120
