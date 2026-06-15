#!/bin/bash
# Azure App Service Python Startup Script
# SIMPLIFIED: No pip install here — Oryx handles it during deployment.

echo "=== Azure App Service: Starting setup script ==="
echo "Date: $(date)"
echo "Python: $(python --version 2>&1)"

# ── 1. Persistent Hugging Face model cache (survives restarts) ───────────────
export HF_HOME="/home/site/hf_cache"
export TRANSFORMERS_CACHE="/home/site/hf_cache"
export SENTENCE_TRANSFORMERS_HOME="/home/site/hf_cache/sentence_transformers"
mkdir -p "$HF_HOME"

# ── 2. Resolve port (Azure sets $PORT dynamically; default 8000 for local) ──
APP_PORT="${PORT:-8000}"
echo "Configuring application to run on port: $APP_PORT"

# ── 3. Boot Uvicorn with exec to forward OS signals cleanly ─────────────────
echo "Launching FastAPI application via Uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port "$APP_PORT" --timeout-keep-alive 120
