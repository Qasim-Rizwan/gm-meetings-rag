#!/bin/bash
# Azure App Service Python Startup Script

# Log output
echo "Azure App Service: Starting setup script..."

# 1. Install FastAPI-specific dependencies (if Oryx build used the default requirements.txt)
if [ -f "requirements-api.txt" ]; then
    echo "Installing API-specific dependencies from requirements-api.txt..."
    pip install --no-cache-dir -r requirements-api.txt
else
    echo "requirements-api.txt not found. Relying on default environment."
fi

# 2. Resolve port (Azure App Service sets $PORT dynamically; default to 8000 for local)
APP_PORT="${PORT:-8000}"
echo "Configuring application to run on port: $APP_PORT"

# 3. Boot Uvicorn with exec to forward OS signals cleanly
echo "Launching FastAPI application via Uvicorn..."
exec uvicorn main:app --host 0.0.0.0 --port "$APP_PORT"
