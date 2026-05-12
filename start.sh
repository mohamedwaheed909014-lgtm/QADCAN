#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  OpenSCAD Mechanical Copilot — start script
#  Run from the project root:  bash start.sh
# ══════════════════════════════════════════════════════════════════
set -e
cd "$(dirname "$0")"

# Load .env if it exists
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  echo "[start] Loaded .env"
fi

# Install / update Python dependencies
echo "[start] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# Move into backend directory where app.py and rag.py live
cd backend

echo "[start] Starting server at http://localhost:8000"
echo "[start] Press Ctrl+C to stop."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
