#!/usr/bin/env bash
#
# run.sh — launch the FIM benchmark web UI (backend + frontend) for local use.
#
# One command for development/demo: starts the FastAPI results server
# (daml_fim_benchmark.server.app:app) via uvicorn in the background, waits until
# /health responds, then runs the Vite dev server in the foreground. Stopping
# this script (Ctrl-C) also stops the backend it started (via the EXIT trap).
#
# Unlike scripts/restart_backend.sh — which only (re)starts the backend so a
# running UI picks up backend code changes — this brings up both halves together
# from a clean slate.
#
# Environment:
#   API_PORT                 Backend port (default: 8010). Vite proxies /api and
#                            /health to it (see vite.config.ts, VITE_API_TARGET).
#   DAML_FIM_BENCHMARK_ROOT  Root the backend resolves run data against (fim/logs,
#                            mlruns, ...). Defaults to the benchmark repo root
#                            (this script's parent directory).
#   FIM_FRONTEND_DIR         Frontend dir the backend serves static assets from.
#                            Defaults to this script's directory.
#
# Side effects: activates $ROOT_DIR/.venv if present; logs to
# frontend/.logs/backend.log and frontend/.logs/vite.log. Exits non-zero (after
# dumping the backend log tail) if the backend never becomes healthy.
#
# Usage:
#   ./run.sh
#   API_PORT=8020 ./run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/.logs"
BACKEND_LOG="$LOG_DIR/backend.log"
VITE_LOG="$LOG_DIR/vite.log"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

API_PORT="${API_PORT:-8010}"
export DAML_FIM_BENCHMARK_ROOT="${DAML_FIM_BENCHMARK_ROOT:-$ROOT_DIR}"
export FIM_FRONTEND_DIR="${FIM_FRONTEND_DIR:-$SCRIPT_DIR}"
mkdir -p "$LOG_DIR"

timestamp() {
  date "+%Y-%m-%dT%H:%M:%S%z"
}

echo "[$(timestamp)] starting backend on :$API_PORT" | tee -a "$BACKEND_LOG"
(
  cd "$ROOT_DIR"
  python -m uvicorn daml_fim_benchmark.server.app:app --host 127.0.0.1 --port "$API_PORT"
) >>"$BACKEND_LOG" 2>&1 &
API_PID=$!

cleanup() {
  kill "$API_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# Wait for backend health so Vite doesn't start against a dead proxy target.
for _ in {1..40}; do
  if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

if ! curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
  echo "Backend failed to start. Last backend log lines:"
  tail -n 120 "$BACKEND_LOG" || true
  exit 1
fi

cd "$SCRIPT_DIR"
echo "[$(timestamp)] starting vite dev server" | tee -a "$VITE_LOG"
npm run dev 2>&1 | tee -a "$VITE_LOG"
