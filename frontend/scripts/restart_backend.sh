#!/usr/bin/env bash
#
# restart_backend.sh — (re)start the FIM benchmark results server for the web UI.
#
# Use during development after editing backend/server code: the server runs
# without auto-reload, so restarting it is how the running instance — and thus
# the web UI — picks up your latest backend changes.
#
# Stops any backend already listening on $API_PORT, then launches
# `daml_fim_benchmark.server.app:app` (FastAPI) via uvicorn in the background and
# waits until /health responds. Idempotent — safe to run repeatedly; it always
# kills the previous instance first. If the first start fails it force-stops and
# retries once, exiting non-zero if the backend never becomes healthy.
#
# Environment:
#   API_PORT                 Backend port (default: 8010).
#   DAML_FIM_BENCHMARK_ROOT  Root the server resolves run data against
#                            (mlruns/, logs, datasets). Defaults to the repo root.
#   FIM_FRONTEND_DIR         Frontend dir the server serves static assets from.
#                            Defaults to this script's parent frontend/ directory.
#
# Side effects: activates $ROOT_DIR/.venv if present; logs to
# frontend/.logs/backend.log and writes the PID to frontend/.logs/backend.pid.
#
# Usage:
#   ./scripts/restart_backend.sh
#   API_PORT=8020 ./scripts/restart_backend.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT_DIR="$(cd "$REACT_DIR/.." && pwd)"
LOG_DIR="$REACT_DIR/.logs"
BACKEND_LOG="$LOG_DIR/backend.log"
PID_FILE="$LOG_DIR/backend.pid"
API_PORT="${API_PORT:-8010}"
APP_PATTERN="uvicorn daml_fim_benchmark.server.app:app"
export DAML_FIM_BENCHMARK_ROOT="${DAML_FIM_BENCHMARK_ROOT:-$ROOT_DIR}"
export FIM_FRONTEND_DIR="${FIM_FRONTEND_DIR:-$REACT_DIR}"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

mkdir -p "$LOG_DIR"

if [[ -f "$ROOT_DIR/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.venv/bin/activate"
fi

stop_existing_backend() {
  local self_pid="$$"

  if command -v pkill >/dev/null 2>&1; then
    pkill -f "$APP_PATTERN" >/dev/null 2>&1 || true
  fi

  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  fi

  if command -v lsof >/dev/null 2>&1; then
    while IFS= read -r pid; do
      if [[ -n "$pid" && "$pid" != "$self_pid" ]]; then
        kill "$pid" >/dev/null 2>&1 || true
      fi
    done < <(lsof -tiTCP:"$API_PORT" -sTCP:LISTEN 2>/dev/null || true)
  fi

  for _ in {1..30}; do
    if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
      if command -v lsof >/dev/null 2>&1; then
        while IFS= read -r pid; do
          if [[ -n "$pid" && "$pid" != "$self_pid" ]]; then
            kill -9 "$pid" >/dev/null 2>&1 || true
          fi
        done < <(lsof -tiTCP:"$API_PORT" -sTCP:LISTEN 2>/dev/null || true)
      fi
      sleep 0.15
      continue
    fi
    break
  done
}

start_backend_once() {
  echo "[$(ts)] restarting backend on :$API_PORT" >>"$BACKEND_LOG"
  (
    cd "$ROOT_DIR"
    nohup python -m uvicorn daml_fim_benchmark.server.app:app --host 127.0.0.1 --port "$API_PORT" >>"$BACKEND_LOG" 2>&1 &
    echo $! > "$PID_FILE"
  )

  for _ in {1..40}; do
    if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

stop_existing_backend

if start_backend_once; then
  exit 0
fi

echo "[$(ts)] first start attempt failed, retrying after forced stop" >>"$BACKEND_LOG"
stop_existing_backend
if start_backend_once; then
  exit 0
fi

echo "[$(ts)] warning: backend did not become healthy after restart attempt" >>"$BACKEND_LOG"
exit 1
