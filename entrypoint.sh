#!/usr/bin/env bash
# PRD 8.2 + adapted 11: serve API + frontend by default (local demo image);
# APP_MODE=api serves ONLY uvicorn on ${PORT:-8000} — Render free exposes a
# single HTTP port per web service (https://render.com/docs), and the UI
# lives on Vercel there. Both modes tear down together on SIGTERM (this
# script is PID 1; the trap stops the children).
set -euo pipefail

MODE="${APP_MODE:-both}"
API_PORT="${PORT:-8000}"

uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!
FE_PID=""

shutdown() {
  kill -TERM "$API_PID" 2>/dev/null || true
  if [ -n "$FE_PID" ]; then
    kill -TERM "$FE_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit 0
}
trap shutdown TERM INT

echo "[entrypoint] mode=$MODE — waiting for API /health on :$API_PORT ..."
healthy=0
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  echo "[entrypoint] API failed to become healthy within 60s" >&2
  exit 1
fi
echo "[entrypoint] API healthy"

if [ "$MODE" = "api" ]; then
  # Render: single-port API-only; frontend is on Vercel.
  echo "[entrypoint] APP_MODE=api — frontend not started, supervising uvicorn"
  wait "$API_PID" || true
  echo "[entrypoint] uvicorn exited — shutting down"
  shutdown
fi

echo "[entrypoint] starting frontend on :3000"
cd /app/frontend
node node_modules/next/dist/bin/next start -H 0.0.0.0 -p 3000 &
FE_PID=$!

# Supervise: if either child exits on its own, tear the other one down too.
wait -n "$API_PID" "$FE_PID" || true
echo "[entrypoint] a child exited — shutting down"
shutdown
