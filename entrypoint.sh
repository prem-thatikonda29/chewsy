#!/usr/bin/env bash
# PRD 8.2: start uvicorn on :8000, poll /health until it responds, then start
# `next start` on :3000. Both stop together on SIGTERM (Docker stops the
# container with SIGTERM → this script is PID 1 and the trap tears both down).
set -euo pipefail

uvicorn app.main:app --host 0.0.0.0 --port 8000 &
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

echo "[entrypoint] waiting for API /health ..."
healthy=0
for _ in $(seq 1 60); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  echo "[entrypoint] API failed to become healthy within 60s" >&2
  exit 1
fi
echo "[entrypoint] API healthy — starting frontend on :3000"

cd /app/frontend
node node_modules/next/dist/bin/next start -H 0.0.0.0 -p 3000 &
FE_PID=$!

# Supervise: if either child exits on its own, tear the other one down too.
wait -n "$API_PID" "$FE_PID" || true
echo "[entrypoint] a child exited — shutting down"
shutdown
