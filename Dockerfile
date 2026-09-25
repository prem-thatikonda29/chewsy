# Chewsy — multi-stage image: Node build + Python runtime (PRD Stage 8).
# Fresh clones / CI must run `dvc pull` BEFORE `docker build` — model.joblib
# is DVC-tracked (gitignored) and is baked into the image (PRD 8.3 decision).

# ---- Stage A: frontend build (PRD 8.1) ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
# Next.js inlines NEXT_PUBLIC_* at build time; the browser reaches the API on
# the host-mapped port (`docker run -p 8000:8000`), so that is the URL.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# `npm run build` produces .next/; prune dev deps — `next start` only needs
# the runtime dependencies (next, react, …).
RUN npm run build && npm prune --omit=dev

# ---- Stage B: runtime — FastAPI on :8000 + Node runtime for `next start` ----
# 3.12 not the PRD's suggested 3.11: shap==0.52.0 (pinned Stage 0.3) declares
# Requires-Python >=3.12 — 3.11 fails at pip install. Host is 3.12 too.
FROM python:3.12-slim
# Node 20 runtime (PRD 8.1): needed to run `next start`, not just build.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# requirements before app code → pip layer caches across app changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ app/
COPY src/ src/
# Bake the champion in (self-contained demo image — decision recorded in PRD 8.1).
COPY models/model.joblib models/model.joblib
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh
COPY --from=frontend-build /app/frontend ./frontend
EXPOSE 8000 3000
ENTRYPOINT ["./entrypoint.sh"]
