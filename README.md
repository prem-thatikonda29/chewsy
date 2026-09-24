# Chewsy

Yuka-style barcode scanner for a Feature Engineering & MLOps mini-project.
Scan a real product barcode → the backend live-fetches it from the Open Food
Facts API → a trained model predicts its **NOVA processing level (1–4)** →
the UI shows the verdict plus a SHAP explanation of *why*.

## Stack

- **ML pipeline:** pandas, scikit-learn, custom transformers (`src/`), DVC
- **Experiment tracking:** MLflow (SQLite tracking + model registry)
- **API:** FastAPI (`app/`) — `POST /predict` is the single source of truth
- **Frontend:** Next.js (App Router, TypeScript) + camera barcode decode
- **Serving:** multi-stage Docker image, GitHub Actions → Docker Hub

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/fetch_training_set.py   # Stage 1: pull labeled training data
dvc repro                          # Stage 10: fetch → clean → features → train

uvicorn app.main:app --reload --port 8000
cd frontend && npm install && npm run dev   # :3000
```

## Docs

Full stage-by-stage plan: [`Chewsy_PRD_Roadmap.md`](./Chewsy_PRD_Roadmap.md)
