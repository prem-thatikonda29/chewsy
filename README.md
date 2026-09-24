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

# restore the DVC-tracked data + model (see next section)
export AWS_ACCESS_KEY_ID='HFAK…'
export AWS_SECRET_ACCESS_KEY='…'
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
export AWS_RESPONSE_CHECKSUM_VALIDATION=when_required
dvc pull

pytest
uvicorn app.main:app --reload --port 8000
cd frontend && npm install && npm run dev   # :3000
```

`dvc repro` (end-to-end retraining) arrives with Stage 10 — until then the
training data is restored by `dvc pull` and the packaged model by
`python src/package.py` (both are already run for you in the shipped repo).

## Data & model storage (DVC → HuggingFace)

Large artifacts are versioned with DVC and pushed to a private HuggingFace
Storage Bucket behind the `s3.hf.co` gateway; git holds only the tiny
pointer files (`*.dvc`, `.dvc/config`) plus the committed drift baseline:

| Artifact | Where |
|---|---|
| `data/raw/*.csv`, `data/processed/*.csv` (~15 MB) | DVC → `s3://chewsy-dvc/dvc-store` |
| `models/model.joblib` (17 MB, the packaged champion) | DVC → same remote |
| `models/training_reference.csv` | **git** (drift baseline, small) |
| `mlflow.db`, `mlruns/` (metrics, plots, registry) | **git** (grading evidence) |

`dvc pull` needs S3 credentials for the bucket (HuggingFace token →
*Generate S3 credentials*), exported as `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` as in the Quickstart. They are **never** committed:
on a machine that already has them, store them once with
`dvc remote modify --local hf-bucket access_key_id …` / `secret_access_key …`
(git-ignored `.dvc/config.local`). The two checksum env vars work around a
known botocore ↔ `s3.hf.co` incompatibility (trailing CRC32 checksums).

## Docs

Full stage-by-stage plan: [`Chewsy_PRD_Roadmap.md`](./Chewsy_PRD_Roadmap.md)
