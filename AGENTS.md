# Chewsy — Agent Instructions

## Project overview

Yuka-style barcode scanner for a Feature Engineering & MLOps course mini-project.
User enters a real product barcode → backend live-fetches it from the Open Food
Facts public API → a trained model predicts **NOVA processing level (1–4)** →
UI shows the verdict plus a SHAP explanation of *why*.

**Pitch framing:** Open Food Facts is crowdsourced; most products have no NOVA
label. This tool fills in a classification the community hasn't labeled yet —
the same job Yuka does commercially. Do not frame work as "I analyzed a dataset."

**Trained vs. live (memorize this — it's the judge's most likely question):**
- **Trained offline, once:** the model. `fetch_training_set.py` pulls ~6000
  community-labeled products; Stage 4 trains a classifier on them; Stage 5
  freezes it as `model.joblib`. It does not change until you retrain.
- **Live, every scan:** the input, not the model. The scanned barcode is
  almost certainly *not* one of the ~6000 training rows — the app fetches
  that product's raw nutrients/ingredients from the OFF API at that instant
  and feeds them into the frozen model.
- **The rule:** the model computes the verdict on **every scan, 100% of the
  time** — never a passthrough, never conditioned on "OFF is missing a
  label." It also never reads OFF's stored `nova_group` for that product,
  even when one exists.
- **Judge one-liner** ("is this just looking up OFF's data?"): *no — OFF
  tells you NOVA when they know it; Chewsy tells you NOVA when they don't,
  and it arrives at the same answer they did using only the raw
  ingredients, never peeking at their label.*

**NOVA classes:**
1 = unprocessed/minimally processed · 2 = processed culinary ingredients ·
3 = processed foods · 4 = ultra-processed foods

**Non-goals (state if asked, don't apologize):** not predicting Nutri-Score;
not a meal-logging/calorie app; not global coverage (training pulls are
class-balanced and capped ~1500/class; India is a demo story, never a
training filter — see Stage 1).

**Deadline:** submission + presentation 26 Sept 2026. ~2 days of build time as
of 24 Sept 2026. Treat PRD "MUST" stages as non-negotiable.

**Full stage detail:** `Chewsy_PRD_Roadmap.md` — re-read the relevant stage
section before implementing it. Each stage has a "Done when" condition; do not
advance until it is verifiably true.

## Hard rules (never violate)

1. **Headline metric is macro-F1, not accuracy** — NOVA classes are imbalanced;
   accuracy alone is misleading.
2. **Nutri-Score fields are never features** — `nutrition_grade_fr` (old
   bulk-export name) / `nutrition_grades` / `nutriscore_*` (live-API names)
   are a computed label sitting next to the target (leakage), not a
   legitimate predictor.
3. **Frontend must call FastAPI `/predict` over HTTP** — never reimplement
   scoring logic or ship/duplicate the model in the UI. One source of truth:
   the API owns prediction. Enable CORS (`CORSMiddleware`) for the Next.js
   origin or the browser will block fetches ("works with curl, fails in browser").
4. **Custom transformer classes live in `src/features.py`** — an importable
   module, never inline in a notebook. Avoids the pickling "can't get attribute
   FeatureCreator" bug.
5. **Open Food Facts API calls need a timeout (~5s) and clean 404/503 fallbacks**
   — never let a hung request kill the live demo. Applies to both the Stage 6
   barcode lookup and the Stage 1 training-set fetch.
6. **If behind schedule:** cut Stage 11 (cloud deploy) first, then trim Stage 12
   to a single `/metrics` endpoint with no live drift job. **Never cut Stages 4,
   5, 6, 9, or 10** — the rubric explicitly names them (MLflow, packaging,
   FastAPI, CI/CD, DVC pipeline).
7. **Verify live-API response fields against what `fetch_training_set.py`
   actually keeps** — the bulk TSV is retired (no `nova_group`); training data
   comes from `search.openfoodfacts.org` now. The script's
   `FORBIDDEN_LEAKAGE_FIELDS` assert is the leakage guard: never request or
   keep Nutri-Score fields (`nutrition_grades` / `nutriscore_*` — the API-era
   names for what the old export called `nutrition_grade_fr`).
8. **Stage-by-stage commits** with the PRD's message style (`chore:`, `feat:`,
   `ci:`) — real commit history is graded; no giant dump commits.
9. **Scale numeric nutrients (`StandardScaler`) before both PCA and the
   Logistic Regression baseline** — required, not optional. Unscaled PCA
   just finds "whichever column has the biggest raw numbers," not a real
   energy-density axis, and unscaled Logistic Regression is dominated by
   whichever nutrient has the largest units.
10. **Run a filter-method pass on numeric nutrients before modeling** —
    `VarianceThreshold` + a correlation check (e.g. `fat_100g` vs.
    `saturated_fat_100g`). Cheap, and the one selection family this
    project would otherwise skip entirely.
11. **The model computes every verdict at inference; OFF's stored NOVA label
    is never read** — the product API response includes `nova_group` for
    already-labeled products, but it must never enter the feature row or
    the response (assert it's absent in `off_client.py`/`main.py`, same
    pattern as `FORBIDDEN_LEAKAGE_FIELDS`). The output is always
    model-computed, never a passthrough. (Training labels *come from* OFF's
    community via the Stage 1 `nova_groups:<n>` query — that's fine;
    prediction never *reads* OFF's label.)

## Target repo structure

Create exactly this in Stage 0 (with `.gitkeep` where needed); don't improvise later:

```
chewsy/
├── .gitignore
├── README.md
├── dvc.yaml
├── dvc.lock                  (generated)
├── requirements.txt
├── data/
│   ├── raw/                  (DVC-tracked, not git-tracked)
│   └── processed/            (DVC-tracked)
├── src/
│   ├── fetch_training_set.py  (live API pull, replaces slice_openfoodfacts.py)
│   ├── clean.py
│   ├── features.py
│   ├── train.py
│   └── batch_predict.py      (offline CSV scoring)
├── models/
│   ├── model.joblib          (gitignored — DVC/artifact)
│   └── training_reference.csv
├── app/
│   ├── main.py               (FastAPI)
│   ├── off_client.py         (Open Food Facts client)
│   └── schemas.py            (Pydantic)
├── frontend/                  (Next.js — Stage 7)
│   ├── app/
│   │   ├── page.tsx           (scanner UI)
│   │   └── layout.tsx
│   ├── components/
│   │   └── BarcodeScanner.tsx (camera + html5-qrcode decode)
│   ├── package.json
│   └── next.config.js
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── Dockerfile                 (multi-stage: Node build + Python runtime)
├── entrypoint.sh
└── .github/workflows/ci.yml
```

`.gitignore` must exclude: `.venv/`, `data/raw/*` + `data/processed/*`
**but re-include `*.dvc` pointers** (`!/data/raw/*.dvc`,
`!/data/processed/*.dvc` — git tracks the pointers, DVC tracks the CSVs),
`en.openfoodfacts.org.products.tsv`, `models/*.joblib`, `__pycache__/`,
`frontend/node_modules/`, `frontend/.next/`.

`.gitignore` must **NOT** exclude (grading evidence — commit these):
`mlflow.db`, `mlruns/` (MLflow runs + registry + Stage 4 plot artifacts),
`dvc.yaml`, `dvc.lock`, any `*.dvc` pointer (data or models),
`models/training_reference.csv`.

## Stage workflow

Work the PRD stages in order (0→13). For each stage:

1. Re-read that stage's section in `Chewsy_PRD_Roadmap.md`.
2. Implement every checklist item.
3. Verify the "Done when" condition before moving on.
4. Commit with the stage's prescribed message.

Priority reminder: MUST = non-negotiable; SHOULD = cut if behind. Timeline blocks
and cut order are in PRD §3 and Hard rule 6.

## Key commands

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests
pytest

# Data pipeline
dvc init
dvc add data/raw/openfoodfacts_training_set.csv
dvc repro

# Experiment tracking
mlflow ui   # sqlite:///mlflow.db

# Local serving
uvicorn app.main:app --reload --port 8000
cd frontend && npm install && npm run dev   # Next.js on :3000

# Container (API :8000, frontend :3000)
docker build -t chewsy .
docker run -p 8000:8000 -p 3000:3000 chewsy
```

## Standing risks (PRD §6)

- `nova_group` isn't in the bulk TSV export by default (resolved: Stage 1
  now fetches training data from the live search API instead)
- India-only training is unviable (< 25 labeled rows/class) — India is a
  live-demo talking point only, never a training filter
- NOVA class balance is enforced by construction (one `nova_groups:<n>`
  query per class in `fetch_training_set.py`) — verify the printed balance
  after Stage 1.1 anyway; if any class comes back short of ~200 rows, check
  for paging issues or raise `TARGET_PER_CLASS` and re-run
- Live API dependency — Stage 1's training fetch and Stage 6's demo lookup
  both hit OFF live; timeouts are not optional on either
- Multi-runtime Docker image (Stage 8) — Python+Node in one container is
  fragile; test locally before Stage 9 CI, don't debug Dockerfile for the
  first time inside GitHub Actions
- Scope creep on Stage 12 — `/metrics` + one manual KS-test is enough

**Stack note:** frontend is Next.js (App Router, TypeScript) with camera
decode via `html5-qrcode` — not Streamlit. Frontend calls
`POST ${NEXT_PUBLIC_API_URL}/predict`; API port 8000, frontend port 3000.
