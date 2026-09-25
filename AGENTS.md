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
- **Trained offline, once:** the model. `fetch_training_set.py` pulls
  20,000 community-labeled products (5,000/NOVA-class; 19,998 after
  Stage 2 cleaning); Stage 4 trains a classifier on them; Stage 5
  freezes it as `model.joblib`. It does not change until you retrain.
- **Live, every scan:** the input, not the model. The scanned barcode is
  almost certainly *not* one of the 20,000 training rows — the app fetches
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
class-balanced and capped 5,000/class — API hard-ceiling is 10,000/class;
India is a demo story, never a
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
├── params.yaml               (Stage 10: pipeline params the scripts read)
├── requirements.txt
├── conftest.py                (pytest: puts repo root on sys.path)
├── data/
│   ├── raw/                  (DVC-tracked, not git-tracked)
│   └── processed/            (DVC-tracked)
├── src/
│   ├── fetch_training_set.py  (live API pull, replaces slice_openfoodfacts.py)
│   ├── clean.py
│   ├── features.py
│   ├── train.py
│   ├── params.py                 (Stage 10: params.yaml reader)
│   ├── package.py               (registry champion → model.joblib, Stage 5)
│   └── batch_predict.py      (offline CSV scoring)
├── models/
│   ├── model.joblib          (gitignored — DVC pipeline out via dvc.yaml train)
│   └── training_reference.csv  (git-committed — Stage 12 drift baseline)
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
│   ├── test_train.py
│   ├── test_package.py
│   └── test_api.py
├── docs/
│   └── data_quality_report.md (LOCAL ONLY — in .git/info/exclude,
│                               never git add/push; append evidence)
├── Dockerfile                 (multi-stage: Node build + Python runtime)
├── entrypoint.sh
└── .github/workflows/ci.yml
```

`.gitignore` must exclude: `.venv/`, `data/raw/*.csv` +
`data/processed/*.csv` (ignore only the data files, **never** the
directories themselves — a `data/raw/*` pattern makes DVC's
`collect_files` prune `data/raw/` via `scm.is_ignored("data/raw/")`
and silently skip the `.dvc` pointer, so `dvc status` reports the
out as deleted), `en.openfoodfacts.org.products.tsv`,
`models/*.joblib`, `__pycache__/`, `frontend/node_modules/`,
`frontend/.next/`.

`.gitignore` must **NOT** exclude (grading evidence — commit these):
`mlflow.db`, `mlruns/` (MLflow runs + registry + Stage 4 plot artifacts),
`dvc.yaml`, `dvc.lock`, any `*.dvc` pointer (data or models),
`models/training_reference.csv`.

## Stage workflow

Work the PRD stages in order (0→13). For each stage:

0. **Propose before you build.** Present the stage plan first — the PRD
   checklist plus any evidence-backed enhancements or deviations you
   intend — and get approval before writing code. The PRD is a baseline
   to improve on, not a script: Stages 2 and 3 both deviated after
   measuring the data (sqrt vs log1p for energy skew, tag-level category
   encoding, PCA kept out of the pipeline). Every deviation must be
   backed by a printed measurement and recorded in the PRD checkbox notes
   and `docs/data_quality_report.md`.
1. Re-read that stage's section in `Chewsy_PRD_Roadmap.md`.
2. Implement every checklist item.
3. Verify the "Done when" condition before moving on.
4. Commit with the stage's prescribed message.

Priority reminder: MUST = non-negotiable; SHOULD = cut if behind. Timeline blocks
and cut order are in PRD §3 and Hard rule 6.

## As-built interfaces (Stages 0–6, as of 25 Sep 2026)

- **Data:** `data/processed/openfoodfacts_clean.csv` — 19,998 × 22,
  DVC-tracked; nutrient NaNs are **retained on purpose** (45,453 nulls)
  and imputed at train time by `GroupMedianImputer` inside the Stage 3
  pipeline (moved out of Stage 2 to fix pre-split leakage — see PRD 2.6
  notes / `docs/data_quality_report.md` §7), classes {1:5000, 2:4998,
  3:5000, 4:5000}. The bulk TSV was deleted from disk — nothing references it.
- **Feature pipeline:** `src/features.py::build_feature_pipeline(
  include_text=True, include_categorical=True, n_svd_components=25, ...)`
  → 44 columns (17 numeric + brand freq + tag freq + 25 SVD). Head steps
  in order: `create` → `impute` (`GroupMedianImputer`, train-fit) →
  `skew` → `ColumnTransformer`. Input must
  be a **DataFrame** with the cleaned column names. Stage 4 run mapping
  (as built): Run 1 = both flags False; Run 2 = text only; Runs 3–6 =
  defaults.
- **Inference-robust inside** (`GroupMedianImputer` group→global median,
  median imputer, NaN text → `""`, unseen brand/tag → mean training
  frequency, negatives → impute, `*_was_missing` indicators computed
  when absent from the input row) but it does NOT re-run Stage 2 field
  normalization — Stage 6 must normalize the raw OFF response (brand
  parse, mass cap, HTML unescape) first; see the Stage 6 handoff note in
  the PRD.
- **PCA is analysis-only** (SHAP must explain real nutrients); SVD lives
  in the text branch. `get_feature_names_out()` gives SHAP labels.
- **Trained model (Stage 4, as built):** `src/train.py` —
  `python src/train.py --runs 1 2 3 4 5 6`, then
  `python src/train.py --register`. Six runs in MLflow experiment
  `chewsy-nova` (tracking `sqlite:///mlflow.db`, committed). Macro-F1
  **at `AUGMENT_TEXT_DROPOUT = 0.20` (Gen 3, current):**
  run1 LR nutrients-only 0.7936 → run2 LR +text 0.8749 (text worth
  **+0.09**), run3 RandomForest 0.9394, run4 HistGB 0.9432,
  **run5 XGBoost 0.9465 = winner → registry `chewsy-nova` v4 alias
  `champion`** (sparse full-stub F1 0.7726 on the same eval), run6 KNN
  0.8509. **Gen-3 sparse-view augmentation (approved):**
  `augment_text_dropout` appends full-stub TRAIN twins (blank text + blank
  `categories_tags` + counts NaN + `text_was_missing=1`) of a stratified
  20% of train rows, post-split only (eval-hash tripwire); rate swept as
  tracked param `text_dropout_frac` via
  `python src/train.py --sweep 0 0.1 0.2 0.3` (0 → 0.4135, 0.1 → 0.7540,
  **0.2 chosen** → 0.7726, 0.3 → 0.7754; baseline 0.9489 → 0.9465).
  **Every run logs baseline + sparse metrics pair**
  (`f1_macro`/`sparse_f1_macro`/`sparse_f1_class_3`/`sparse_f1_class_4`).
  Root cause: no-text training rows = {1:179, 2:3759, 3:96, 4:0}.
  Split: stratified 80/20 `random_state=42`, split before any fit —
  refit is deterministic, so `build_full_pipeline(5)` on the same split
  reproduces the champion's weights exactly. `X` excludes `code`
  (barcode identifier, not a feature; uint64 breaks signature inference).
- **`NovaLabelAdapter`** (in `src/features.py`, Hard rule 4): xgboost ≥ 2
  rejects class ids outside `[0..n-1]`, so the wrapper encodes
  {1,2,3,4}→{0,1,2,3} in `fit` and decodes predictions back — every
  artifact still predicts NOVA **1..4**, and `predict_proba` is exposed
  (Stage 6's confidence comes from it). Never defines custom classes in
  `src/train.py` — executed as `python src/train.py` they pickle as
  `__main__.*`, unimportable in a fresh process (that bug bit us once).
- **Champion-only MLflow model logging is deliberate:** comparison runs
  store params/metrics/plots only (PRD 4.2–4.4 don't require model
  binaries); only the champion is logged as an artifact (pickle format,
  17 MB) because MLflow's default skops serialization inflated identical
  pipelines to 78–155 MB each → 673 MB of `mlruns/`, over GitHub's
  100 MB/file limit. Don't "fix" this by logging every run's model.
  `mlruns/` + `mlflow.db` as committed = 19 MB and contain everything
  grading needs (metrics, confusion matrices, SHAP plots, registry).
- **Packaged artifact (Stage 5, as built):** `python src/package.py` →
  `models/model.joblib` (17.7 MB, Gen-3 champion) +
  `models/training_reference.csv`
  (19,998 × 20, git-committed, NaN-preserving, no `code`/`nova_group` —
  the live input space for Stage 12 KS drift checks). Source of the
  dump: `mlflow.sklearn.load_model("models:/chewsy-nova@champion")`, so
  joblib == registry == committed `mlruns/` pickle (enforced by
  `tests/test_package.py`, which asserts registry-vs-joblib prediction
  equality inside a fresh subprocess — the PRD 5.2 pickle guard).
  Re-run after any retrain: `python src/package.py && dvc add
  models/model.joblib && dvc push`.
- **DVC remote (as built, Stage 5):** default remote `hf-bucket` in
  `.dvc/config` (committed, secret-free):
  `s3://chewsy-dvc/dvc-store` via `endpointurl https://s3.hf.co/prem2903`,
  `region us-east-1` — a **private** HuggingFace Storage Bucket. S3
  creds (HFAK pair from an HF token) live only in git-ignored
  `.dvc/config.local` (machine) or CI env (Stage 9 secrets). Push/pull
  must export `AWS_REQUEST_CHECKSUM_CALCULATION=when_required` and
  `AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` (botocore trailing
  CRC32 breaks the gateway). Fresh-clone flow: creds → `dvc pull` →
  `pytest`. **Stage 10:** `dvc remove` the static `*.dvc` pointers
  (data + model) before pipeline `outs` claim the same files.
  `requirements.txt` pins `dvc[s3]==3.67.1`.
- **Serving (Stage 6, as built):** `uvicorn app.main:app --port 8000` —
  lifespan loads `models/model.joblib` once and caches a
  `shap.TreeExplainer` + the 44 feature names. `POST /predict` =
  `off_client.fetch_product` (5s timeout, typed `ProductNotFound`/
  `OffAPIUnavailable` → 404/503) → `off_client.extract_feature_input`
  (mirrors `flatten_hit()`: **`energy-kcal_100g`, never the kJ
  `energy_100g`**) → `src/clean.py::normalize_live_row` (runs
  `clean(df, verbose=False)` on the one row — single cleaning code path;
  `clean()`'s only change for Stage 6 is the new `verbose` param) →
  `app.main.score_row` (one scoring path shared with batch) → top-5 SHAP
  for the predicted class (NOVA p → adapter column p−1). Also
  `GET /health`, `GET /metrics` (in-memory counters), CORS origins from
  env `FRONTEND_ORIGINS` (default `http://localhost:3000`). **Three Hard
   rule 11 guards:** `off_client.strip_forbidden` → `normalize_live_row`
   asserts → `score_row` asserts. Response fields `predicted_nova` /
   `confidence` / `shap_top_features` + `product_name`/`image_url` (for
   Stage 7.5) — never OFF's label. **Sparse honesty (PRD 6.8):**
   `data_sparse: bool` — `is_sparse_input` on the raw extracted row
   *before* Stage 2 clean (`build_feature_frame` → `(frame, bool)` with
   an assert the flag never lands in feature columns); when true the UI
   forces a "Low information" tier and the headline says "Not enough
   information to classify processing". Tests `tests/test_api.py` (35):
   zero network by default (fixture `tests/fixtures/off_product_3017620422003.json`
   deliberately keeps `nova_group`+Nutri-Score to prove stripping) +
   `TestDataSparse`; live
   e2e gated by `CHEWSY_LIVE_OFF=1`. Batch: `python src/batch_predict.py
   --input b.csv --output p.csv` (reuses `score_row`, per-row error
   isolation, `data_sparse` output column). Dep added: `httpx==0.28.1`
   (fastapi TestClient).
- **Evidence:** `python src/features.py` prints skew/filter/PCA/SVD
  numbers. Full evidence log: `docs/data_quality_report.md` —
  **local-only** (listed in `.git/info/exclude`): append each stage's
  findings for Q&A prep, never `git add`/push it.
- **`conftest.py`** at repo root puts the root on `sys.path` so bare
  `pytest` can `import src.*`.
- **DVC pipeline (Stage 10):** `dvc.yaml` = fetch → clean → features →
  train (+ register + package) with honest deps/outs; the three static
  `*.dvc` pointers were `dvc remove`d (files/cache/remote unchanged).
  `params.yaml` holds the only params the scripts read
  (`fetch.target_per_class: 5000`, `features.n_svd_components: 25`,
  `train.text_dropout_frac: 0.20`) via `src/params.py` — **dict-form
  params entries with dotted paths** (flat `params.yaml:key` strings are
  rejected by DVC 3.67). `mlflow.db`/`mlruns/`/`training_reference.csv`
  stay git-managed, never outs. `dvc.lock` was seeded with
  `dvc commit -f <stage>`; after any demo retrain, restore with
  `git checkout -- params.yaml dvc.lock mlflow.db mlruns` +
  `dvc checkout models/model.joblib` (champion v4 byte-identical), and
  re-run the mlflow-path relativize (commit 3da0c3f) before pushing if a
  register ever ships.

## Key commands

```bash
# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests
pytest

# Feature-engineering evidence (skew/filter/PCA/SVD printout)
python src/features.py

# Data pipeline (DVC remote = HF bucket; set AWS_* env first — see README)
dvc pull                       # restore training CSVs + model from the remote
dvc status -c                  # confirm cache/remote sync
dvc repro                      # Stage 10: fetch→clean→features→train;
                               # all-skip = up to date. Never `dvc add` the
                               # pipeline outs (pointers were removed)

# Packaging (Stage 5)
python src/package.py          # champion → models/model.joblib + reference
dvc commit train && dvc push   # re-pin the pipeline out (pointers removed)

# Experiment tracking (Stage 4)
python src/train.py --runs 1 2 3 4 5 6   # six tracked runs
python src/train.py --sweep 0 0.1 0.2 0.3 # text-dropout rate sweep (tracked runs)
python src/train.py --register           # winner → alias 'champion'
mlflow ui   # sqlite:///mlflow.db

# Local serving
uvicorn app.main:app --reload --port 8000
cd frontend && npm install && npm run dev   # Next.js on :3000

# Batch scoring (Stage 6.5)
python src/batch_predict.py --input barcodes.csv --output predictions.csv

# Container (API :8000, frontend :3000)
docker build -t chewsy .
docker run -p 8000:8000 -p 3000:3000 chewsy
```

## Standing risks (PRD §6)

- `nova_group` isn't in the bulk TSV export by default (resolved: Stage 1
  now fetches training data from the live search API instead; the TSV
  itself was deleted from disk — only the gitignore entry remains)
- India-only training is unviable (< 25 labeled rows/class) — India is a
  live-demo talking point only, never a training filter
- NOVA class balance is enforced by construction (one `nova_groups:<n>`
  query per class in `fetch_training_set.py`) — verified ✅ at 20k
  (5000/5000/5000/5000 pre-dedup). If you ever re-pull, re-check the
  printed balance; if any class comes back short of ~200 rows, check
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
