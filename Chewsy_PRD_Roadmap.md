# Chewsy — Product Requirements Document & Execution Roadmap

**Course:** Feature Engineering & MLOps mini-project
**Submission deadline:** 26 Sept 2026
**Presentation slot:** 26 Sept 2026, 9:00 AM – 2:00 PM
**Today:** 24 Sept 2026 — **~2 days of build time remaining**

---

## 1. Product Overview

**One-liner:** A Yuka-style barcode scanner. You type or look up a real product
barcode, the backend live-fetches that product from the Open Food Facts public
API, a trained model predicts its **NOVA processing level (1–4)**, and the UI
shows the verdict plus a SHAP explanation of *why*.

**Why it matters (use this framing in the pitch, not "I analyzed a dataset"):**
Open Food Facts is real, crowdsourced, and — critically — **most products in it
have no NOVA label at all**. This tool doesn't repeat a known answer; it fills
in a classification the community hasn't gotten to yet, the same job Yuka
(60M+ users) does commercially.

**Trained vs. live (the rule — state it exactly like this if asked):**
- **Offline, once:** the model. Stage 1 fetches 20,000 community-labeled
  products (5,000/class → 19,998 after Stage 2 cleaning) → Stage 4
  trains → Stage 5 freezes `model.joblib`. No live
  retraining, ever.
- **Live, every scan:** only the input. The scanned barcode is almost
  certainly not a training row — the app fetches that product's raw
  nutrients/ingredients from the OFF API at that instant and feeds them
  into the frozen model.
- **The model runs on every single scan, 100% of the time** — it never
  reads OFF's stored `nova_group`, even when the product has one. It's
  never a passthrough and never conditioned on "OFF is missing a label."
- **Train/predict distinction (survives a judge pressing on "you trained
  on their labels"):** training labels *come from* OFF's community (the
  Stage 1 `nova_groups:<n>` query); prediction never *reads* OFF's label —
  only the raw ingredients/nutrients, which are almost always present.
- **Why OFF's missing labels matter here:** not as the product's excuse —
  as *evidence the model is real*. Scan a labeled product (e.g. Nutella,
  OFF says NOVA 4): the model independently computes 4 from raw
  ingredients without peeking → credibility. Scan an unlabeled regional
  snack: the model still produces a confident 1/2/3/4, because it only
  ever needed the raw ingredients → a pure lookup tool has nothing to say
  about this product; we do, because it learned what ultra-processed food
  *looks like*, not the answer for specific barcodes.

**Target NOVA classes:**
1 = unprocessed/minimally processed, 2 = processed culinary ingredients,
3 = processed foods, 4 = ultra-processed foods.

**Non-goals (say this explicitly if asked, don't apologize for it):**
- Not predicting Nutri-Score — that's a public formula on nutrient columns;
  predicting it would be reverse-engineering arithmetic, not real ML.
- Not building a meal-logging or calorie-tracking app.
- Not attempting global coverage — training pulls are class-balanced and
  deliberately capped (5000/class in Stage 1, under the API's 10,000/class
  hard ceiling), not a full-corpus crawl;
  India is a demo story, not a training filter (see Stage 1).

---

## 2. Grading Rubric → Deliverable Map

| Requirement | Where it lives in this project |
|---|---|
| GitHub repo, real commit history | Stages 0–13 below = your commit sequence |
| Working UI → API → model | Next.js frontend → FastAPI `/predict` → `model.joblib` |
| Docker Hub image via CI/CD | GitHub Actions workflow, Stage 9 |
| DVC for dataset + pipeline | `dvc.yaml` stages, Stage 1 & 10 |
| MLflow experiment tracking + registry | Stage 4 |
| Cloud deploy (bonus) | Stage 11, only if time survives |
| "Why this over the obvious alternative" for every choice | Called out inline in every stage below |

---

## 3. Timeline (time-boxed — treat MUST as non-negotiable, SHOULD as cut-if-behind)

| Block | Window | Stages | Priority |
|---|---|---|---|
| Day 1 AM | Today, next 3–4 hrs | 0, 1, 2 | MUST |
| Day 1 PM | Today, next 4–5 hrs | 3, 4 | MUST |
| Day 1 evening | 2–3 hrs | 5, 6 | MUST |
| Day 2 AM | 3–4 hrs | 7, 8, 9 | MUST |
| Day 2 midday | 2 hrs | 10 | MUST |
| Day 2 PM | 2–3 hrs | 12 (light), slides | MUST |
| Day 2 evening | 1–2 hrs, only if ahead | 11 (EC2) | SHOULD (bonus) |
| Day 2 night / morning of 26th | 1–2 hrs | 13 (rehearsal + backup recording) | MUST |

If you're behind schedule at any checkpoint: **cut Stage 11 first, then trim
Stage 12 to a single `/metrics` endpoint with no live drift job.** Never cut
Stages 4, 5, 6, 9, or 10 — those are the ones the rubric explicitly names.

---

## 4. Repo Structure (create this in Stage 0, don't improvise it later)

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
│   ├── fetch_training_set.py       (live API pull — Stage 1, replaces slice_openfoodfacts.py)
│   ├── clean.py                    (Stage 2)
│   ├── features.py                 (Stage 3 — custom transformers)
│   ├── train.py                    (Stage 4)
│   ├── package.py                  (Stage 5 — registry → model.joblib + reference)
│   └── batch_predict.py             (Stage 6 — offline scoring of a CSV of barcodes)
├── models/
│   ├── model.joblib            (Stage 5, gitignored — DVC-tracked: model.joblib.dvc)
│   └── training_reference.csv  (Stage 12 — drift baseline, git-committed)
├── app/
│   ├── main.py                 (FastAPI — Stage 6)
│   ├── off_client.py           (live Open Food Facts API wrapper — Stage 6)
│   └── schemas.py              (Pydantic models — Stage 6)
├── frontend/                    (Next.js app — Stage 7)
│   ├── app/
│   │   ├── page.tsx             (scanner UI)
│   │   └── layout.tsx
│   ├── components/
│   │   └── BarcodeScanner.tsx   (camera capture + client-side decode)
│   ├── package.json
│   └── next.config.js
├── tests/
│   ├── test_features.py
│   ├── test_train.py
│   ├── test_package.py
│   └── test_api.py
├── Dockerfile                   (Stage 8 — multi-stage: Node build + Python runtime)
├── entrypoint.sh                 (Stage 8)
└── .github/workflows/ci.yml      (Stage 9)
```

---

## 5. Stage-by-Stage Tasks

Each task below is written to be handed directly to a coding agent. Every
task has an explicit **Done when** condition — don't move to the next task
until that condition is verifiably true.

### Stage 0 — Repo & Environment Setup
**Goal:** a real Git history starts here, not with a giant dump commit.

- [x] 0.1 `git init`, create `.gitignore` **before** the first `git add`
  (must exclude `.venv/`, `data/raw/*.csv` + `data/processed/*.csv`
  — ignore only the data files, **not** the directories: a
  `data/raw/*` pattern makes DVC's `collect_files` prune `data/raw/`
  and miss the `.dvc` pointer),
  `en.openfoodfacts.org.products.tsv`, `models/*.joblib`,
  `__pycache__/`, `frontend/node_modules/`, `frontend/.next/` —
  the last two don't exist yet at this stage, but set them up front so
  Stage 7's `npm install` doesn't dump a huge, ungitignored `node_modules/`
  into your next commit).
  **Do NOT ignore** `mlflow.db`, `mlruns/`, `dvc.yaml`, `dvc.lock`, or any
  `*.dvc` pointer — those are the MLflow/DVC grading evidence and must be
  committed with the repo.
  **Done when:** `git status` shows no venv/data/TSV files as
  untracked-but-wanted, and a dry-run confirms `mlflow.db`/`mlruns/`/`*.dvc`
  would be addable once they exist.
- [x] 0.2 Create the folder structure in Section 4 (empty `.gitkeep` files
  where needed).
  **Done when:** structure matches Section 4 exactly.
- [x] 0.3 `requirements.txt` pinned: `pandas`, `scikit-learn`, `fastapi`,
  `uvicorn`, `mlflow`, `dvc`, `shap`, `requests`, `joblib`,
  `pytest`, `python-multipart`. (Frontend deps are separate — `frontend/package.json`
  in Stage 7, not this file.)
  **Done when:** `pip install -r requirements.txt` succeeds in a fresh venv.
- [x] 0.4 First commit: `chore: repo scaffolding`.
  **Done when:** `git log` shows this as commit #1 with only structure/config
  files, no data or model files.

### Stage 1 — Data Acquisition (revised: live API, not the bulk TSV)
**Goal:** a balanced, correctly-labeled NOVA training set.

**Why this changed from the original plan:** the bulk TSV export
(`en.openfoodfacts.org.products.tsv`) turned out not to include
`nova_group` at all — confirmed against OFF's own export docs:
computed fields like `nova_group` are opt-in "extra_fields" on custom
exports, not present in the standard dump. Also confirmed separately:
filtering that file to India-only rows produced under 25 labeled rows
per class, far below the ~200/class floor — India-only training was
never viable regardless of the missing-column issue. **Fix:** pull
labeled rows directly from Open Food Facts' live search API
(`search.openfoodfacts.org`) instead — the same live API the FastAPI
backend already calls at inference time (Stage 6), so this reuses a
skill you need anyway rather than adding a new one. `slice_openfoodfacts.py`
is retired; `fetch_training_set.py` replaces it.

- [x] 1.1 Run `python src/fetch_training_set.py` (already written and
  live-tested against the real API — pulls 5000 rows per NOVA class,
  4 classes, balanced by construction since each class is fetched by an
  explicit `nova_groups:<n>` query. Originally 1500/class; rescaled to
  5000/class on request — 20k rows total, ~5.6% of the bulk dump's row
  count. The API hard-caps paging at 50 pages × 200 = 10,000/class
  (page 51 → HTTP 400), so 5000/class sits comfortably under the
  ceiling with room to scale further.)
  **Done when:** `data/raw/openfoodfacts_training_set.csv` exists with
  20,000 rows and a perfectly even `nova_group` value count across
  1/2/3/4 (verified in testing: 50/50/50/50 on a smoke-test run at
  `TARGET_PER_CLASS=50` before scaling up to the real pull).
  ✅ 20,000 rows, exactly 5000/5000/5000/5000.
- [x] 1.2 Expect real, uneven missingness on optional fields
  (`additives_n`, `fiber_100g`, `unknown_ingredients_n` were ~65–80%
  null in live testing at 1500/class; at 5000/class the deeper pages
  are a bit better documented — see null % report from `clean.py`) —
  this is normal OFF sparsity, not a fetch bug; Stage 2's conditional
  imputation (now the Stage 3 pipeline's `GroupMedianImputer`) already
  accounts for it.
  ✅ Observed at 20k: `unknown_ingredients_n` 69%, `additives_n` 70%,
  `fiber_100g` 53% null — matches expectation.
- [x] 1.3 India stays a **live-demo talking point only**, not a training
  filter — the model trains on a global, class-balanced pull; the live
  barcode lookup at demo time can still be any real product, Indian or
  otherwise, since that's a separate live API call at inference time,
  unrelated to what the training set was built from.
- [x] 1.4 `dvc init`, then `dvc add data/raw/openfoodfacts_training_set.csv`.
  **Done when:** a `.dvc` file exists for the training set and it's the
  object being git-tracked, not the CSV itself.
  ✅ Pointer committed in `e4340cc`; CSV is gitignored. (Fixed a
  `.gitignore` bug where `data/raw/*` made DVC prune the dir and miss
  the pointer — now `data/raw/*.csv` only.)
- [x] 1.5 Commit: `feat: initial data acquisition via live API + DVC tracking`.

### Stage 2 — Data Verification & Cleaning
**Goal:** the same rigor applied to every prior dataset in this project —
verify before trusting.

- [x] 2.1 Load `openfoodfacts_training_set.csv`, report: row count, null %
  per column, duplicate `code` (barcode) count, `nova_group` value counts,
  and value ranges for
  every `*_100g` nutrient column (flag anything negative or implausibly
  large — e.g. `energy_100g` > 4000 kcal/100g is almost certainly bad data).
  **Done when:** a printed diagnostic report exists and every flagged
  anomaly has a documented decision (drop / cap / impute) in code comments.
  ✅ `clean.py::report()` prints all of the above plus an independent
  consistency pass: macro sums vs 100g, energy vs Atwater estimate,
  near-duplicate name+brand groups, brand/text hygiene. Observed at 20k:
  salt max 1630 g, energy max 8200 kcal, 1 negative fiber, 46 HTML-entity
  names, 16107 list-repr brands — every flag has a code-commented decision.
- [x] 2.2 Drop exact-duplicate `code` rows (keep first).
  ✅ 2 dropped (19,998 remain; classes 1/3/4 at 5000, class 2 at 4998).
- [x] 2.3 Missing nutrient handling: for each `*_100g` column, impute using
  the **median within the same `categories_tags` group**, not the global
  median (justify: a snack's missing fiber should be filled from other
  snacks, not from beverages). Add a `<col>_was_missing` indicator column
  for at least `fiber_100g` and `sodium_100g` — missingness itself may be
  informative for less-documented (often smaller-brand) products.
  ✅ Group median → global median → 0 fallback ladder; indicators added for
  `fiber_100g`, `sodium_100g`, plus analysis-driven `text_was_missing` and
  `nutrients_all_missing` (17% of rows report no nutrient at all).
  **Deviation (25 Sep 2026):** imputation moved to Stage 3's pipeline
  (`GroupMedianImputer`, fit on train-only) — pre-split leakage fix;
  `clean.py` now preserves these NaNs; indicators unchanged (row-local,
  computed before any fill). See 2.6 deviation note.
- [x] 2.4 Normalize the text feature (currently named
  `ingredients_pseudo_text` — built by `fetch_training_set.py` from
  `ingredients_tags`): lowercase, strip punctuation noise, collapse
  whitespace. Do **not** remove stopwords yet — TF-IDF in Stage 3
  handles that via `stop_words='english'`.
  ✅ Lowercase + punctuation strip + whitespace collapse + HTML unescape;
  stopwords deliberately left for Stage 3 TF-IDF.
- [x] 2.5 **Leakage check (write this as an actual code comment, not just a
  mental note):** confirm no Nutri-Score-derived field is in the feature
  set — `fetch_training_set.py` already excludes them via
  `FORBIDDEN_LEAKAGE_FIELDS` (API names: `nutrition_grades`,
  `nutriscore_grade`, `nutriscore_score`, `nutriscore_data`; the old bulk
  export called this `nutrition_grade_fr`). It's a different computed label
  sitting next to the target, not a legitimate predictor — assert it's
  absent from the clean CSV here too, don't just trust the fetch script.
  ✅ Asserted twice in `clean.py` (on raw load and on final frame), with
  the PRD's rationale as an actual comment.
- [x] 2.6 Save output to `data/processed/openfoodfacts_clean.csv`,
  `dvc add` it.
  ✅ 19,998 rows × 22 cols, DVC-tracked.
  **Deviation (25 Sep 2026, leak fix):** the earlier "zero nulls at write"
  came from group-median imputation run on the *full* frame before the
  train/test split — pre-split leakage. `clean.py` is now row-local only
  and writes **45,453 nutrient nulls through on purpose**; imputation
  lives in the Stage 3 pipeline (`GroupMedianImputer`, fit on train rows
  only). Verified after the change: shape 19,998 × 22 unchanged, balance
  {1:5000, 2:4998, 3:5000, 4:5000} unchanged, indicators unchanged
  (fiber 10,501 / sodium 4,388 ones), leakage assert passes,
  `dvc status` up to date, final matrix still 0 non-finite.
  Full table: `docs/data_quality_report.md` §7 (local-only).
- [x] 2.7 Commit: `feat: data cleaning pipeline (clean.py)`.

### Stage 3 — Feature Engineering
**Goal:** every technique here should map to a phase in the course mindmap
and have a one-sentence "why this, not the obvious alternative" ready.

**Leak-fix addition (25 Sep 2026):** `GroupMedianImputer` (per-`categories_tags`
median → global-median fallback → `SimpleImputer` safety net) now sits between
the `create` and `skew` steps of the pipeline; `fit()` learns those medians on
whatever rows the pipeline is fit on — Stage 4 fits on train rows only, so no
held-out row ever contributes to its own filled value. At inference, an unseen
category group falls back to the global **train** median. Justification for
group medians is unchanged from 2.3 (a snack's missing fiber comes from other
snacks).

- [x] 3.1 Nutrient ratio features: `sugar_fiber_ratio`, `sat_fat_fat_ratio`.
  Justify: raw grams alone don't capture *proportion*, which is the
  nutritionally meaningful signal.
  ✅ `FeatureCreator` in `src/features.py`; sugar_fiber uses a 0.5 g fiber
  floor (finite ratios preserve the sugar signal for fiber-free products);
  sat_fat_fat returns 0.0 for fat==0, NaN for missing fat (imputed later).
- [x] 3.2 `brands` → frequency encoding (not one-hot — thousands of unique
  brands would explode dimensionality for little gain per brand).
  ✅ `FrequencyEncoder` (7,323 brands → 1 column); maps built in `fit()`
  only; unseen brands at inference → mean training frequency.
- [x] 3.3 `categories_tags` → same frequency-encoding treatment, or target
  encoding computed from **train rows only** (state explicitly in code
  which rows the encoding map was fit on, to avoid the leakage pattern
  called out in the mindmap).
  ✅ Frequency encoding (no target used at all), but tag-level, not
  exact-combo: measured 5,528 unique combos, top-50 cover only 33.8% of
  rows → score = mean document-frequency of the row's individual tags.
  Train-only by construction: maps built in `fit()`; Stage 4 fits the
  pipeline on train rows only (stated in the module docstring).
- [x] 3.4 Skew correction: log-transform `energy_100g`, `sugars_100g` (check
  skewness before/after with a quick `.skew()` printout as evidence).
  ✅ Printout in `python src/features.py`. Evidence-driven amendment:
  log1p **overshoots** energy so energy uses sqrt; sugars uses log1p as
  written. Current (NaN-preserving CSV) measurements: energy **+0.825 →
  −0.067 (sqrt)**, sugars **+2.179 → +0.533 (log1p)** (pre-fix
  measurements on the imputed CSV: 0.909 → −0.173 / 1.851 → 0.559;
  log1p on energy overshot to −1.934, which is why sqrt was chosen).
- [x] 3.5 **Filter-based selection on the numeric nutrient block** — before
  anything else touches these columns: drop any near-zero-variance
  column (`VarianceThreshold`), and check pairwise correlation between
  `fat_100g` and `saturated_fat_100g` (and any other suspiciously related
  pair) — drop or combine one side if correlation is high. This is the
  filter-method phase from the mindmap; it's cheap and it's the one
  selection family this project would otherwise skip entirely.
  ✅ `VarianceThreshold(1e-4)` runs first in the numeric branch (before
  scaling): 0 drops (variances 26.96–63,040.55 — fiber 26.96 is the
  minimum, all nutrients genuinely vary; pre-fix minimum was 10.18).
  Correlation check: the PRD-suggested fat vs saturated_fat = **0.626**
  → keep both; the real redundancy was **salt vs sodium = 0.967** with an
  exact 2.5× unit relation → `sodium_100g` dropped. No kept pair ≥ 0.9.
  (Pre-fix measurements: fat/sat-fat 0.648, salt/sodium 0.982 — same
  decisions.)
- [x] 3.6 **Scale the numeric nutrient block** (`StandardScaler`) inside the
  `ColumnTransformer`'s numeric branch — **required**, not optional, for
  two separate reasons: (a) the Logistic Regression baseline in Stage 4
  is distance/gradient-based and will be dominated by whichever nutrient
  happens to have the largest raw units (`energy_100g` in kcal vs.
  `fiber_100g` in grams) without it; (b) PCA in 3.8 is itself scale-sensitive
  — an unscaled PCA would just rediscover "whichever column has the
  biggest numbers," not a genuine energy-density axis. Scale *before* PCA,
  same transformer branch.
  ✅ Numeric branch = median imputer → VarianceThreshold →
  StandardScaler, sitting behind the pipeline-head `GroupMedianImputer`
  (see the leak-fix note above); PCA diagnostics scale the same block
  separately.
- [x] 3.7 TF-IDF on the ingredient text column
  (`ingredients_pseudo_text`, or `ingredients_text` if a future fetch
  variant provides raw text) (uni+bigrams, `max_features` capped
  at ~500, `min_df` to prune rare tokens, `max_df` to prune near-universal
  ones — this is the "vocabulary pruning" step from the mindmap).
  ✅ uni+bigrams, max_features=500 (vocab: 500), min_df=5, max_df=0.95,
  stop_words='english'. Enhancement: TF-IDF reads
  `product_name + " " + ingredients_pseudo_text` (names like "Potato
  chips" carry processing signal); empty-safe.
- [x] 3.8 **Dimensionality reduction (do both of these — they answer
  different mindmap phases):**
  - `TruncatedSVD` on the TF-IDF matrix (sparse-text equivalent of PCA) —
    compress to ~20–30 components, report explained variance.
  - `PCA` on the **scaled** numeric nutrient block (`energy`, `fat`,
    `sat-fat`, `carbs`, `sugars`, `fiber`, `proteins`, `salt`) — report a
    scree plot / explained variance ratio, and name what PC1 represents
    (likely an "energy density" axis) — this is your direct parallel to
    the mock-test PCA example.
  ✅ SVD: 25 components in the pipeline, 48.3% cumulative EVR (diffuse —
  ingredient vocabulary is spread across many terms; unchanged by the
  leak fix). PCA (analysis only): PC1 = **30.9%**; loadings fat +0.587 /
  sat-fat +0.510 / energy +0.506 vs carbs −0.244 / sugars −0.236 → PC1
  is still a **fat-energy-density vs sugar-carbohydrate axis** (PRD's
  "energy density" guess, but with the fat/sugar contrast made explicit;
  pre-fix: PC1 32.9%, fat +0.557 / sat-fat +0.495 / energy +0.454 vs
  carbs −0.325 / sugars −0.320). PCA deliberately NOT in the
  pipeline: Stage 4 SHAP must explain real nutrients, not components.
- [x] 3.9 Assemble everything into one `sklearn.Pipeline` +
  `ColumnTransformer` (numeric branch, categorical/frequency branch, text
  branch) inside `src/features.py`. **Any custom transformer class must
  live in this importable file**, not inline in a notebook — this is the
  exact `FeatureCreator` pickling bug the mindmap calls out; avoid it now.
  ✅ `build_feature_pipeline(include_text=, include_categorical=)` —
  3 branches, all custom classes module-level in `src/features.py`.
  Head steps in order: `create` → `impute` (`GroupMedianImputer`) →
  `skew` → `ColumnTransformer`. Flags exist for Stage 4's three runs.
  Final matrix: **19,998 × 44**
  (17 numeric + 1 brand + 1 category + 25 SVD), 0 non-finite values.
- [x] 3.10 Unit test: `tests/test_features.py` — pipeline `.fit_transform()`
  runs on a small sample without error and produces the expected shape.
  ✅ 20 tests pass: both shapes (numeric-only/full), ratio math, both
  encoders incl. unseen-value fallbacks, NaN/negative/unseen inference
  input, joblib round-trip (Stage 5 pickle guard), feature names (SHAP),
  diagnostics evidence keys, plus the leak-fix additions (`GroupMedianImputer`
  unit tests, shared-group NaN pipeline test, indicator
  compute/preserve tests).
- [x] 3.11 Commit: `feat: feature engineering pipeline`.

### Stage 4 — Modeling & Experiment Tracking
**Goal:** at least 3 genuinely different runs, tracked, with one promoted.

**Stage 3 handoff — use, don't rebuild:** `build_feature_pipeline()`
from `src/features.py` (output = 44 cols for the full config):
- Run 1 (4.2) = `include_text=False, include_categorical=False`
  → nutrients + ratios + counts + indicators only, exactly as 4.2 says.
- Run 2 (4.3) = `include_text=True, include_categorical=False`
  → Run 1 + TF-IDF→SVD text branch (the "add text" delta stays clean).
- Run 3 (4.4) = both `True` (defaults) → full feature set.
Feature names for the SHAP plot:
`pipeline.named_steps["features"].get_feature_names_out()`.
The pipeline includes `GroupMedianImputer` fit on train only, which
completes the "every learned statistic is fit on train" story — group
medians, frequency maps, scaler stats and TF-IDF idfs all come from the
training fold (Q&A: leakage question).
PCA deliberately has **no** features here (SHAP must explain real
nutrients; see 3.8) — but the PC1 finding (fat-density vs sugar-density,
30.9%) is presentation material (docs report §6, local-only file).

- [x] 4.1 Set `mlflow.set_tracking_uri("sqlite:///mlflow.db")`.
  ✅ Top of `src/train.py`; experiment `chewsy-nova`; `mlflow.db` +
  `mlruns/` committed (grading evidence, never gitignored).
- [x] 4.2 **Run 1 (baseline):** nutrients + ratios only, no text — Logistic
  Regression. Log params, accuracy/F1 (macro, since NOVA classes are
  imbalanced), confusion matrix as an artifact.
  ✅ `run1_logreg`: macro-F1 **0.7973**, accuracy 0.7980, log-loss 0.5258,
  confusion-matrix PNG logged. Split: stratified 80/20, `random_state=42`,
  split *before* any fit (train 15,998 / test 4,000). `class_weight="balanced"`
  is a near no-op by construction — classes are {1:5000, 2:4998, 3:5000,
  4:5000} — used anyway for standardness.
- [x] 4.3 **Run 2:** add TF-IDF + SVD text features — same model family.
  Log the same metrics; this run should meaningfully beat Run 1 or you have
  a genuine, presentable finding either way ("text didn't help as much as
  expected because...").
  ✅ `run2_logreg`: macro-F1 **0.8875** (+0.0902 over run 1) — text helps a
  lot; ingredient/name text carries real processing signal. Same LR family
  keeps the ablation delta clean.
- [x] 4.4 **Run 3:** swap to a tree-based model (Random Forest or XGBoost)
  on the full feature set. Log SHAP summary plot as an artifact.
  ✅ `run3_random_forest` (300 trees, full 44-col set): macro-F1 **0.9415**,
  log-loss 0.2001. SHAP artifacts: TreeExplainer on 200 fixed-seed train
  rows, one summary PNG per NOVA class (4), feature names from
  `get_feature_names_out()`. **Deviation (25 Sep 2026):** SHAP artifacts
  also logged for runs 4–5 (same code path, TreeExplainer supports both).
- [x] 4.5 Compare all 3 runs in the MLflow UI, pick the winner by macro-F1
  (accuracy alone is misleading with imbalanced NOVA classes).
  ✅ **Deviation (25 Sep 2026, approved): 6 runs, not 3** — model diversity
  added on top of the PRD's feature ablation (xgboost==3.4.1 added to
  `requirements.txt`). Final ranking by macro-F1:
  **run5 xgboost 0.9494** > run4 hist_gbdt 0.9442 > run3 random_forest
  0.9415 > run2 logreg+text 0.8875 > run6 knn 0.8555 > run1 logreg 0.7973.
  Winner per-class F1: {1: 0.9645, 2: 0.9864, 3: 0.9179, 4: 0.9288} —
  NOVA 3 (processed foods) is the hard class everywhere. `print_comparison`
  prints the table; `mlflow ui` shows the same.
- [x] 4.6 Register the winner in the Model Registry, alias it `champion`.
  **Done when:** `mlflow.pyfunc.load_model` can load it back by alias.
  ✅ `chewsy-nova` v1, alias `champion`, source = `champion_packaging_run5`
  (refit of the winning config on the same deterministic split);
  `mlflow.pyfunc.load_model("models:/chewsy-nova@champion")` verified —
  sample test row predicts NOVA 4 (true 4).
  **Deviation (25 Sep 2026): champion-only model logging.** PRD 4.2–4.4
  require params/metrics/plots, not model binaries. mlflow's default skops
  serialization inflated these pipelines ~5× (78–155 MB/model → 673 MB of
  `mlruns/` with all six, exceeding GitHub's 100 MB/file limit), so only the
  champion's model is logged — pickle format, **17 MB**; `mlruns/` = 19 MB
  total. Also: xgboost ≥ 2 rejects non-0-based class ids, so
  `NovaLabelAdapter` (in `src/features.py`, Hard rule 4) maps {1,2,3,4} →
  {0,1,2,3} and decodes predictions back; the logged input example marks
  indicator/text columns nullable so Stage-6-style rows pass pyfunc schema
  enforcement; the `code` barcode column is dropped from `X` (identifier,
  not a feature — uint64 values break signature inference).
- [x] 4.7 Commit: `feat: MLflow tracking + 3 experiments, champion registered`.
  ✅ Committed as `feat: MLflow tracking + 6 experiments, champion registered`
  (message amended to match the approved 6-run matrix).

### Stage 5 — Packaging
**Stage 3 handoff:** the feature-stage pickle round-trip already passes
(`tests/test_features.py::test_joblib_roundtrip_reproduces_transform` —
5.2 extends that to the *model*-bearing pipeline). All classes are
module-level in `src/features.py`, so a fresh `joblib.load` resolves
them; run tests from the repo root (root `conftest.py` puts it on
`sys.path`).

**Stage 4 handoff (25 Sep 2026):** the model is the **XGBoost champion**
(macro-F1 0.9494). Get it either way — both give identical weights
(refit is deterministic, stratified 80/20 `random_state=42`):
- `mlflow.sklearn.load_model("models:/chewsy-nova@champion")` →
  `joblib.dump(...)` as `models/model.joblib` (provenance-clean: it is
  the exact registered artifact), or
- refit `src.train.build_full_pipeline(5)` on the same split.
The Pipeline contains `NovaLabelAdapter` (xgboost label shim, lives in
`src/features.py` — Hard rule 4 already satisfied for pickle). `train.py`
drops `code` from `X`, so 5.3's `training_reference.csv` should snapshot
the feature columns without it. Comparison runs in MLflow carry **no**
model binaries by design (champion-only logging) — don't go looking for
per-run models; metrics/plots/registry are all committed and sufficient.
- [x] 5.1 `joblib.dump()` the full fitted pipeline (features + model) as
  `models/model.joblib`. — done by **`src/package.py`** (approved
  deviation: a small packaging script instead of an ad-hoc one-liner, so
  registry → artifact is reproducible and gradeable: `python src/package.py`).
  Source is `mlflow.sklearn.load_model("models:/chewsy-nova@champion")`
  per the Stage 4 handoff (provenance-clean); dump = **17.3 MB**,
  DVC-tracked (`models/model.joblib.dvc`) and pushed to the HF bucket —
  git carries the pointer, never the blob (matches the `.gitignore`
  `models/*.joblib` design).
- [x] 5.2 Verify it reloads cleanly in a fresh Python process (`joblib.load`
  → `.predict()` on one sample) — this catches the "can't get attribute
  FeatureCreator" bug before it reaches the API.
  `tests/test_package.py::test_fresh_process_loads_predicts_and_matches_registry`
  spawns a brand-new interpreter: load → predict on a reference row →
  4-class `predict_proba` sanity → **registry-vs-joblib prediction
  equality** (the packaged artifact *is* the registered champion).
  Full suite: 31/31 green.
- [x] 5.3 Save `models/training_reference.csv` — a snapshot of the training
  feature distributions, needed for Stage 12 drift checks.
  Built by `src/package.py` from `load_dataset()`'s column selection:
  **19,998 × 20**, git-committed, NaN-preserving. "20" reconciles the two
  handoffs: Stage 4's "feature columns without `code`" + Stage 12's "live
  input space" both mean the 22 cleaned columns minus `code` (barcode id)
  and `nova_group` (target — Hard rule 11: a live row never carries it).
- [x] 5.4 Commit: `feat: packaged model artifact`.

**DVC remote on HuggingFace (approved deviation, 25 Sep 2026):** until
this stage there was **no DVC remote at all** — `dvc push`/`dvc pull`
were impossible and every clone was stranded without the CSVs/model.
Setup (HF's documented "Version data with DVC" recipe):
- default remote `.dvc/config` (committed, no secrets):
  `url = s3://chewsy-dvc/dvc-store`, `endpointurl = https://s3.hf.co/prem2903`,
  `region = us-east-1` → a private HF Storage Bucket under our namespace;
- credentials: HF token → *Generate S3 credentials* → `HFAK…` pair, kept
  in git-ignored `.dvc/config.local` (machine) / CI secrets (Stage 9) —
  never in git, never in chat-repo;
- `dvc[s3]==3.67.1` in `requirements.txt`; push/pull needs
  `AWS_REQUEST_CHECKSUM_CALCULATION=when_required` +
  `AWS_RESPONSE_CHECKSUM_VALIDATION=when_required` (recent botocore's
  trailing CRC32 breaks the gateway);
- **verified:** `dvc push` (2 CSVs + model), `dvc status -c` in sync,
  fresh `git clone` → `dvc pull` → byte-identical files (md5 match),
  `pytest` 31/31.

**Stage 9 handoff (CI):** a fresh clone has no `.dvc/config.local` —
store `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (the HFAK pair) and the
two checksum env vars as GitHub Actions secrets before any job that runs
`dvc pull`.

**Stage 10 handoff:** pipeline `outs` in `dvc.yaml` cannot coexist with
static `.dvc` pointers for the same file — when wiring stages 10.1/10.2,
first `dvc remove data/raw/openfoodfacts_training_set.csv.dvc
data/processed/openfoodfacts_clean.csv.dvc models/model.joblib.dvc`
(then `dvc add`-recreate after any repro that must re-pin, or let the
pipeline own them); the HF remote config in `.dvc/config` stays as-is.

### Stage 6 — FastAPI Serving Layer
**Stage 2/3 handoff — the feature row the API sends:**
- Input to the fitted pipeline is a **one-row `pd.DataFrame`** with the
  columns `fetch_training_set.py` keeps (the cleaned input space).
- Live rows may now contain nutrient **NaNs** (that is what the
  NaN-preserving clean CSV ships, so inference looks like training): the
  pipeline fills them with **group medians** — same `GroupMedianImputer`
  logic used at training time — then the safety-net median imputer in the
  numeric branch. Unseen category group → global **train** median.
- The API does **not** need to provide the indicator columns
  (`*_was_missing`, `text_was_missing`, `nutrients_all_missing`) —
  `FeatureCreator` computes them when absent, straight from the raw
  missingness of the incoming row (training rows already carry them from
  `clean.csv` and those stored values are preserved).
- The pipeline is inference-robust *inside* (empty text → `""`, unseen
  brand/tag → mean training
  frequency, negative nutrients → NaN → impute) — but it does **not**
  re-run Stage 2's field normalizations. The API must apply the same
  normalizations to the raw OFF response first: parse list-repr brands
  (`['X']` → `X`, lowercase), cap mass values at 100 g, negatives →
  NaN, HTML-unescape names. NaN is now handled *inside* the pipeline, so
  normalization only has to make the row's fields the right *shape*, not
  fill values. Consider exposing a shared helper from
  `src/clean.py` rather than duplicating the logic (Hard rule: one
  source of truth).
- `nova_group` from the OFF response must never enter that frame
  (Hard rule 11) — assert it, same pattern as Stage 2.5.

**Stage 4 handoff (25 Sep 2026):** the shipped model is the XGBoost
champion wrapped in `NovaLabelAdapter` — it predicts NOVA **1..4** (the
adapter decodes xgboost's 0-based ids) and exposes `predict_proba`, so
`/predict`'s **confidence comes from the model itself**, never from OFF.
Indicator columns (`*_was_missing`) are absent from API rows *and* absent
from the MLflow signature's required columns (input example marks them
optional) — `FeatureCreator` computes them at transform time, exactly as
the pipeline's inference-robust behavior above describes. The joblib
path has no schema enforcement anyway; this only confirms the row shape
matches what the champion was packaged with.

**Goal:** this is where the "live" feeling of the product actually lives.

- [x] 6.1 `app/off_client.py`: a function `fetch_product(barcode: str)` that
  calls `https://world.openfoodfacts.org/api/v2/product/{barcode}.json`,
  with a timeout (e.g. 5s) and a clear fallback/error path if the barcode
  isn't found or the API is unreachable — **do not let this crash the
  request**, return a clean 404/503 with a message instead.
  **Inference leakage guard (required):** the product API response
  includes `nova_group` for already-labeled products — strip it / assert
  it never reaches the feature row (mirror `FORBIDDEN_LEAKAGE_FIELDS`
  from `fetch_training_set.py`). The model must compute the verdict from
  raw nutrients/ingredients only, never read OFF's stored label, even
  when one exists.
  — done: `fetch_product` uses `requests.get(timeout=5.0)` + a
  `User-Agent`; failures raise typed `ProductNotFound` / `OffAPIUnavailable`
  which `main.py` maps to **404 / 503** (verified live: `0000000000000` →
  404, timeout class → 503). `FORBIDDEN_RESPONSE_FIELDS =
  FORBIDDEN_LEAKAGE_FIELDS ∪ {nova_group}` is stripped by
  `strip_forbidden()` (asserts nothing survives) and re-asserted twice
  downstream (`normalize_live_row`, `build_feature_frame`) — three
  independent guards, same pattern as PRD 2.5. `extract_feature_input()`
  mirrors Stage 1's `flatten_hit()` mapping (`energy-kcal_100g` kcal, not
  the kJ `energy_100g`; comma-joined `categories_tags`; tag-derived
  pseudo-text) so live rows look like training rows.
  **Code-review fixes (25 Sep):** (a) multi-brand plain strings
  (`"Nutella, Ferrero"`) are cut to the first segment so live brands land
  in the training `FrequencyEncoder` vocab — measured 203/19,998 training
  rows carry a comma; `nutella` ∈ vocab vs `nutella, ferrero` ∉, and the
  old test asserted the skewed value; (b) any non-200 (429/5xx) and
  malformed payload shapes now raise `OffAPIUnavailable` — a 429 body with
  `{"status":0}` used to surface as a bogus 404, a list-typed `product`
  as a 500; both paths now covered by 9 mocked-response tests that run
  without the live gate.
- [x] 6.2 `app/schemas.py`: Pydantic request/response models for
  `POST /predict` (input: barcode string; output: NOVA class, confidence,
  top SHAP features for this prediction). Field names/docstrings must
  make clear the NOVA class is **model-computed** (e.g.
  `predicted_nova`), not fetched from Open Food Facts.
  — done: `PredictRequest` (barcode, `^\d{6,14}$` → 422 on junk) and
  `PredictResponse(predicted_nova, nova_label, confidence,
  class_probabilities, shap_top_features)`; docstrings state explicitly
  that the class comes from `model.joblib` and that `nova_group` is
  stripped and never echoed. **Approved enhancement:** `product_name` +
  `image_url` included — Stage 7.5's result view needs them from this
  response.
- [x] 6.3 `app/main.py`:
  - Load `model.joblib` once at startup (lifespan event), not per-request.
  - `GET /health` — trivial liveness check.
  - `POST /predict` — takes a barcode, calls `off_client.fetch_product`,
    builds the feature row (assert no `nova_group` in it), runs the
    pipeline, computes a SHAP explanation for that one prediction,
    returns it. The model runs on **every** scan — never a passthrough,
    never conditioned on whether OFF already has a label.
  - `GET /metrics` — request count, average latency (simple in-memory
    counters are fine for this project's scope).
  - **Enable CORS** (`fastapi.middleware.cors.CORSMiddleware`) for the
    Next.js origin — the browser will block the frontend's fetch calls
    without this, and it's the single most common "works with curl, fails
    in the browser" bug you'll hit in Stage 7.
  — done: lifespan loads the joblib once and builds a **cached
  `shap.TreeExplainer`** over the unwrapped `XGBClassifier` plus the 44
  feature names (same `named_steps` navigation as `train.py::feature_names`);
  `/predict` = fetch → `normalize_live_row` → `score_row` → top-5 SHAP
  values for the **predicted** class (`NovaLabelAdapter` maps NOVA
  `p` → estimator column `p-1`); `/metrics` = requests/errors/avg latency
  (live: ~520 ms/scan incl. OFF fetch); CORS origins from env
  `FRONTEND_ORIGINS` (default `http://localhost:3000`) — verified the
  `access-control-allow-origin` header on an OPTIONS preflight. SHAP
  features are the real engineered columns (`num__…`, `text__…`,
  `cat_tagfreq__…`), never PCA components.
- [x] 6.4 `tests/test_api.py`: at minimum, test `/health` returns 200 and
  `/predict` returns a well-formed response for one known real barcode.
  — done: 17 tests. A **captured real OFF payload**
  (`tests/fixtures/off_product_3017620422003.json`, keeps `nova_group=4`
  + the Nutri-Score family on purpose) is served via a monkeypatched
  `fetch_product`, so the default suite makes **zero network calls**
  (CI-safe, Stage 9). Covers: leak-guard stripping, `nova_group`/Nutri-Score
  rejection, Stage 2 normalization on the live row (incl. kcal-not-kJ),
  `/health`, `/metrics` increments, well-formed `/predict` (≤5 SHAP
  features, probs sum to 1, no forbidden key anywhere in the body),
  route-equals-direct-model, 404/503/422, and batch scoring with failure
  isolation. True end-to-end live test gated by `CHEWSY_LIVE_OFF=1`
  (passes). Full suite: **47 passed, 1 gated skip**.
  **Approved deviation:** added `httpx==0.28.1` to `requirements.txt` —
  `fastapi.testclient` requires it.
- [x] 6.5 `src/batch_predict.py`: loads `model.joblib` once, takes a CSV of
  barcodes as input, scores all of them, writes predictions to an output
  CSV. This is the "batch vs. real-time inference" phase from the mindmap
  — the API handles one request at a time; this script is for scoring a
  whole backlog of unlabeled OFF products offline, which is a genuinely
  different use case (and a nice thing to mention in the pitch: "this is
  literally how you'd backfill NOVA labels for the products the community
  hasn't gotten to yet").
  — done: `--input/--output/--barcode-col`; reuses the API's scoring path
  (`app.main.score_row` — one source of truth, no SHAP since explanations
  are a per-scan UI concern); per-barcode failures become an `error` column
  instead of aborting the batch. Live run over 4 barcodes: **3 scored +
  1 clean "not found" error**, ~0.5 s/barcode.
  **Code-review fix:** per-row isolation widened to *any* `Exception`
  (a malformed payload — e.g. non-iterable `categories_tags` — previously
  killed the run and lost already-scored rows) + barcode-format validation
  before anything is sent to OFF.
- [x] 6.6 Commit: `feat: FastAPI prediction service + batch scoring script`.
- [ ] 6.7 **Scope addition (approved 25 Sep 2026 — NOT yet built; planned
  together with Stage 7):** extend `PredictResponse` with the facts a
  scan-result screen needs (data is already in hand during scoring — zero
  extra API calls):
  - `nutrition_100g`: the 8 per-100g values taken from the **normalized**
    row (Stage 2 clean values — capped, invalid → `null`; frontend renders
    `null` as "—", never a raw impossible number),
  - `additives_n`, `ingredients_n` (ints),
  - `ingredients_text` (raw OFF text — what's actually on the package).
  Must NOT add any OFF label field (Hard rule 11 unchanged); extend
  `tests/test_api.py` to assert the new fields (e.g. Nutella: energy 539,
  salt 0.107). Feeds Stage 7.8's facts panel.

### Stage 7 — Next.js Frontend
**Goal:** a real, camera-driven scanner UI — not a form. This is the layer
that has to feel like a product, so it gets more care than a typical
course-project UI.

**Result-screen design (approved 25 Sep 2026 — brainstormed scope change):**
Structure = **single scrollable result card** (approach chosen over
tabbed/accordion variants: nothing hidden from judges, least interactive
state to break in Stage 8's container). Top → bottom:
1. **Hero** — image, product name, barcode (small mono), color-coded
   verdict badge (green→red across NOVA 1→4) + plain label, confidence
   chip with tier word: ≥0.8 "Confident" · 0.6–0.8 "Fairly sure" ·
   <0.6 **or** top-2 gap <15 pts → "**Borderline**" (honest for
   cheese-stick-type 0.87/0.12 results).
2. **Probability distribution** — 4 bars, each labeled with class name +
   %, predicted class highlighted; makes model ambiguity *visible*.
3. **Why — SHAP chart** (`recharts`, PRD 7.5 baseline) — top-5, sign =
   toward/away from verdict; humanized labels (`num__additives_n` →
   "Number of additives", SVD components → "learned ingredient-text
   signal").
4. **Facts panel** — per-100g nutrition grid + additive/ingredient counts
   + full ingredient list (needs 6.7).
5. **NOVA explainer** — the only collapsed element (tap "What is NOVA
   1–4?").
6. Footer microcopy: *"Chewsy model verdict — computed from ingredients,
   not copied from OFF"* (pitch credibility).
Plus: session **scan-history chips** (last 10) on the scanner screen and a
"Scan another" reset. Scope tiers chosen: A (existing response fields) +
B (frontend-only) + C (6.7 response extension). **Rejected:** tabbed
layout, showing OFF's `nova_group`/Nutri-Score (Hard rule 11 + leakage
optics with judges), anything requiring a second backend fetch.

- [ ] 7.1 Scaffold with `create-next-app` (TypeScript, App Router) inside
  `frontend/`.
- [ ] 7.2 `components/BarcodeScanner.tsx`: use the browser's camera via a
  client-side barcode-decoding library — **`html5-qrcode`** is the easiest
  fit (handles camera permission prompts and supports EAN-13/UPC, the
  formats real grocery barcodes use, not just QR codes). Decoding happens
  entirely in the browser; only the decoded barcode *string* gets sent to
  the backend.
- [ ] 7.3 Keep a manual text-entry fallback input alongside the camera view
  (not a compromise — every real scanner app has this, for when lighting
  or focus fails mid-demo).
- [ ] 7.4 On decode/submit: `fetch(POST, '<API_URL>/predict', { barcode })`.
  **This frontend must call the FastAPI endpoint over HTTP — it must never
  reimplement scoring logic or duplicate the model on the frontend.** Same
  "one source of truth" principle the mindmap describes for Streamlit,
  just carried over to this stack.
- [ ] 7.5 Result view (single scroll card per the approved design above):
  product name/image from the API response, a color-coded NOVA verdict
  badge (green→red across 1→4) + plain label, confidence chip with tier
  word (Confident / Fairly sure / Borderline), and the SHAP top-features
  rendered as a simple bar chart (e.g. `recharts`) with humanized
  feature labels.
- [ ] 7.6 `NEXT_PUBLIC_API_URL` as an env var, not a hardcoded localhost URL
  — you'll need this to differ between local dev and the Docker/EC2 build.
- [ ] 7.7 `components/ProbabilityBars.tsx`: the 4-class distribution —
  every bar labeled with NOVA class name + %, predicted class highlighted;
  this is what makes model ambiguity visible (e.g. cheese stick
  0.87 / 0.12 instead of a silent single number).
- [ ] 7.8 Facts panel: per-100g nutrition grid (`null` → "—"), additive +
  ingredient counts, full ingredient list — **depends on 6.7** (do 6.7
  first; it is a backend change with its own tests).
- [ ] 7.9 Scan flow extras: NOVA explainer (the one collapsed element),
  session scan-history chips (last 10: "Nutella → 4"), "Scan another"
  reset.
- [ ] 7.10 Error/edge states + typing: 404 → "Product not found in OFF",
  503/network → "OFF unreachable — retry" with retry button, camera
  denied → manual entry emphasized, missing image → placeholder;
  `types/predict.ts` mirrors the Pydantic response schema exactly.
- [ ] 7.11 Verification: `npm run typecheck` + `npm run build` green, one
  manual camera e2e against the local API, and 6.7's extended pytest
  suite green (no frontend test suite required by the PRD).
- [ ] 7.12 Commit: `feat: Next.js scanner UI`.

### Stage 8 — Containerization
**Goal:** one image, both a Python and a Node runtime inside it — this is
the one place the stack swap actually adds real complexity, budget extra
time here.

- [ ] 8.1 **Multi-stage `Dockerfile`:**
  - Stage A (`node:20-slim`): `COPY frontend/`, `npm install`, `npm run build`
    → produces a production Next.js build.
  - Stage B (`python:3.11-slim`): install Python deps (`COPY
    requirements.txt` before app code, for layer caching), **also install
    Node.js runtime** (needed to run `next start`, not just to build),
    `COPY` the built frontend output from Stage A, `COPY` the FastAPI app.
- [ ] 8.2 `entrypoint.sh`: start `uvicorn` on `:8000`, poll `/health` until
  it responds, then start `next start` on `:3000`. Both must stop together
  on `SIGTERM`.
- [ ] 8.3 Local test: `docker build` + `docker run`, hit both ports from the
  host.
  **Done when:** a barcode scanned through the containerized frontend
  returns a real prediction end-to-end, not just "container starts."
- [ ] 8.4 Commit: `feat: Docker containerization`.

### Stage 9 — CI/CD to Docker Hub
- [ ] 9.1 `.github/workflows/ci.yml`: on every push — run `pytest`, then (only
  if tests pass) `docker build` + `docker push` to Docker Hub using repo
  secrets for credentials.
  **Done when:** a push to `main` results in a new tagged image visible on
  Docker Hub, triggered by GitHub Actions — not a manual `docker push` from
  your laptop.
- [ ] 9.2 Commit: `ci: GitHub Actions build + push to Docker Hub`.

### Stage 10 — Wire Up the DVC Pipeline
**Goal:** the whole thing becomes one re-runnable, versioned pipeline —
this is the step that makes Stages 1–5 legible as a *pipeline*, not just a
sequence of scripts you happened to run in order.

- [ ] 10.1 `dvc.yaml` with stages: `fetch` → `clean` → `features` → `train`,
  each with explicit `deps` and `outs` pointing at the real file paths from
  Section 4.
- [ ] 10.2 `dvc repro` runs the whole thing end-to-end from the tracked raw
  training set to a fresh `model.joblib`.
  **Done when:** `dvc repro` with no changes reports everything up to date;
  changing one parameter (e.g. `TARGET_PER_CLASS` or an SVD component count) and
  re-running only re-executes the affected downstream stages.
- [ ] 10.3 Commit: `feat: DVC pipeline (dvc.yaml + dvc.lock)`.

### Stage 11 — Cloud Deployment (bonus, only if on schedule)
- [ ] 11.1 Launch EC2 instance, open ports 22/8000/3000 in the security
  group.
- [ ] 11.2 `docker run --restart unless-stopped` pulling the Docker Hub
  image.
  **Done when:** the UI is reachable from a browser using the instance's
  public IP, not just `localhost`.

### Stage 12 — Lightweight Monitoring (trim to this if short on time)
**Stage 2/3 handoff:** `training_reference.csv` should snapshot the
**input** space — the 22 cleaned columns (what a live API row looks like
after Stage 2 normalization), not the 44 transformed features — drift is
about what *arrives*, and live rows start in that space too. It is built
from the **NaN-preserving** clean CSV, so the KS comparison should also
compare per-column **null fractions** (a documentation-drift signal is a
null-rate change: OFF pages that stop recording fiber show up as a rising
`fiber_100g` NaN share before any value distribution moves).
- [ ] 12.1 At minimum: structured request logging in FastAPI (barcode,
  latency, predicted class) and a working `/metrics` endpoint.
- [ ] 12.2 If time allows: a small script comparing live-fetched nutrient
  distributions against `training_reference.csv` using `scipy.stats.ks_2samp`
  — even a single manual run of this, shown as a plot in your slides, is
  enough to demonstrate the concept without a full always-on drift job.

### Stage 13 — Presentation Prep
- [ ] 13.1 Slides: problem (Yuka framing) → live demo → brief technical
  depth (pick 2–3 "why this, not that" moments to go deep on, don't try to
  narrate every stage above) → MLOps pipeline diagram → close.
- [ ] 13.2 Rehearse the live demo with **3 real barcodes** picked in advance
  — one confidently NOVA 1, one confidently NOVA 4, one genuinely
  ambiguous (best for the SHAP-explanation moment).
- [ ] 13.3 Record a backup video of the full demo working, in case of
  live network/deployment issues on presentation day.
- [ ] 13.4 Rehearse the trained-vs-live two-beat (from §1) as the answer to
  "is this just looking up OFF's data?":
  1. **Credibility beat:** scan a labeled product (Nutella, OFF says
     NOVA 4) — our model independently computes 4 from raw ingredients,
     having never read OFF's stored label.
  2. **Coverage beat:** scan an unlabeled regional snack — the model
     still produces a confident verdict, because it only ever needed the
     raw ingredients. A pure lookup has nothing to say here; we do.

---

## 6. Standing Risks to Watch

- **`nova_group` missing from bulk exports (resolved)** — the standard OFF
  TSV/CSV dump doesn't include computed fields like `nova_group` by
  default; confirmed against OFF's own export docs. Resolved by fetching
  training data from the live search API instead (Stage 1) — if you ever
  swap back to a bulk-export-based approach, re-verify this before
  trusting it, don't assume the column exists.
- **Search API's reported `count` is capped at 10,000 and not exact** —
  fine for this project (we pull 5000/class, under the cap), but don't rely
  on that
  number for anything beyond "there are plenty of rows available."
- **NOVA class imbalance** — check this in Stage 1.1; if one class is nearly
  empty, your model will look "accurate" while being useless on it. Use
  macro-F1, not accuracy, as your headline metric.
- **Live API dependency** — Stage 1's training fetch and Stage 6's demo lookup
  both hit OFF live. Stage 6.1's timeout/fallback isn't
  optional. If the venue's network is bad, a hung request with no timeout
  will kill your live demo. Test on real conference/venue-style wifi if you
  can before the 26th.
- **Multi-runtime Docker image (Stage 8)** — a Python+Node image in one
  container is more fragile than a pure-Python one. Budget extra time here
  specifically, and test the container locally well before Stage 9's CI
  run, so you're not debugging the Dockerfile for the first time inside
  GitHub Actions.
- **Scope creep on Stage 12** — a full production drift-monitoring system is
  not what gets you marks here; a working `/metrics` endpoint and one
  manual KS-test comparison is enough. Don't let this eat Stage 4/9/10 time.
