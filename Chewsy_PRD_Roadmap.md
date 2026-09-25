# Chewsy — Product Requirements Document & Execution Roadmap

**Course:** Feature Engineering & MLOps mini-project
**Submission deadline:** 26 Sept 2026
**Presentation slot:** 26 Sept 2026, 9:00 AM – 2:00 PM
**Today:** 24 Sept 2026 — **~2 days of build time remaining**

---

## 1. Product Overview

**One-liner:** A scanner that answers two separate questions about a
product: **how it's made** (NOVA 1–4, model-computed from ingredients) and
**what's in it** (nutrient traffic lights computed from published NHS
thresholds). You scan or type a real product barcode, the backend
live-fetches it from the Open Food Facts public API, the frozen model
computes the processing class, and the UI shows both axes plus a SHAP
explanation of *why*.

**Design principle (binding — drives Stage 6.7 and Stage 7):**
**NOVA is a descriptor, never a verdict.** The model classifies industrial
formulation, not healthiness. The UI must never present a NOVA class as a
judgement on whether a food is good or bad. Muesli is NOVA 4 and
nutritionally fine — both are true, and the UI must be able to say both.
(Trigger: the muesli scan — model said NOVA 4, no bug found; the bug was
the headline. SHAP showed `additives_n` at −1.62 *arguing against* class 4,
i.e. the model isn't equating "many additives" with "ultra-processed" —
it reads industrial formulation from ingredients text + count.)

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
- Not predicting, computing, or displaying Nutri-Score — OFF's or our own.
  Predicting it would be reverse-engineering arithmetic, not real ML;
  displaying it would put a computed grade next to our model's output and
  muddy the two axes we actually own (traffic lights are published-threshold
  lookups, not a score).
- Not building a meal-logging or calorie-tracking app.
- Not attempting global coverage — training pulls are class-balanced and
  deliberately capped (5000/class in Stage 1, under the API's 10,000/class
  hard ceiling), not a full-corpus crawl;
  India is a demo story, not a training filter (see Stage 1).

**Hard rule 12 (new — sync into AGENTS.md "Hard rules" at next chore):**
**Traffic lights are computed locally from `nutrition_100g`, never
fetched.** The nutrition axis is a deterministic threshold lookup applied
to nutrients already in the response. No OFF-derived grade
(`nutrition_grades`, `nutriscore_*`) may enter the response or the UI —
same exclusion as the feature set (Hard rule 2).

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
│   ├── batch_predict.py             (Stage 6 — offline scoring of a CSV of barcodes)
│   └── nutrition_flags.py           (Stage 6.7 — NHS traffic-light bands + headline rule)
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
  **Count-null taxonomy (25 Sep 2026 — the project's key data finding):**
  the three count columns' null rates are not bad-data noise — they are two
  distinct populations, each verified against the live product API:
  - **~50% lossy pull** (10,054 rows): the search API *never* returns a
    count of 0 — min non-null = **1.0** in all three columns — and **18/18**
    live product-API lookups on null-with-text rows came back `0`. These
    nulls are true zeros the pull dropped; `clean.py` fills them **0 only
    when ingredient text exists** (text-conditioned fill, commit
    `fix: text-conditioned count fill`), never blindly.
  - **20.2% genuine unknowns** (4,034 rows — every no-ingredient-list row):
    null in training *and* in the live API (sparse Diet Coke) → kept NaN;
    imputed later by `GroupMedianImputer`.
  Null rates are class-structured, not random (class1 92.6%, class2 98.4%,
  class3 76.9%, class4 14.0% — class 2 is mostly single-ingredient
  products whose count was never reported); where training *has* a value,
  live agrees **4/4 exact**. `ingredients_n` is null only on no-text rows.
  Full evidence: `docs/data_quality_report.md` §14 (local-only).
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
  **Retrain generations + sparse companion metrics (25 Sep 2026 —
  approved scope addition, this is the project's headline finding):**
  - *Root cause:* no-ingredient-list training rows are
    **{1: 179, 2: 3759, 3: 96, 4: 0}** — the model was never shown an
    ultra-processed product without an ingredient list, so
    `text_was_missing=1` was a learned near-deterministic pull toward
    NOVA 1/2 (sparse Diet Coke: NOVA 1 @ 0.9988 pre-fix). Caveat that
    shaped the fix: class 2's share is *legitimate domain logic* too —
    single-ingredient products (olive oil, sugar) have no list because
    the product IS the ingredient — and training rows all carry category
    tags, so stubs and single-ingredient rows are only separable by
    nutrients/brand/name (measured: 44% of no-text rows have all
    nutrients missing vs ~10% of text rows).
  - *Gen 2* (text-conditioned count fill): run5 = **0.9489**; sparse
    helped but missed the bar (0.575 acc under the legacy sim).
  - *Gen 3* (sparse-view text dropout, approved): `augment_text_dropout`
    appends **full-stub twins** (blank ingredient text + blank
    `categories_tags` + counts NaN + `text_was_missing=1`, nutrients/
    brand kept) of a **stratified** 20% of TRAIN rows — post-split only,
    with a hash tripwire asserting the eval frame is never mutated.
    The **rate is a tracked MLflow param** (`text_dropout_frac`) and was
    **swept as separate runs** (`--sweep 0 0.1 0.2 0.3`, champion config,
    `role=text-dropout-sweep`): baseline F1 / sparse F1 =
    0.9489/0.4135 → 0.9465/0.7540 → 0.9465/0.7726 → 0.9455/0.7754.
    Chose **0.20** (Pareto-dominates 0.10; 0.30 buys +0.003 sparse for
    baseline cost). **Every run now logs both populations**: `f1_macro`
    (complete records) + `sparse_f1_macro` / `sparse_f1_class_3` /
    `sparse_f1_class_4` (stub-shaped eval input).
  - *Final matrix @ drop=0.20:* run5 xgboost **0.9465** (sparse 0.7726,
    c3 0.685, c4 0.696) > run4 0.9432 > run3 0.9394 > run2 0.8749 >
    run6 0.8509 > run1 0.7936 — same ordering as Gen 1. Held-out
    before/after under the SAME full-stub sim: **sparse F1 0.4135 →
    0.7726**; class-4 sparse **0.000 → 0.696**; class-3 0.353 → 0.685;
    baseline cost only −0.0024 (0.9489 → 0.9465, both ≫ the 0.94 bar).
  - *2,000-row flip analysis* (seed 42, mixed train/test rows — label as
    sample stats if quoted): under the **identical legacy sim**, v2 → v4:
    correct→wrong flips **42.1% → 19.4%**, confident-wrong (≥0.9)
    **21.4% → 16.6%**; v4 under the full-stub sim: flips **18.0%**,
    confident-wrong **9.5%**, full-stub acc .8155 / F1 .8163 (v2 on the
    same sim: .5750/.5277 — model rewritten in place, sim comparison
    only valid for the legacy shape; the held-out 0.4135 → 0.7726 above
    is the cross-generation full-stub number). Per-class full-stub v4:
    {1: 0.892, 2: 0.887, 3: 0.686, 4: 0.795} vs baseline {.992/1.000/
    .980/.982}. Full logs: `docs/data_quality_report.md` §14
    (local-only).
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
  **Champion generations (25 Sep 2026):** v1 = Gen-1 run5 (0.9494) →
  v2 = Gen-2 run5 after the count-fill fix (0.9489) → **v4 = Gen-3 run5
  at `text_dropout_frac=0.20` (0.9465 baseline / 0.7726 sparse), current
  `champion` alias**. `register_champion` now filters candidate runs by
  *all three* of: current `params.data_sha256`, selected
  `params.text_dropout_frac`, and `tags.role` ∉ {champion,
  text-dropout-sweep} — so stale-data runs, sweep measurements and
  packaging runs can never win the registry slot. Refit inside register
  replays the same train-only augmentation (deterministic seed), so the
  registered weights match the comparison run exactly.
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
  **Repackaged for Gen 3 (25 Sep 2026):** after the text-dropout retrain,
  `python src/package.py` re-froze the v4 champion — dump **17.7 MB**,
  reference regenerated (19,998 × 20, unchanged shape), roundtrip check
  predict=NOVA 1 on the reference's first row (true NOVA 1, 100% orange
  juice), `dvc add` + `dvc push` re-synced (`dvc status -c`: in sync),
  full suite 110 passed + 1 gated skip (incl. the fresh-process
  registry-vs-joblib equality guard in `tests/test_package.py`).
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
- [x] 6.7 **Scope addition (approved 25 Sep 2026 — built same day with the
  two-axis reframe):**
  extend `PredictResponse` with the facts a scan-result screen needs (data
  is already in hand during scoring — zero extra API calls):
  - `nutrition_100g`: the 8 per-100g values taken from the **normalized**
    row (Stage 2 clean values — capped, invalid → `null`; frontend renders
    `null` as "—", never a raw impossible number),
  - `additives_n`, `ingredients_n` (ints),
  - `ingredients_text` (raw OFF text — what's actually on the package),
  - **`traffic_lights`**: `sugars`, `fat`, `saturated_fat`, `salt` →
    `"low" | "medium" | "high" | null` — `null` when the nutrient is
    missing; never guess a band,
  - **`positives`**: `fiber`, `proteins` → `"good" | "moderate" | "low" |
    null` (same null rule),
  - **`headline`**: the combined one-liner (rule below).
  Must NOT add any OFF label field (Hard rules 11 + 12 unchanged); extend
  `tests/test_api.py` to assert the new fields (e.g. Nutella: energy 539,
  salt 0.107). Feeds Stage 7.8's facts panel and 7.5's hero.

  **Traffic-light thresholds (current NHS/FSA guidance):**

  Solids, per 100g:

  | Nutrient | Low (green) | Medium (amber) | High (red) |
  |---|---|---|---|
  | Sugars | ≤ 5g | > 5–22.5g | > 22.5g |
  | Fat | ≤ 3g | > 3–17.5g | > 17.5g |
  | Saturated fat | ≤ 1.5g | > 1.5–5g | > 5g |
  | Salt | ≤ 0.3g | > 0.3–1.5g | > 1.5g |

  Drinks, per 100ml (NHS published drink variants — exactly half the solid
  values):

  | Nutrient | Low (green) | Medium (amber) | High (red) |
  |---|---|---|---|
  | Sugars | ≤ 2.5g | > 2.5–11.25g | > 11.25g |
  | Fat | ≤ 1.5g | > 1.5–8.75g | > 8.75g |
  | Saturated fat | ≤ 0.75g | > 0.75–2.5g | > 2.5g |
  | Salt | ≤ 0.15g | > 0.15–0.75g | > 0.75g |

  **Documented decisions (all three as code comments):**
  1. These are the **current NHS figures**, not the older 2007 FSA ones
     (which used fat 20g, sugars 15g) — a judge who knows the scheme may
     check which version we used.
  2. **Beverages are handled, not scoped out** (revised decision — small
     build, so worth doing right): a product is a drink iff
     `"en:beverages" in categories_tags` (OFF's `categories_tags`
     includes the ancestor chain; the field already reaches the scoring
     row via `extract_feature_input` → `normalize_live_row`, so this is
     one boolean on data we already have — **zero plumbing**). Pick the
     drinks table when true; missing/empty `categories_tags` → default to
     solids, documented. Code comment notes two caveats: (a) OFF reports
     drink nutrients in the same `_100g` fields though they are
     per-100ml values — fine because beverages have density ≈ 1;
     (b) alcoholic/high-sugar-syrup drinks are the known edge cases and
     are accepted, not special-cased.
  3. Fibre and protein have **no official traffic-light thresholds** (the
     FSA scheme covers only the four nutrients of concern). `positives`
     bands are clearly-labelled **custom** bands (fibre: good ≥6g,
     moderate ≥3g — anchored to EU "source of fibre" claim levels;
     protein: good ≥10g, moderate ≥5g — pragmatic custom cut) and must
     never be called "traffic lights" in the API docs or UI. **Cut
     first if behind schedule** — the four traffic lights carry the whole
     argument alone.

  **New file `src/nutrition_flags.py`** — one home for thresholds,
  headline rule, and band helpers; imported by the API, unit-testable in
  isolation (Hard rule 4's "no inline logic" spirit).

  **Headline rule** — four quadrants from (NOVA ≤2 vs ≥3) × (any red vs
  none):

  | Processing | Nutrients | Headline |
  |---|---|---|
  | Low (1–2) | No reds | "Minimally processed and nutritionally solid" |
  | Low (1–2) | Has reds | "Minimally processed, but high in [nutrient]" |
  | High (3–4) | No reds | "Ultra-processed, but nutritionally decent" |
  | High (3–4) | Has reds | "Ultra-processed and high in [nutrient]" |

  Wording refinement (build-time, deliberate): render "…**Processed**,
  but nutritionally decent" when predicted class = 3 and "…**Ultra**-processed…"
  only for class 4 — class 3 is processed, not ultra-processed, and a
  judge will notice. Red-noun list: `sugars → sugar`, `fat → fat`,
  `saturated_fat → saturated fat`, `salt → salt`.

  **Edge case:** when all four traffic lights are `null` (~17% of training
  rows report no nutrients at all — Stage 2 findings), fall back to a
  processing-only headline and state that nutrition data is unavailable —
  never imply the product is fine.

  **Tests (`tests/test_api.py` additions):**
  - Nutella (sugars ~56.3, salt ~0.107) → sugars `high`, salt `low`,
  - a product with a missing nutrient → key is `null`, not a fabricated
    band,
  - boundaries: exactly 5.0 sugars → `low`; exactly 22.5 → `medium`
    (lock the inclusive/exclusive edges against the table above),
   - beverage band-flip: a drink row with `categories_tags` containing
     `en:beverages` and sugars 4.8 → drink band `medium` (> 2.5) while the
     same value as a solid is `low` (≤ 5); same product with empty
     categories → falls back to the solid table.

  — done: `src/nutrition_flags.py` (both threshold tables + all three
  documented decisions as code comments, `is_beverage`, band helpers,
  custom positives, headline rule incl. class-3 wording + all-null
  fallback). `PredictResponse` gains `nutrition_100g`, `traffic_lights`,
  `positives`, `headline` + `additives_n`/`ingredients_n`/`ingredients_text`
  (raw OFF package text, HTML-unescaped for the facts panel). **Interpretation
  recorded:** "the 8 per-100g values" = the standard UK/EU panel rows —
  energy (kcal), fat, saturated_fat, carbohydrates, sugars, fiber, proteins,
  salt; `sodium_100g` dropped as redundant with salt (salt = sodium × 2.5).
  All fields computed from the `df.iloc[0]` already in `predict()` — zero
  extra OFF calls; Hard rule 12 holds (pure `float` threshold lookups, no
  grade fields touched) and the three Hard rule 11 guards + forbidden-field
  response test pass unchanged over the extended body.
  Tests: the 4 PRD cases in `tests/test_api.py::TestStage67TwoAxisResponse`
  (Nutella sugars `high`/salt `low` + energy 539/salt 0.107, missing →
  `null`, 5.0→`low`/22.5→`medium` edges, beverage flip 4.8 with empty-
  categories fallback) + 45 pure unit tests in `tests/test_nutrition_flags.py`
   (full quadrant table, class-3 wording, all-null fallback, descriptor-not-
   verdict property, solid/drink edge locks, positives bands).
   Suite: **107 passed, 1 gated skip**.
- [x] 6.8 **Scope addition (approved 25 Sep 2026 — sparse-input honesty):**
  `PredictResponse.data_sparse: bool` — true when the OFF record is a
  **stub** (no ingredient text AND no category tags AND at least one count
  unpublished; the measured live Diet Coke shape). Computed by
  `app.main.is_sparse_input` from the **raw extracted row, BEFORE Stage 2
  cleaning** (`build_feature_frame` returns `(frame, data_sparse)`), and
  the frame carries an explicit assert that `data_sparse` is never in the
  feature columns — response metadata only, Hard rule 11 spirit: the flag
  reports *input quality*, it never conditions scoring (the verdict stays
  100% model-computed on every scan; three label guards unchanged).
  `headline` gains the honest fallback when sparse —
  `"Not enough information to classify processing — [nutrient clause]"` —
  so the loudest text on screen refuses to assert a processing level the
  input can't support, while the nutrient axis keeps reporting.
  `batch_predict.py` writes the flag too (`data_sparse` output column).
  Tests: `TestDataSparse` (stub→True, documented→False, tag-only/
  text-only/counts-present→False, sparse scan →200 with model verdict +
  honest headline + `nova_group` absent, frame-column assert) — suite
  **110 passed, 1 gated skip**.

### Stage 7 — Next.js Frontend
**Goal:** a real, camera-driven scanner UI — not a form. This is the layer
that has to feel like a product, so it gets more care than a typical
course-project UI.

**Result-screen design (approved 25 Sep 2026 — brainstormed scope change;
revised same day after the muesli finding — two-axis reframe):**
Structure = **single scrollable result card** (approach chosen over
tabbed/accordion variants: nothing hidden from judges, least interactive
state to break in Stage 8's container). Top → bottom:
1. **Hero** — image, product name, barcode (small mono), **`headline`
   from 6.7 as the largest text** (the card leads with the combined
   two-axis sentence, not the badge), confidence chip with tier word:
   ≥0.8 "Confident" · 0.6–0.8 "Fairly sure" · <0.6 **or** top-2 gap
   <15 pts → "**Borderline**" (honest for cheese-stick-type 0.87/0.12
   results).
2. **Two-axis row** — side by side: "How it's made: NOVA 4 ·
   Ultra-processed" | "What's in it: 4 nutrient chips" (green/amber/red
   traffic lights from 6.7's `traffic_lights`).
3. **Probability distribution** — 4 bars, each labeled with class name +
   %, predicted class highlighted; makes model ambiguity *visible*.
4. **Why — SHAP chart** (`recharts`, PRD 7.5 baseline) — top-5, sign =
   toward/away from verdict; humanized labels (`num__additives_n` →
   "Number of additives", SVD components → "learned ingredient-text
   signal").
5. **Facts panel** — per-100g nutrition grid + additive/ingredient counts
   + full ingredient list (needs 6.7).
6. **NOVA explainer** — **promoted from collapsed** (now always visible
   after the axes row), with the line: *"NOVA describes how a food is
   made, not how nutritious it is."* The descriptor-not-verdict principle
   (§1) stated on the card itself.
7. Footer microcopy: *"Chewsy model verdict — computed from ingredients,
   not copied from OFF"* (pitch credibility).

**Colour change (binding):** the NOVA badge **stops being green→red** —
red reads as "bad," which is the exact conflation being fixed. Use a
neutral ramp (light→dark blue, or grey→charcoal) for processing level.
Green/amber/red is reserved **exclusively** for nutrient chips, where it
matches the published FSA meaning.

Plus: session **scan-history chips** (last 10) on the scanner screen and a
"Scan another" reset. Scope tiers chosen: A (existing response fields) +
B (frontend-only) + C (6.7 response extension). **Rejected:** tabbed
layout, showing OFF's `nova_group`/Nutri-Score (Hard rules 11 + 12),
anything requiring a second backend fetch.

- [x] 7.1 Scaffold with `create-next-app` (TypeScript, App Router) inside
  `frontend/`. (Next 16.3.6 + React 19 + Tailwind 4; `shadcn/ui` init'd
  on top as the agreed component base — deviation from bare Tailwind,
  approved 25 Sep 2026.)
- [x] 7.2 `components/BarcodeScanner.tsx`: use the browser's camera via a
  client-side barcode-decoding library — **`html5-qrcode`** is the easiest
  fit (handles camera permission prompts and supports EAN-13/UPC, the
  formats real grocery barcodes use, not just QR codes). Decoding happens
  entirely in the browser; only the decoded barcode *string* gets sent to
  the backend.

  **Mechanism (record for the PRD's own sake):** decoding reads **bar
  widths, not printed digits** — it is not OCR and needs no text
  extraction. EAN-13's 13th digit is a checksum recomputed by the decoder,
  so a decode either succeeds correctly or emits nothing — no invalid
  barcode can reach the API. The only thing sent to the backend is a digit
  string, which the existing `^\d{6,14}$` validator already handles —
  **backend needs zero changes.** Comment this in the component (checksum
  validation makes decodes self-validating — no extra client-side barcode
  validation needed).
  **Built:** `components/BarcodeScanner.tsx` — dynamic-import of
  `html5-qrcode` (SSR-safe), format whitelist enforced via
  `formatsToSupport`, `facingMode: environment`, `qrbox 240×160`,
  `fps 10`, stream stopped before `onDecode` fires; mechanism comment
  recorded in-file.

  **Config specifics:**
  - Restrict formats to `EAN_13`, `UPC_A`, `EAN_8` — scanning all formats
    every frame is measurably slower; these three cover Indian, European
    and US packaging.
  - `facingMode: 'environment'` for the rear camera.
  - Stop the camera stream on successful decode — otherwise it keeps
    firing and double-submits.

  **Keep `html5-qrcode`. Do NOT switch to the native `BarcodeDetector`
  API:** Safari/iOS has it disabled by default even currently, Firefox
  doesn't support it, and Chrome desktop ships it only on macOS/ChromeOS —
  not Windows or Linux.
- [x] 7.3 Keep a manual text-entry fallback input alongside the camera view
  (not a compromise — every real scanner app has this, for when lighting
  or focus fails mid-demo).
- [x] 7.4 On decode/submit: `fetch(POST, '<API_URL>/predict', { barcode })`.
  **This frontend must call the FastAPI endpoint over HTTP — it must never
  reimplement scoring logic or duplicate the model on the frontend.** Same
  "one source of truth" principle the mindmap describes for Streamlit,
  just carried over to this stack.
  (Built: `frontend/lib/api.ts` — the UI's only data path; digits are
  stripped on input and the client pattern mirrors the API's
  `^\d{6,14}$`; no scoring/band/headline math anywhere in `frontend/`.)
- [x] 7.5 Result view (single scroll card per the approved design above):
  product name/image from the API response, **`headline` as the largest
  text**, two-axis row ("How it's made" NOVA badge on a **neutral
  light→dark ramp — never green→red** · "What's in it" 4 nutrient chips
  green/amber/red — colours reserved for traffic lights only), confidence
  chip with tier word (Confident / Fairly sure / Borderline), and the
  SHAP top-features rendered as a simple bar chart (e.g. `recharts`) with
  humanized feature labels.
  (Built: `components/ResultCard.tsx` — headline renders as the largest
  text (`text-3xl`); NOVA badge uses the neutral `--nova-*` ramp
  (e2e-computed bg `lab(21.1 …)` = charcoal, never green/red); G/A/R
  appears only on the four nutrient chips with band words + icons;
  confidence tiers Confident/Fairly sure/Borderline incl. the top-2-gap
  <15pts rule; `ShapChart.tsx` (recharts) colors toward=honey/away=blue
  with humanized labels.)
- [x] 7.6 `NEXT_PUBLIC_API_URL` as an env var, not a hardcoded localhost URL
  — you'll need this to differ between local dev and the Docker/EC2 build.
  (Committed `.env.local` → gitignored by `frontend/.gitignore`'s `.env*`;
  value is read at build time — Stage 8 passes it during `next build`.)
- [x] 7.7 `components/ProbabilityBars.tsx`: the 4-class distribution —
  every bar labeled with NOVA class name + %, predicted class highlighted;
  this is what makes model ambiguity visible (e.g. cheese stick
  0.87 / 0.12 instead of a silent single number).
  (Built: `components/ProbabilityBars.tsx` — 4 rows, class name + % label,
  predicted row highlighted on the same neutral ramp with a `predicted`
  tag.)
- [x] 7.8 Facts panel: per-100g nutrition grid (`null` → "—"), additive +
  ingredient counts, full ingredient list — **depends on 6.7** (do 6.7
  first; it is a backend change with its own tests).
  (Built in `ResultCard.tsx`: 2-col dl grid, `null` → `—`, counts line
  `N additives · N ingredients`, full ingredient text block; 6.7
  `positives` render as muted OUTLINE chips on the Fibre/Protein rows
  with the "Chewsy's own bands — not official traffic lights" caption.)
- [x] 7.9 Scan flow extras: NOVA explainer (per revised design: **always
  visible**, not collapsed — carries the "how it's made ≠ how nutritious
  it is" line), session scan-history chips (last 10: "Nutella → 4"),
  "Scan another" reset.
  (Built: explainer section always rendered with the exact binding line;
  `page.tsx` keeps a session `history` state — last 10, re-scan chips
  `name → NOVA`, `Scan another` returns to the scan screen.)
- [x] 7.10 Error/edge states + typing: 404 → "Product not found in OFF",
  503/network → "OFF unreachable — retry" with retry button, missing
  image → placeholder; `types/predict.ts` mirrors the Pydantic response
  schema exactly. **Camera errors:** `navigator.mediaDevices` undefined
  (insecure context) → manual entry + explicit "camera requires HTTPS"
  message, not a broken viewfinder; permission denied → manual entry
  emphasized; no camera device found → manual entry.
  (Built: `ScanError` kinds not_found/unavailable/invalid/unknown with the
  PRD copy verbatim + Retry button on 503/network; `types/predict.ts`
  mirrors `app/schemas.py` field-for-field; camera gates by `name` of the
  DOMException — insecure/permission/nodevice each render an Alert and
  keep manual entry live; missing image → `ImageOff` placeholder.)
- [ ] 7.11 Verification: `npm run typecheck` + `npm run build` green, one
  manual camera e2e against the local API, and 6.7's extended pytest
  suite green (no frontend test suite required by the PRD).
  **Status 25 Sep 2026:** typecheck ✅ lint ✅ build ✅ (static `/`),
  pytest ✅ 110 passed + 1 gated skip, plus a 33-check automated
  headless-Chrome e2e (scan → result card → history → 404 error → reset
  → sparse stub scan showing the forced low-information state, zero
  unexpected console/network errors). **Remaining: the manual camera
  e2e — needs a physical camera + human at `localhost:3000` with the
  API on :8000.**
- [x] 7.12 Commit: `feat: Next.js scanner UI` (this commit).
- [x] 7.13 **Scope addition (approved 25 Sep 2026 — sparse result state,
  Fix 2 frontend half):** when the API returns `data_sparse: true` the
  result card never shows a confident processing verdict:
  1. **Processing axis** → the NOVA badge is replaced by a dashed
     `"Not enough information to classify"` state (mirrors
     `nutrition_flags.PROCESSING_UNKNOWN` — one string, not two);
  2. **Tier chip** → forced to the existing `{word:"Low information",
     variant:"dashed"}` treatment **regardless of the raw probability**
     — the percentage is hidden from the chip (the real number stays in
     the API response and in `class_probabilities` for transparency,
     per the approved contract);
  3. **Nutrition axis still renders** — traffic-light chips are computed
     from the nutrient half of the record, which Diet Coke's stub
     actually has (all-zero values are *real* for a diet drink), so the
     two-axis promise survives the thin input;
  4. probability caption switches to `"4-class distribution · thin
     input"`.
  `types/predict.ts` mirrors the field; `DESIGN.md` Do/Don't line
  added; the capability is demonstrated live (stub 5000112644906 →
  NOVA 1 @ 0.81 post-fix with the honest framing — headline owns the
  gap, not the badge).
- [x] 7.14 **Layman copy pass (user review 25 Sep 2026 — "too techy
  language"):** on-screen text no longer names internals.
  Result card: "How sure is the model?" → **"How sure are we?"**
  (caption "all four levels" / "all four levels · little to go on" —
  supersedes 7.13's "4-class distribution" caption wording),
  "Why — what the model read" → **"Why we say this"** (caption "top 5
  reasons for NOVA n"), "predicted" chip → **"our call"**,
  "descriptor, not verdict" → **"not a health score"**, footer →
  **"Chewsy's own verdict — worked out from this product's ingredients
  and nutrients"**, tier chip one
  decimal (**"Confident · 99.9%"** — was rounding 99.9 → "100%" next to
  a 99.9% bar). SHAP chart footnote → "Honey bars push toward this
  verdict… (Shown with SHAP, the standard way to explain a decision.)"
  — the single SHAP mention kept for the course audience; tooltip
  "SHAP value" → "push". Labels: SVD "Ingredient-text signal #21" →
  **"Ingredient wording pattern #21"**; brand/category frequency labels
  drop "training data". Loading screen drops "scoring with the frozen
  model" → "fetching the label, then working out how it's made…".
  Errors (title + detail): "OFF unreachable — retry" → **"Couldn't
  reach the food database"**, "API unreachable — check the backend is
  running." → **"We couldn't connect. Check your connection and try
  again."**, "Product not found in OFF" → "Product not found" (+ detail
  "We couldn't find this barcode — try another product.").
  Follow-up rule (user, 25 Sep): **no "Open Food Facts"/"OFF" anywhere
  in the UI at all** — site footer ("Chewsy works out how each product
  is made — a description of processing, not a health score."), meta
  description, SHAP caption, card footer, and error copy all scrubbed;
  only an internal comment/README mention may name the source.
  typecheck/lint/build green.

### Stage 8 — Containerization
**Goal:** one image, both a Python and a Node runtime inside it — this is
the one place the stack swap actually adds real complexity, budget extra
time here.

- [x] 8.1 **Multi-stage `Dockerfile`:**
  - Stage A (`node:20-slim`): `COPY frontend/`, `npm install`, `npm run build`
    → produces a production Next.js build.
  - Stage B (`python:3.11-slim`): install Python deps (`COPY
    requirements.txt` before app code, for layer caching), **also install
    Node.js runtime** (needed to run `next start`, not just to build),
    `COPY` the built frontend output from Stage A, `COPY` the FastAPI app.
  ✅ Built as specified, with three recorded decisions:
  1. **Base is `python:3.12-slim`, not 3.11** — evidence: pip build
     failed on 3.11 with *"shap==0.52.0 Requires-Python >=3.12"* (the
     Stage 0.3 pin); host Python is 3.12 too. Node 20 runtime via
     NodeSource apt repo (Debian's own nodejs is 18, below Next 16's
     floor); Stage A also runs `npm prune --omit=dev` after build so
     only runtime deps ship.
  2. **Model baked into the image** (`COPY models/model.joblib` —
     approved 25 Sep 2026): self-contained demo image; fresh clones and
     CI must `dvc pull` **before** `docker build` (comment at top of the
     Dockerfile says so; Stage 9's workflow does exactly this).
  3. **`NEXT_PUBLIC_API_URL=http://localhost:8000` as a build ARG** —
     Next inlines `NEXT_PUBLIC_*` at build time; the browser reaches the
     API on the host-mapped port. Plus a new **`.dockerignore`**
     (excludes `.venv/`, `mlruns/` 77 MB, `data/`, `docs/`, git,
     `frontend/node_modules`, `.next`) — without it the build context
     would ship the whole ML history.
- [x] 8.2 `entrypoint.sh`: start `uvicorn` on `:8000`, poll `/health` until
  it responds, then start `next start` on `:3000`. Both must stop together
  on `SIGTERM`.
  ✅ PID-1 bash script: `trap shutdown TERM INT` kills uvicorn **and**
  `node …/next start` together; health poll via `curl` (installed in
  Stage B) retries 60×1s and exits non-zero if the API never comes up;
  `wait -n` supervises — if either child dies on its own, the other is
  torn down too.
- [x] 8.3 Local test: `docker build` + `docker run`, hit both ports from the
  host.
  **Done when:** a barcode scanned through the containerized frontend
  returns a real prediction end-to-end, not just "container starts."
  ✅ Runtime: **Colima** (`colima start --cpu 4 --memory 4 --disk 40`,
  macOS Virtualization.Framework, docker 29.5.2 arm64) — approved choice
  over Docker Desktop (lighter, CLI-only, fits 8 GB host). Evidence:
  `docker build -t chewsy .` succeeds (frontend stage cached on
  rebuild); `docker run -p 8000:8000 -p 3000:3000` → `/health`
  `{"status":"ok","model_loaded":true}` (44 features, explainer ready),
  `GET :3000/` 200, live `POST /predict` on the sparse stub →
  NOVA 1 @ 0.81 with `data_sparse:true`. **Done-when met:** the full
  33-check headless-Chrome harness run against the *containerized*
  frontend (local dev servers stopped first, ports 8000/3000 owned by
  the container) → **33/33 PASS**, zero unexpected console/network
  errors — scans through the UI hit the container's own API and return
  real predictions (incl. the sparse forced-tier path). SIGTERM:
  `docker stop` → exit 0 in <1s, uvicorn graceful shutdown in logs,
  `docker start` recovers healthy.
- [x] 8.4 Commit: `feat: Docker containerization`.
  ✅ Committed with `Dockerfile`, `entrypoint.sh`, `.dockerignore` + this note.

### Stage 9 — CI/CD to Docker Hub
- [x] 9.1 `.github/workflows/ci.yml`: on every push — run `pytest`, then (only
  if tests pass) `docker build` + `docker push` to Docker Hub using repo
  secrets for credentials.
  **Done when:** a push to `main` results in a new tagged image visible on
  Docker Hub, triggered by GitHub Actions — not a manual `docker push` from
  your laptop.
  ✅ As-built: three jobs — `test` (setup-python **3.12** to match the
  shap pin, pip cache, `dvc pull`, `pytest -q` — every push), `frontend`
  (`npm ci` → typecheck → lint → build — every push; **enhancement**,
  the PRD only names pytest and the frontend had zero CI otherwise), and
  `docker` (`needs: test` = "only if tests pass"; additionally gated to
  `push` + `refs/heads/main` so feature pushes never touch Docker Hub:
  `dvc pull` → `docker login` → build → push `:latest` **and**
  `:sha-<7>`). **Secrets** (repo, set 25 Sep 2026 via `gh`):
  `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` (the HF S3 pair from local
  `.dvc/config.local` — values never echoed) + `DOCKERHUB_USERNAME`/
  `DOCKERHUB_TOKEN` (PAT validated with a local `docker login` first);
  the two `AWS_*_CHECKSUM` env vars mirror the local push/pull
  requirement (botocore trailing-CRC32 vs the HF gateway).
  **The first runs found two fresh-checkout portability bugs — both
  fixed, both also break any clone on any new machine (not just CI):**
  1. `test` failed `test_package.py` with `MlflowException: No such
     artifact: ''` — `mlflow.db` stored **absolute paths from this
     laptop** (`model_versions.storage_location`, `runs.artifact_uri`,
     `experiments.artifact_location` all `/Users/prem…/mlruns/…`), and
     the registry resolves `models:/chewsy-nova@champion` through
     `storage_location`. Fix: all three columns relativized to
     `mlruns/…` (cwd-relative; pytest/registry load run from repo
     root). Proven by moving the local `mlruns/` away entirely and
     still loading registry == joblib from a clone. Commit
     `fix: relativize mlflow registry/run artifact paths…` (`3da0c3f`).
  2. `frontend` failed typecheck: `Cannot find name 'LayoutProps'` —
     Next generates that global into `.next/types/`, absent on a fresh
     checkout and masked locally by build caches. Fix: `typecheck` =
     `next typegen && tsc --noEmit` (verified with `.next/` deleted).
     Commit `fix: typecheck generates route types first…` (`0946763`).
  **Done-when evidence (run 36141265005):** `test: success`,
  `frontend: success`, `docker: success`; image pulled back from Docker
  Hub: `premthatikonda2903/chewsy:latest` digest
  `sha256:730ae35f…` (3.87 GB — Python+Node+model baked; slimming is
  optional future work, not a rubric item).
- [x] 9.2 Commit: `ci: GitHub Actions build + push to Docker Hub`.
  ✅ Committed as that message (`a669f50`) + the two CI-found fix
  commits `3da0c3f`, `0946763`; first fully-green run: `36141265005`.

### Stage 10 — Wire Up the DVC Pipeline
**Goal:** the whole thing becomes one re-runnable, versioned pipeline —
this is the step that makes Stages 1–5 legible as a *pipeline*, not just a
sequence of scripts you happened to run in order.

- [x] 10.1 `dvc.yaml` with stages: `fetch` → `clean` → `features` → `train`,
  each with explicit `deps` and `outs` pointing at the real file paths from
  Section 4.
- [x] 10.2 `dvc repro` runs the whole thing end-to-end from the tracked raw
  training set to a fresh `model.joblib`.
  **Done when:** `dvc repro` with no changes reports everything up to date;
  changing one parameter (e.g. `TARGET_PER_CLASS` or an SVD component count) and
  re-running only re-executes the affected downstream stages.
- [x] 10.3 Commit: `feat: DVC pipeline (dvc.yaml + dvc.lock)`.

**Stage 10 as-built (25 Sep 2026):**
- **10.1:** four stages with only honest deps/outs — `fetch`
  (`deps`: fetch script + `src/params.py`; `param` `fetch.target_per_class`;
  `out` raw training CSV) → `clean` (`deps` clean.py + raw CSV; `out`
  processed CSV) → `features` (evidence cmd, `deps` incl. processed CSV,
  `param` `features.n_svd_components`, **no outs by design** — its product
  is the printed Stage 3 evidence) → `train`
  (`python src/train.py --runs 1 2 3 4 5 6 --register && python src/package.py`;
  params `n_svd_components` + `train.text_dropout_frac`; `out`
  `models/model.joblib`). `mlflow.db`, `mlruns/`, `models/training_reference.csv`
  stay **git-managed** — grading evidence, never DVC outs. Static `.dvc`
  pointers removed first per the Stage 10 handoff note (files untouched;
  backup copy taken; `dvc status -c`: cache + HF remote in sync).
  **Deviations, all measured:** (a) params must be *read* — new
  `params.yaml` (3 keys) + `src/params.py`; wired into fetch
  (`TARGET_PER_CLASS`), `features.__main__` (SVD count), `train.py`
  (`AUGMENT_TEXT_DROPOUT` + `build_full_pipeline(n_svd_components=…)`),
  defaults identical to committed values so direct runs and tests are
  unchanged; PyYAML 6.0.3 pinned (now imported directly). (b) **DVC 3.67
  rejects flat `params: [params.yaml:key]` strings** (stores the whole
  entry as one key → MissingParamsError) — dict form `- params.yaml:\n
  - fetch.target_per_class` with **dotted paths** works; validated in a
  scratch repo before touching the real one. (c) `dvc.lock` **seeded via
  `dvc commit -f <stage>`** from existing workspace state — no live
  refetch, no retrain; `tests/test_package.py` DVC-tracking test rewritten
  to the Stage 10 contract (pipeline out + lock entry; pointer must be gone).
- **10.2 (verified both halves):** first `dvc repro` → all four stages
  `didn't change, skipping` → *Data and pipelines are up to date*.
  Param demo: `n_svd_components 25 → 24` → repro **skipped `fetch` +
  `clean`, re-ran `features` + `train` only** (83 s total; run5 xgboost
  f1_macro **0.9457 at 24 vs 0.9465 at 25** — the param provably reaches
  the model; lock diff shows `features.n_svd_components: 24` and a new
  model md5). Register re-picked the best drop=0.20 run → v5, then state
  restored byte-identical: `git checkout` params/dvc.lock/mlflow.db/mlruns
  + `dvc checkout models/model.joblib` → md5 `5ef31f46…`, 17,669,223 bytes
  (= original), champion alias back to **v4**, `git status` clean,
  `dvc repro` all-skip, pytest **110 passed + 1 skipped**. Demo log:
  `/tmp/s10_repro_demo.log`.
- **10.3:** `db64c1c feat: DVC pipeline (dvc.yaml + dvc.lock)`.
  **Operational notes:** registering from this laptop re-introduces
  absolute mlflow paths — rerun the 3da0c3f relativize UPDATE before
  pushing if a train/register ever ships; restore after any demo run is
  git-checkout + `dvc checkout` (both shown above), never `dvc add`.

### Stage 11 — Cloud Deployment (bonus, only if on schedule)
- [x] 11.1 ~~Launch EC2 instance, open ports 22/8000/3000 in the security
  group.~~ **Adapted (no AWS/GCP account): API on Render free** —
  deployed from the **CI-pushed Docker Hub image** (blueprint
  `render.yaml`, `runtime: image` + registry credential), model baked in,
  Render never runs DVC. Single exposed port per Render service →
  `entrypoint.sh` gained `APP_MODE=api` (uvicorn only, on `$PORT`);
  default both-mode unchanged (verified locally: api + both, SIGTERM
  exit 0 in <0.4s). **First-deploy lesson:** creating the service from
  the *GitHub repo* runs `docker build` in a checkout without
  `model.joblib` (gitignored/DVC-only) → BuildKit
  `"/models/model.joblib": not found`. The repo is deliberately not
  self-buildable — the Blueprint/image-pull path is the only correct one.
- [x] 11.2 `docker run --restart unless-stopped` pulling the Docker Hub
  image. **Adapted done-when met:** UI reachable from a browser at
  **https://chewsy-scanner.vercel.app** (public URL, not
  localhost) → API **https://chewsy-api.onrender.com**.
  Verified server-side: `/health` `{"status":"ok","model_loaded":true}`;
  preflight 200 with
  `access-control-allow-origin: https://chewsy-scanner.vercel.app`
  (CORS env `FRONTEND_ORIGINS` set via render.yaml push + Blueprint
  sync + manual redeploy — the app reads origins once at startup, so an
  env change needs a restart); live `POST /predict` from Render's
  network → Nutella `3017620422003` NOVA 4 @0.9989, Coca-Cola
  `5449000000996` 200 + CORS header; frontend serves the real app
  (`<title>Chewsy — scan a barcode…</title>`).
- [x] 11.3 **Decision recorded: HTTPS via Vercel** — the UI on
  `*.vercel.app` is a secure context, so `getUserMedia`/html5-qrcode
  **camera scanning works on the deployed build** (the EC2-HTTP pitfall
  never applies; no tunnel needed). Camera is therefore demoed from the
  public Vercel URL directly, not from localhost. **Caveat:** Render
  free instances **spin down after 15 min idle** → ~30–60s cold start
  on the first scan after idle (health poll covers model load;
  keep-alive ping optional if rehearsal shows it's annoying).

### Stage 12 — Lightweight Monitoring (trim to this if short on time)
**Stage 2/3 handoff:** `training_reference.csv` should snapshot the
**input** space — the 22 cleaned columns (what a live API row looks like
after Stage 2 normalization), not the 44 transformed features — drift is
about what *arrives*, and live rows start in that space too. It is built
from the **NaN-preserving** clean CSV, so the KS comparison should also
compare per-column **null fractions** (a documentation-drift signal is a
null-rate change: OFF pages that stop recording fiber show up as a rising
`fiber_100g` NaN share before any value distribution moves).
- [x] 12.1 At minimum: structured request logging in FastAPI (barcode,
  latency, predicted class) and a working `/metrics` endpoint.
  **Done 25 Sep 2026 (TDD — 4 tests first, watched red):** one JSON line
  per `/predict` on stdout via logger `chewsy` (`app/main.py`, emitted
  from the handler's `finally` so error paths log too with
  `predicted_nova: null`):
  `{"event":"predict","barcode":"3017620422003","latency_ms":692.9,"status":200,"predicted_nova":4,"data_sparse":false}` —
  verified live on uvicorn (stdout, not stderr: `basicConfig(stream=sys.stdout)`
  guard because uvicorn never attaches a root handler, so INFO records
  would otherwise be dropped by the last-resort handler; `propagate`
  stays on so pytest `caplog` captures the same lines — Render's log
  drain shows them as-is). `/metrics` extended beyond the minimum with
  `per_class: {"1":n..}` + `sparse` counters (`MetricsResponse` gained
  both fields) so the endpoint backs the monitoring slide:
  `{"requests":1,"errors":0,"avg_latency_ms":692.88,"per_class":{"1":0,"2":0,"3":0,"4":1},"sparse":0}`.
  Tests: `TestStructuredRequestLogging` (success line fields, 404 line
  with null class, per-class increment, sparse increment on stub).
  Suite 114 passed + 1 gated skip (was 110+1).
- [x] 12.2 If time allows: a small script comparing live-fetched nutrient
  distributions against `training_reference.csv` using
  `scipy.stats.ks_2samp` — even a single manual run of this, shown as a
  plot in your slides, is enough to demonstrate the concept without a
  full always-on drift job.
  **Done 25 Sep 2026 (TDD — 14 tests, red first):** `src/drift_check.py`
  (`--n 100` random sweep / `--barcodes file.csv` deterministic re-run /
  `--ref`/`--out`): live rows travel the exact serving path
  (`off_client.fetch_product` → `build_feature_frame` → compare in the
  reference's 20-col input space), KS on non-null values for numeric
  nutrient/count columns + **null-fraction delta on every shared column**
  (the Stage 12 handoff's documentation-drift signal), indicators
  (`*_was_missing`) get null-delta only, text null-delta counts `""` as
  missing (serving rows carry `""` where training has NaN — first run's
  spurious −0.202 pseudo-text shift was this artifact). Sort worst-first,
  threshold line 0.15, plot `docs/drift_report.png` (committed — slides
  asset). **Live run (25 Sep, 100/100 rows, 0 skips):** worst KS
  `energy_100g` 0.302 (p=7e-07), `fiber_100g` 0.279 (p=0.018), 4/12 cols
  over threshold; biggest null shift `fiber_100g +0.185 (53% → 71%)` —
  the PRD's own fiber example, live. Live-run findings baked back in as
  tests: search API 400s without a `q` param (empty string = broad
  sweep); product API 429s under sweep → `collect_live` now paces 0.3s
  and backs off once on 429 (first run dropped 80/100, rerun 0/100).
  Suite 128 passed + 1 gated skip (was 114+1).
### Stage 13 — Presentation Prep
- [ ] 13.1 Slides: problem (Yuka framing) → live demo → brief technical
  depth (pick 2–3 "why this, not that" moments to go deep on, don't try to
  narrate every stage above) → MLOps pipeline diagram → close.
  **Add the muesli beat:** the model said NOVA 4 on a product that sounds
  healthy; investigation found no bug — *the bug was the headline*, so the
  product changed (two axes). Cite our own SHAP output as evidence of
  digging in: `additives_n` at **−1.62 arguing against class 4** — the
  model isn't counting additives, it's reading industrial formulation.
  **Add the sparse-view beat (the honest-metrics slide, content ready
  25 Sep 2026):** "Our model scores **0.95 macro-F1 on complete
  records**… and on thin records — no ingredient list, no category tags —
  the original model scored **0.41** on the exact same held-out rows."
  Then the investigation → fix arc: 76% of live OFF records are thin
  (null taxonomy §14.1), training contained **{1:179, 2:3759, 3:96, 4:0}**
  no-text rows so blank input pulled to class 1/2, fix = stratified
  full-stub augmentation swept 0/10/20/30% as tracked MLflow params →
  final: **0.95 complete / 0.77 thin** (class-4 thin: 0.00 → 0.70),
  cost = −0.002 on complete records. The 0.41 → 0.77 delta *is* the
  story; never present 0.95 alone. Both numbers come from the same
  eval split and the same simulator — say so.
- [ ] 13.2 Rehearse the live demo with **3 real barcodes** picked in
  advance, chosen to cover the headline quadrants:
  1. **ultra-processed but nutritionally decent** — muesli,
  2. **ultra-processed and red** — candy bar,
  3. **minimally processed but red** — butter or olive oil (this one
     proves the two axes are genuinely independent).
  Also rehearse the camera on the **exact device and URL you'll present
  from** — localhost and deployed behave differently (see 11.3).
  **Verified barcodes (25 Sep 2026, live recheck on the v4 champion):**
  Nutella `3017620422003` → NOVA 4 @ 0.999 (credibility beat, OFF-labeled);
  Coca-Cola `5449000000996` → NOVA 4 @ 0.999 (complete entry);
  Red Bull `9002490100070` → NOVA 4 @ 0.993 / Sting `8902080000227` →
  NOVA 4 @ 1.000 (energy drink — the headline-complaint fix, they
  classify correctly). Quadrant picks 1–3 still to be locked in by
  eye.
  **Deliberate 4th scan (approved): Diet Coke `5000112644906`** — a
  real stub record (all-zero nutrients, no ingredients, no tags).
  Rehearse it as the sparse-honesty beat: response carries
  `data_sparse: true`, UI shows "not enough information to classify"
  while nutrition chips stay green, and narrate the 0.41 → 0.77 fix
  (13.1 beat) — then optionally show `data_sparse: false` on the same
  product's fuller record if OFF grows one (name+brand record matching
  is the future-work slide: when a stub barcode and a documented
  record exist for the same product, a `name+brand` lookup could join
  them — not built, timeboxed out).
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
