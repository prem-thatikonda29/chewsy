# Multi-Tool Agent Config Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create shared agent instructions (`AGENTS.md`) plus thin shims so opencode, Claude Code, and Antigravity all load NutriScan project context from one source.

**Architecture:** One fat `AGENTS.md` at repo root is the single source of truth (distilled from `NutriScan_PRD_Roadmap.md`). `CLAUDE.md` is a ~5-line pointer for Claude Code. `opencode.json` explicitly registers `AGENTS.md` via the `instructions` array. No custom agents/skills/MCP.

**Tech Stack:** Markdown instruction files, `opencode.json` (JSON, schema `https://opencode.ai/config.json`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-24-agent-config-design.md`
- Source PRD: `NutriScan_PRD_Roadmap.md` — do not modify it
- `AGENTS.md` is the only file with project content; `CLAUDE.md` and `opencode.json` must not duplicate it
- `opencode.json` may only use schema-backed fields: `$schema`, `instructions`
- Every "Hard rule" in the spec must appear in `AGENTS.md` and be traceable to the PRD
- Deadline context: 26 Sept 2026; today in project context: 24 Sept 2026
- Git is not yet initialized (PRD Stage 0 does that later) — no commit steps in this plan; files land on disk and get included in Stage 0's first commit
- No code, tests, Docker, or CI in this plan — those are PRD stages

---

### Task 1: Write `AGENTS.md` (shared source of truth)

**Files:**
- Create: `AGENTS.md`

**Interfaces:**
- Consumes: `NutriScan_PRD_Roadmap.md` sections 1, 3, 4, 5 (stages), 6
- Produces: root `AGENTS.md` — read by opencode, Antigravity, and (via pointer) Claude Code

- [ ] **Step 1: Create `AGENTS.md` with this exact content**

```markdown
# NutriScan — Agent Instructions

## Project overview

Yuka-style barcode scanner for a Feature Engineering & MLOps course mini-project.
User enters a real product barcode → backend live-fetches it from the Open Food
Facts public API → a trained model predicts **NOVA processing level (1–4)** →
UI shows the verdict plus a SHAP explanation of *why*.

**Pitch framing:** Open Food Facts is crowdsourced; most products have no NOVA
label. This tool fills in a classification the community hasn't labeled yet —
the same job Yuka does commercially. Do not frame work as "I analyzed a dataset."

**NOVA classes:**
1 = unprocessed/minimally processed · 2 = processed culinary ingredients ·
3 = processed foods · 4 = ultra-processed foods

**Non-goals (state if asked, don't apologize):** not predicting Nutri-Score;
not a meal-logging/calorie app; not global coverage (scope is deliberately narrow).

**Deadline:** submission + presentation 26 Sept 2026. ~2 days of build time as
of 24 Sept 2026. Treat PRD "MUST" stages as non-negotiable.

**Full stage detail:** `NutriScan_PRD_Roadmap.md` — re-read the relevant stage
section before implementing it. Each stage has a "Done when" condition; do not
advance until it is verifiably true.

## Hard rules (never violate)

1. **Headline metric is macro-F1, not accuracy** — NOVA classes are imbalanced;
   accuracy alone is misleading.
2. **`nutrition_grade_fr` is never a feature** — it's a computed label sitting
   next to the target (leakage), not a legitimate predictor.
3. **Streamlit must call FastAPI `/predict` over HTTP** — never import or load
   `model.joblib` in the UI process. Two processes loading the model independently
   can silently disagree ("a client, not a copy").
4. **Custom transformer classes live in `src/features.py`** — an importable
   module, never inline in a notebook. Avoids the pickling "can't get attribute
   FeatureCreator" bug.
5. **Open Food Facts API calls need a timeout (~5s) and clean 404/503 fallbacks**
   — never let a hung request kill the live demo.
6. **If behind schedule:** cut Stage 11 (cloud deploy) first, then trim Stage 12
   to a single `/metrics` endpoint with no live drift job. **Never cut Stages 4,
   5, 6, 9, or 10** — the rubric explicitly names them (MLflow, packaging,
   FastAPI, CI/CD, DVC pipeline).
7. **Verify OFF export column names** against the actual TSV header before
   trusting the slice script (`pd.read_csv(path, sep='\t', nrows=0).columns.tolist()`).
8. **Stage-by-stage commits** with the PRD's message style (`chore:`, `feat:`,
   `ci:`) — real commit history is graded; no giant dump commits.

## Target repo structure

Create exactly this in Stage 0 (with `.gitkeep` where needed); don't improvise later:

```
nutriscan/
├── .gitignore
├── README.md
├── dvc.yaml
├── dvc.lock                  (generated)
├── requirements.txt
├── data/
│   ├── raw/                  (DVC-tracked, not git-tracked)
│   └── processed/            (DVC-tracked)
├── src/
│   ├── slice_openfoodfacts.py
│   ├── clean.py
│   ├── features.py
│   └── train.py
├── models/
│   ├── model.joblib          (gitignored — DVC/artifact)
│   └── training_reference.csv
├── app/
│   ├── main.py               (FastAPI)
│   ├── off_client.py         (Open Food Facts client)
│   └── schemas.py            (Pydantic)
├── streamlit_app.py
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── Dockerfile
├── entrypoint.sh
└── .github/workflows/ci.yml
```

`.gitignore` must exclude: `.venv/`, `data/raw/*`, `data/processed/*`,
`models/*.joblib`, `mlflow.db`, `__pycache__/`.

## Stage workflow

Work the PRD stages in order (0→13). For each stage:

1. Re-read that stage's section in `NutriScan_PRD_Roadmap.md`.
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
dvc add data/raw/openfoodfacts_slice.csv
dvc repro

# Experiment tracking
mlflow ui   # sqlite:///mlflow.db

# Local serving
uvicorn app.main:app --reload --port 8000
streamlit run streamlit_app.py

# Container
docker build -t nutriscan .
docker run -p 8000:8000 -p 8501:8501 nutriscan
```

## Standing risks (PRD §6)

- OFF column-name drift between export versions
- NOVA class imbalance — check at Stage 1.1; widen `FOCUS_COUNTRIES` / raise
  `MAX_ROWS` if any class has < ~200 rows
- Live API dependency at demo time — timeouts are not optional
- Scope creep on Stage 12 — `/metrics` + one manual KS-test is enough
```

- [ ] **Step 2: Verify file exists and contains all Hard rules**

Run: `test -f AGENTS.md && grep -c "Hard rules" AGENTS.md && grep -E "macro-F1|nutrition_grade_fr|src/features.py|Never cut Stages" AGENTS.md`
Expected: prints `1` then four matching lines (one per key rule).

- [ ] **Step 3: Cross-check Hard rules against the spec**

Run: `grep -E "macro-F1|nutrition_grade_fr|model.joblib|src/features.py|timeout|cut Stage 11|4, 5, 6, 9" AGENTS.md`
Expected: every pattern from the spec's Hard rules list (rules 1–6) appears at least once.

---

### Task 2: Write thin `CLAUDE.md` shim

**Files:**
- Create: `CLAUDE.md`

**Interfaces:**
- Consumes: `AGENTS.md` from Task 1
- Produces: root `CLAUDE.md` — entrypoint for Claude Code only; no project content

- [ ] **Step 1: Create `CLAUDE.md` with this exact content**

```markdown
# Claude Code — NutriScan

Read and follow `AGENTS.md` in this directory in full. It is the single source
of truth for project context, hard rules, repo structure, stage workflow, and
commands. Do not duplicate or paraphrase its rules here — if something needs
changing, change `AGENTS.md`.

For stage-by-stage tasks, also re-read the relevant section of
`NutriScan_PRD_Roadmap.md` before implementing.
```

- [ ] **Step 2: Verify it is a pointer, not a content copy**

Run: `test -f CLAUDE.md && grep -c "AGENTS.md" CLAUDE.md && ! grep -q "macro-F1" CLAUDE.md && echo OK`
Expected: `1` then `OK` (references AGENTS.md; does not restate Hard rules).

---

### Task 3: Write `opencode.json`

**Files:**
- Create: `opencode.json`

**Interfaces:**
- Consumes: `AGENTS.md` from Task 1
- Produces: root `opencode.json` — opencode project config registering instructions

- [ ] **Step 1: Create `opencode.json` with this exact content**

```json
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": ["AGENTS.md"]
}
```

- [ ] **Step 2: Validate JSON parses and uses only allowed fields**

Run: `python3 -c "import json; c=json.load(open('opencode.json')); assert set(c)=={'$schema','instructions'}; assert c['instructions']==['AGENTS.md']; assert c['$schema']=='https://opencode.ai/config.json'; print('OK')"`
Expected: `OK`

---

### Task 4: Final verification

**Files:**
- Read-only checks across `AGENTS.md`, `CLAUDE.md`, `opencode.json`

**Interfaces:**
- Consumes: Tasks 1–3
- Produces: confirmation all three files exist and don't duplicate content

- [ ] **Step 1: All three files exist at repo root**

Run: `test -f AGENTS.md && test -f CLAUDE.md && test -f opencode.json && echo ALL_PRESENT`
Expected: `ALL_PRESENT`

- [ ] **Step 2: No content duplication — only AGENTS.md has Hard rules**

Run: `grep -l "Hard rules" AGENTS.md CLAUDE.md opencode.json`
Expected: only `AGENTS.md` listed

- [ ] **Step 3: Remind user to restart opencode**

Config loads once at startup. Tell the user: quit and restart opencode so the new `opencode.json` and `AGENTS.md` are picked up. (Claude Code and Antigravity pick up their files on next session start as well.)
