# NutriScan — Multi-Tool Agent Configuration Design

**Date:** 2026-09-24
**Status:** Approved
**Updated:** 2026-09-24 — synced to PRD revision (Streamlit → Next.js frontend, CORS, multi-stage Docker)
**Scope:** Instruction/config files so opencode, Claude Code, and Antigravity can all work on this repo with shared project context.

## Problem

The NutriScan PRD (`NutriScan_PRD_Roadmap.md`) holds all project knowledge, but coding agents start with no context unless a human pastes it. Three tools are in use — opencode, Claude Code, Antigravity — each with a different native instruction-file convention. Duplicating full instructions per tool guarantees drift.

## Chosen approach: shared core + thin tool shims

**Single source of truth:** `AGENTS.md` at the repo root.

| Tool | How it picks up context |
|---|---|
| opencode | Native `AGENTS.md` discovery; also listed explicitly in `opencode.json` → `instructions` |
| Antigravity | Native `AGENTS.md` convention |
| Claude Code | Native convention is `CLAUDE.md`; a thin `CLAUDE.md` points at `AGENTS.md` |

Content lives in exactly one place (`AGENTS.md`). Tool-specific files are pointers only.

## Files to create

### 1. `AGENTS.md` (root)

Sections distilled from the PRD:

1. **Project overview** — one-liner (Yuka-style barcode scanner → Open Food Facts → NOVA 1–4 prediction + SHAP), course context, deadline 26 Sept 2026.
2. **Hard rules (never violate)** — pull the PRD's non-negotiables inline:
   - Headline metric is **macro-F1**, not accuracy (imbalanced NOVA classes).
   - `nutrition_grade_fr` is **never** a feature (leakage).
   - Frontend must call FastAPI `/predict` over HTTP; never reimplement scoring or ship the model in the UI (API is single source of truth); enable CORS for the Next.js origin.
   - Custom transformer classes live in `src/features.py` (importable module) — avoids the pickling `FeatureCreator` bug.
   - OFF API calls need timeouts + clean 404/503 fallbacks (live-demo risk).
   - If behind schedule: cut Stage 11 first, trim Stage 12; never cut Stages 4, 5, 6, 9, 10.
3. **Repo structure** — the Section 4 tree, as the canonical layout (Stage 0 creates it; don't improvise).
4. **Stage workflow** — work stage-by-stage; a stage is done only when its "Done when" condition is verifiably true; commit message style per stage (`feat:`, `chore:`, `ci:`).
5. **Key commands** — venv + `pip install -r requirements.txt`, `pytest`, `dvc repro`, `mlflow ui`, `docker build/run`, `uvicorn`, `npm run dev` (Next.js frontend).
6. **Pointers** — full stage detail lives in `NutriScan_PRD_Roadmap.md`; agents should re-read the relevant stage section before implementing it.

Tone: imperative, checklist-friendly, no prose padding. Suitable as a system-prompt-adjacent instruction file.

### 2. `CLAUDE.md` (root)

Thin shim (~5–10 lines):

- Instruct Claude Code to read and follow `AGENTS.md` in full.
- Optionally note Claude-only quirks if any emerge (none known at design time — keep empty until needed).

No duplicated project content.

### 3. `opencode.json` (root)

```json
{
  "$schema": "https://opencode.ai/config.json",
  "instructions": ["AGENTS.md"]
}
```

Explicit registration so opencode loads `AGENTS.md` even if convention discovery changes. No agents, skills, MCP, or permissions — out of scope / YAGNI for a 2-day build.

## Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Three fat files (`AGENTS.md` + full `CLAUDE.md` + Antigravity-specific) | Content drift across three copies; maintenance cost with no rubric benefit |
| Custom `.opencode/agent/` subagents, skills, MCP servers | Overhead; the PRD is already the workflow spec; not graded |
| Only `AGENTS.md`, no `CLAUDE.md` | Claude Code may not reliably auto-read `AGENTS.md` depending on version; a 5-line pointer is cheap insurance |

## Error handling / edge cases

- **PRD updates mid-project:** edit `AGENTS.md` only; shims never need changes.
- **opencode config strictness:** `opencode.json` uses only schema-backed fields (`$schema`, `instructions`); invalid config would block startup, so keep it minimal.
- **Claude Code import style:** use plain prose ("Read and follow AGENTS.md") rather than `@AGENTS.md` import syntax to avoid version-specific import behavior.
- **Git not yet initialized:** design doc and agent files are created on disk; first commit happens at PRD Stage 0 (`chore: repo scaffolding`), which should include these files.

## Testing

- Manual verification after implementation:
  - `AGENTS.md`, `CLAUDE.md`, `opencode.json` exist at repo root.
  - `opencode.json` parses as valid JSON and validates against `https://opencode.ai/config.json` fields used.
  - Content cross-check: every "Hard rule" above is present in `AGENTS.md` and traceable to a PRD line.
  - Restart opencode → `AGENTS.md` appears in loaded instructions (user-facing check).

## Out of scope

- Creating the Stage 0 repo scaffold, code, tests, Docker, CI — those are the PRD's stages, not this config task.
- Global (`~/.config/opencode`) config changes.
- Tool-specific subagents/commands/skills.
