"""Stage 10 — reader for params.yaml, the DVC-tracked pipeline parameters.

Only values the pipeline scripts actually read live in params.yaml: no
decorative params. Each dvc.yaml stage declares exactly the keys it
consumes, so `dvc repro` hard-fails if a declared key disappears; the
script-side defaults here exist for direct manual runs only (e.g.
`python src/features.py` outside the pipeline) and never paper over a
broken pipeline.

Dotted key paths match what DVC stores in dvc.lock
(e.g. ``fetch.target_per_class``).
"""

from __future__ import annotations

from pathlib import Path

import yaml

PARAMS_PATH = Path(__file__).resolve().parents[1] / "params.yaml"


def load() -> dict:
    """Whole params.yaml as a dict ({} when the file doesn't exist)."""
    if not PARAMS_PATH.is_file():
        return {}
    data = yaml.safe_load(PARAMS_PATH.read_text())
    return data or {}


def get(section: str, key: str, default):
    """params.yaml#[section][key], or ``default`` when absent/unparseable."""
    try:
        return load()[section][key]
    except (KeyError, TypeError):
        return default
