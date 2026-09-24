"""Stage 5 — freeze the champion as a portable artifact (PRD 5.1–5.3).

From the repo root:

    python src/package.py

Loads the registered champion (`models:/chewsy-nova@champion`) from the
MLflow registry — provenance-clean, byte-identical to the committed
`mlruns/` artifact — and dumps it as `models/model.joblib`, the single
artifact the API (Stage 6) and `batch_predict.py` load.

Also writes `models/training_reference.csv`: a snapshot of the *input*
space the model was trained on (the cleaned columns, NaN-preserving,
without `code` or `nova_group`) — the drift baseline Stage 12 compares
live API rows against.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402

from src.train import load_dataset  # noqa: E402

CHAMPION = "models:/chewsy-nova@champion"
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
REFERENCE_PATH = REPO_ROOT / "models" / "training_reference.csv"


def package_champion() -> Path:
    """Registry champion → models/model.joblib (PRD 5.1)."""
    pipeline = mlflow.sklearn.load_model(CHAMPION)
    joblib.dump(pipeline, MODEL_PATH)
    return MODEL_PATH


def write_training_reference() -> tuple[Path, int, int]:
    """Input-space snapshot for Stage 12 drift checks (PRD 5.3).

    Uses `load_dataset()`'s column selection — the same feature space the
    champion was trained on, minus the barcode identifier and the target
    (a live API row carries neither). NaNs stay in: null-fraction drift is
    part of the Stage 12 comparison.
    """
    X, _ = load_dataset()
    X.to_csv(REFERENCE_PATH, index=False)
    return REFERENCE_PATH, *X.shape


def verify_roundtrip() -> tuple[int, float]:
    """Reload the dump and predict once — the PRD 5.2 check, minus the
    fresh-process part (that lives in tests/test_package.py)."""
    pipeline = joblib.load(MODEL_PATH)
    row = pd.read_csv(REFERENCE_PATH, nrows=1)
    pred = int(pipeline.predict(row)[0])
    proba = float(pipeline.predict_proba(row).max())
    return pred, proba


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_path = package_champion()
    print(f"5.1 model     → {model_path} ({model_path.stat().st_size / 1e6:.1f} MB)")
    ref_path, n_rows, n_cols = write_training_reference()
    print(f"5.3 reference → {ref_path} ({n_rows} x {n_cols})")
    pred, proba = verify_roundtrip()
    print(f"5.2 roundtrip → predict={pred} confidence={proba:.3f}")
    print("next: dvc add models/model.joblib && dvc push && pytest")


if __name__ == "__main__":
    main()
