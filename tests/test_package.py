"""Stage 5 — packaged artifact tests (PRD 5.2–5.3)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"
POINTER_PATH = REPO_ROOT / "models" / "model.joblib.dvc"
REFERENCE_PATH = REPO_ROOT / "models" / "training_reference.csv"

# Runs in a brand-new interpreter: proves every class the pickle references
# (FeatureCreator, NovaLabelAdapter, ...) resolves on `joblib.load`, which
# is exactly what app/main.py does at startup (PRD 5.2).
FRESH_PROCESS_SCRIPT = """
import sys
sys.path.insert(0, ".")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

row = pd.read_csv("models/training_reference.csv", nrows=5)
pipeline = joblib.load("models/model.joblib")
preds = pipeline.predict(row)
proba = pipeline.predict_proba(row)
assert proba.shape == (5, 4), proba.shape
assert set(preds) <= {1, 2, 3, 4}, preds
assert (proba.sum(axis=1) - 1.0).__abs__().max() < 1e-6

# the packaged artifact must BE the registered champion, not a drift
mlflow.set_tracking_uri("sqlite:///mlflow.db")
registered = mlflow.sklearn.load_model("models:/chewsy-nova@champion")
assert (registered.predict(row) == preds).tolist(), "joblib != registry"

print("preds", preds.tolist())
"""


class TestPackagedModel:
    def test_fresh_process_loads_predicts_and_matches_registry(self):
        result = subprocess.run(
            [sys.executable, "-c", FRESH_PROCESS_SCRIPT],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().startswith("preds")

    def test_model_is_dvc_tracked_for_fresh_clones(self):
        """Stage 10 contract: the model is a pipeline out, not a static
        pointer — fresh clones restore it with `dvc pull` from dvc.lock."""
        assert MODEL_PATH.exists(), "missing — run: python src/package.py"
        assert not POINTER_PATH.exists(), (
            "static *.dvc pointer must be gone (AGENTS: dvc remove before "
            "pipeline outs claim the same file)"
        )
        dvc_yaml = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
        assert "models/model.joblib" in dvc_yaml["stages"]["train"]["outs"]
        dvc_lock = yaml.safe_load((REPO_ROOT / "dvc.lock").read_text())
        locked_outs = dvc_lock["stages"]["train"]["outs"]
        assert any(o["path"] == "models/model.joblib" for o in locked_outs), (
            "dvc.lock missing the model out — `dvc pull` would not restore it"
        )


class TestTrainingReference:
    """Committed to git (unlike the joblib) so drift tests never depend
    on `dvc pull`."""

    def test_is_the_live_input_space(self):
        ref = pd.read_csv(REFERENCE_PATH, nrows=10)
        # 22 cleaned columns minus the barcode identifier and the target:
        # a live API row carries neither
        assert "code" not in ref.columns
        assert "nova_group" not in ref.columns
        assert len(ref.columns) == 20

    def test_full_snapshot_with_nulls_preserved(self):
        ref = pd.read_csv(REFERENCE_PATH)
        assert len(ref) == 19_998
        # NaN-preserving: Stage 12 compares per-column null fractions,
        # so dropped-ingredient columns must keep their empties
        assert ref["fiber_100g"].isna().any()

    def test_contains_no_leakage_fields(self):
        cols = pd.read_csv(REFERENCE_PATH, nrows=0).columns
        leaked = [
            c
            for c in cols
            if c.startswith("nutriscore")
            or c in {"nutrition_grade_fr", "nutrition_grades", "nova_group"}
        ]
        assert not leaked, leaked
