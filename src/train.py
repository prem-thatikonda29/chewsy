"""Stage 4 — modeling + MLflow experiment tracking (PRD 4.1–4.7).

From the repo root:

    python src/train.py --runs 1 2 3 4 5 6   # six tracked experiments
    python src/train.py --register           # pick winner by macro-F1,
                                              # register + alias 'champion'

Every run shares one stratified 80/20 split (random_state=42) and the
Stage 3 feature pipeline — only the feature config and the model family
change, so differences are attributable. Headline metric: macro-F1
(Hard rule 1); accuracy is logged but never the selection criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import shap  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    log_loss,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.features import (  # noqa: E402
    INDICATOR_COLS,
    NovaLabelAdapter,
    build_feature_pipeline,
)

# --- MLflow setup (PRD 4.1) -------------------------------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "chewsy-nova"
REGISTERED_MODEL = "chewsy-nova"

# --- split / data -----------------------------------------------------------
SEED = 42
TEST_SIZE = 0.2
TARGET_COL = "nova_group"
CLASSES = [1, 2, 3, 4]
DATA_PATH = REPO_ROOT / "data" / "processed" / "openfoodfacts_clean.csv"

# Nutri-Score fields are a computed label next to the target — never
# features (Hard rule 2; same guard clean.py applies at Stage 2).
FORBIDDEN_LEAKAGE_FIELDS = (
    "nutrition_grade_fr",
    "nutrition_grades",
    "nutriscore_score",
    "nutriscore_grade",
)

EXPECTED_METRIC_KEYS = {
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "log_loss",
    "f1_class_1",
    "f1_class_2",
    "f1_class_3",
    "f1_class_4",
}

# PRD 4.2/4.3/4.4 + model-diversity extension approved 25 Sep 2026:
# runs 1-2 isolate the text ablation (same model family); runs 3-6 compare
# four model families on the full feature set.
RUNS: dict[int, dict] = {
    1: {"include_text": False, "include_categorical": False, "model": "logreg"},
    2: {"include_text": True, "include_categorical": False, "model": "logreg"},
    3: {"include_text": True, "include_categorical": True, "model": "random_forest"},
    4: {"include_text": True, "include_categorical": True, "model": "hist_gbdt"},
    5: {"include_text": True, "include_categorical": True, "model": "xgboost"},
    6: {"include_text": True, "include_categorical": True, "model": "knn"},
}

# Tree-based runs get a SHAP summary artifact (PRD 4.4 requires run 3).
SHAP_RUNS = (3, 4, 5)

SHAP_BACKGROUND_ROWS = 200


def make_model(family: str):
    """Model factories — every family exposes predict_proba (confidence)."""
    if family == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        )
    if family == "hist_gbdt":
        return HistGradientBoostingClassifier(max_iter=300, random_state=SEED)
    if family == "xgboost":
        return NovaLabelAdapter(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                random_state=SEED,
                n_jobs=-1,
                verbosity=0,
            )
        )
    if family == "knn":
        return KNeighborsClassifier(n_neighbors=15, n_jobs=-1)
    raise ValueError(f"unknown model family: {family}")


def build_full_pipeline(run_no: int) -> Pipeline:
    """Feature pipeline (Stage 3) + classifier as ONE object — this is what
    gets logged to MLflow now and pickled as model.joblib in Stage 5."""
    cfg = RUNS[run_no]
    features = build_feature_pipeline(
        include_text=cfg["include_text"],
        include_categorical=cfg["include_categorical"],
    )
    return Pipeline([
        ("feature_pipeline", features),
        ("model", make_model(cfg["model"])),
    ])


def load_dataset(csv_path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    leaked = [
        c
        for c in df.columns
        if c.startswith("nutriscore") or c in FORBIDDEN_LEAKAGE_FIELDS
    ]
    assert not leaked, f"leakage columns present: {leaked}"
    assert TARGET_COL in df.columns, f"missing target {TARGET_COL}"
    y = df[TARGET_COL].astype(int)
    # `code` is the barcode identifier, not a feature (and uint64 values
    # break MLflow signature inference) — every trained model ignores it.
    X = df.drop(columns=[TARGET_COL, "code"], errors="ignore")
    return X, y


def make_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE, seed: int = SEED
):
    """Stratified 80/20 — same split for every run, split BEFORE any fit."""
    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "log_loss": float(log_loss(y_true, y_proba, labels=labels)),
    }
    per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    for cls, score in zip(labels, per_class):
        metrics[f"f1_class_{int(cls)}"] = float(score)
    return metrics


def feature_names(full_pipeline: Pipeline) -> np.ndarray:
    ct = full_pipeline.named_steps["feature_pipeline"].named_steps["features"]
    return np.asarray(ct.get_feature_names_out(), dtype=object)


def _log_confusion_matrix(y_true, y_pred, tmpdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=CLASSES,
        display_labels=CLASSES,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Confusion matrix — NOVA class (test set)")
    fig.tight_layout()
    path = tmpdir / "confusion_matrix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(path))


def _log_shap_summary(feature_step, model, X_train, names, tmpdir: Path) -> None:
    """PRD 4.4: SHAP summary plot artifact for tree models.

    Background = 200 fixed-seed train rows transformed through the fitted
    feature pipeline; features are the real engineered columns (PCA is
    analysis-only and never feeds the model, Hard rule from Stage 3).
    """
    bg = X_train.sample(n=min(SHAP_BACKGROUND_ROWS, len(X_train)), random_state=SEED)
    bg_t = feature_step.transform(bg)
    # unwrap NovaLabelAdapter only — TreeExplainer wants the raw xgboost
    # model (sklearn forests also expose an unfitted `estimator_` template!)
    tree_model = (
        model.estimator_ if isinstance(model, NovaLabelAdapter) else model
    )
    explainer = shap.TreeExplainer(tree_model)
    values = explainer(bg_t).values
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)

    out_dir = tmpdir / "shap"
    out_dir.mkdir(exist_ok=True)
    if values.ndim == 3:
        for i in range(values.shape[2]):
            plt.figure()
            shap.summary_plot(
                values[:, :, i],
                bg_t,
                feature_names=list(names),
                show=False,
                max_display=18,
            )
            plt.tight_layout()
            plt.savefig(
                out_dir / f"shap_summary_nova_{i + 1}.png", dpi=150,
                bbox_inches="tight",
            )
            plt.close()
    else:
        plt.figure()
        shap.summary_plot(
            values, bg_t, feature_names=list(names), show=False, max_display=18
        )
        plt.tight_layout()
        plt.savefig(out_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
    mlflow.log_artifact(str(out_dir))


def _scalar_params(model) -> dict:
    return {
        k: v
        for k, v in model.get_params().items()
        if v is None or isinstance(v, (str, int, float, bool))
    }


def _input_example(X_train: pd.DataFrame) -> pd.DataFrame:
    """Sample rows for the logged model's schema.

    int indicators → float (so serving ints/NaNs both coerce) and one row
    with NaN in the indicator/text/categorical columns, which marks those
    columns *optional* in the inferred signature — Stage 6 rows arrive
    without missingness indicators and may lack brand/name text; the
    pipeline fills them at transform time. Without this, pyfunc schema
    enforcement rejects those rows before the model ever runs.
    """
    example = X_train.head(5).copy()
    for c in example.select_dtypes("int").columns:
        example[c] = example[c].astype("float64")
    first = example.index[0]
    nullable = list(INDICATOR_COLS) + [
        "product_name", "brands", "categories_tags", "ingredients_pseudo_text",
    ]
    for c in nullable:
        if c in example.columns:
            example.loc[first, c] = np.nan
    return example


def run_experiment(
    run_no: int,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    data_sha256: str,
) -> dict[str, float]:
    cfg = RUNS[run_no]
    pipe = build_full_pipeline(run_no)
    model = pipe.named_steps["model"]

    with mlflow.start_run(run_name=f"run{run_no}_{cfg['model']}") as active:
        mlflow.set_tags({
            "stage": "4",
            "run_no": str(run_no),
            "role": "text-ablation" if run_no in (1, 2) else "model-comparison",
        })
        params = {
            "run_no": run_no,
            "model_family": cfg["model"],
            "include_text": cfg["include_text"],
            "include_categorical": cfg["include_categorical"],
            "split_strategy": "stratified",
            "split_seed": SEED,
            "test_size": TEST_SIZE,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "data_sha256": data_sha256,
        }
        params.update(_scalar_params(model))

        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_sec = time.time() - t0
        params["fit_sec"] = round(fit_sec, 2)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        labels = np.asarray(model.classes_)
        metrics = evaluate(y_test, y_pred, y_proba, labels)
        metrics = {k: v for k, v in metrics.items() if k in EXPECTED_METRIC_KEYS}
        params["feature_count"] = int(len(feature_names(pipe)))

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as td:
            _log_confusion_matrix(y_test, y_pred, Path(td))
            if run_no in SHAP_RUNS:
                _log_shap_summary(
                    pipe.named_steps["feature_pipeline"],
                    model,
                    X_train,
                    feature_names(pipe),
                    Path(td),
                )

        print(
            f"run{run_no:>2} {cfg['model']:<15} "
            f"f1_macro={metrics['f1_macro']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"log_loss={metrics['log_loss']:.4f} "
            f"fit={fit_sec:.1f}s "
            f"({active.info.run_id})"
        )
        return metrics


def print_comparison() -> pd.DataFrame:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print("no experiment yet — run --runs first")
        return pd.DataFrame()
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_macro DESC", "metrics.accuracy DESC"],
    )
    cols = [
        c
        for c in (
            "tags.mlflow.runName",
            "params.model_family",
            "params.include_text",
            "params.include_categorical",
            "metrics.f1_macro",
            "metrics.accuracy",
            "metrics.log_loss",
            "run_id",
        )
        if c in df.columns
    ]
    table = df[cols].rename(columns={
        "tags.mlflow.runName": "run",
        "params.model_family": "model",
        "params.include_text": "text",
        "params.include_categorical": "cat",
        "metrics.f1_macro": "f1_macro",
        "metrics.accuracy": "accuracy",
        "metrics.log_loss": "log_loss",
    })
    print("\n=== runs by macro-F1 (headline metric) ===")
    print(table.to_string(index=False))
    return table


def register_champion(csv_path: Path | str = DATA_PATH) -> str:
    """PRD 4.5/4.6: winner by macro-F1 → pack + register → alias
    'champion' → verify mlflow.pyfunc.load_model("models:/<name>@champion").

    Only the champion gets a logged model binary. The six comparison runs
    log params/metrics/plots only (per PRD 4.2–4.4) — mlflow's default
    skops serialization inflates these pipelines ~5x, so pickling every
    run would put 600+ MB of duplicate model binaries in the repo. The
    champion is packed from a dedicated run with pickle format, which
    keeps the single shippable artifact well under GitHub's 100 MB limit.
    """
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    assert exp is not None, "no runs yet"
    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1_macro DESC", "metrics.accuracy DESC"],
    )
    assert not df.empty, "no runs to register"
    # re-runs must not pick a previous packaging run as the winner
    if "tags.role" in df.columns:
        df = df[df["tags.role"].fillna("") != "champion"]
        assert not df.empty, "only packaging runs present — run --runs first"
    best = df.iloc[0]
    source_run_id = best["run_id"]
    run_no = int(best["params.run_no"])
    f1 = float(best["metrics.f1_macro"])
    print(f"winner: {best['tags.mlflow.runName']} "
          f"({best['params.model_family']}) f1_macro={f1:.4f} run={source_run_id}")

    # refit the winning config on the same train split (deterministic —
    # identical model to the comparison run) and log it for registration
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = make_split(X, y)
    pipe = build_full_pipeline(run_no)
    pipe.fit(X_train, y_train)

    params = {
        k.removeprefix("params."): v
        for k, v in best.items()
        if k.startswith("params.") and v is not None
    }
    metrics = {
        k.removeprefix("metrics."): float(v)
        for k, v in best.items()
        if k.startswith("metrics.") and v is not None
    }
    with mlflow.start_run(run_name=f"champion_packaging_run{run_no}") as active:
        mlflow.set_tags({
            "stage": "4",
            "role": "champion",
            "source_run_id": source_run_id,
        })
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        example = _input_example(X_train)
        signature = infer_signature(example, pipe.predict(example))
        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
        packaging_run_id = active.info.run_id

    mv = mlflow.register_model(
        f"runs:/{packaging_run_id}/model", REGISTERED_MODEL
    )
    client = MlflowClient()
    for _ in range(30):
        info = client.get_model_version(REGISTERED_MODEL, mv.version)
        if getattr(info, "status", "READY") == "READY":
            break
        time.sleep(1)
    client.set_registered_model_alias(
        REGISTERED_MODEL, "champion", str(mv.version)
    )

    loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL}@champion")
    sample = X_test.iloc[[0]]
    pred = loaded.predict(sample)
    assert int(pred[0]) in CLASSES, pred
    print(
        f"registered {REGISTERED_MODEL} v{mv.version} alias=champion; "
        f"pyfunc load OK — sample barcode row predicts NOVA {int(pred[0])} "
        f"(true {int(y_test.iloc[0])})"
    )
    return f"models:/{REGISTERED_MODEL}@champion"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", type=int, choices=sorted(RUNS))
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    args = parser.parse_args(argv)
    if not args.runs and not args.register:
        parser.error("pass --runs and/or --register")

    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.runs:
        data_sha256 = hashlib.sha256(args.csv.read_bytes()).hexdigest()[:16]
        X, y = load_dataset(args.csv)
        X_train, X_test, y_train, y_test = make_split(X, y)
        print(f"split: train={len(X_train)} test={len(X_test)} "
              f"classes={dict(y.value_counts().sort_index())}")
        for run_no in args.runs:
            run_experiment(run_no, X_train, X_test, y_train, y_test, data_sha256)
        print_comparison()

    if args.register:
        register_champion(args.csv)
        print_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
