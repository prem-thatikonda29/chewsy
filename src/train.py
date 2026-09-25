"""Stage 4 — modeling + MLflow experiment tracking (PRD 4.1–4.7).

From the repo root:

    python src/train.py --sweep 0 0.1 0.2 0.3  # text-dropout rate sweep
                                                # (champion config, run 5)
    python src/train.py --runs 1 2 3 4 5 6     # six tracked experiments
                                                # at the selected rate
    python src/train.py --register           # pick winner by macro-F1,
                                              # register + alias 'champion'

Every run shares one stratified 80/20 split (random_state=42) and the
Stage 3 feature pipeline — only the feature config and the model family
change, so differences are attributable. Headline metric: macro-F1
(Hard rule 1); accuracy is logged but never the selection criterion.
Every run ALSO logs sparse metrics (stub-shaped eval input) because ~76%
of live OFF records are thin — the full-data number alone overstates
real-world performance.
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
    COUNT_COLS,
    INDICATOR_COLS,
    NovaLabelAdapter,
    build_feature_pipeline,
)
from src.params import get  # noqa: E402

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

# --- sparse-view text dropout (approved 25 Sep 2026 + reviewer conditions) ---
# Root cause (measured): the training source has NO no-ingredient-list
# NOVA 3/4 rows at all -- no-text rows are {1: 179, 2: 3759, 3: 96, 4: 0}.
# But3/4 of those are class 2 LEGITIMATELY: single-ingredient products
# (olive oil, sugar, salt) have no ingredient list because the product IS
# the ingredient, so "no list -> NOVA 2" is partly correct domain logic.
# Blank text alone would teach the model to unlearn that. The real stub is
# thinner and distinguishable: no list AND no category tags AND counts
# unpublished (live Diet Coke 5000112644906 matches all three; measured:
# 100% of training rows carry tags, so training never held this shape).
#
# Fix: append FULL-STUB twins of a seeded, stratified fraction of TRAIN
# rows -- blank ingredient text, blank category tags, counts NaN,
# text_was_missing=1, nutrients/brand/name KEPT (stubs carry them) --
# across all four classes, so stubs inherit label-proportional priors
# instead of the class-2-only prior. The eval split is never augmented
# (run_experiment takes train slices only; a hash tripwire asserts the
# eval frame is unchanged). Rate is a logged param, swept via --sweep.
AUGMENT_TEXT_DROPOUT = float(
    get("train", "text_dropout_frac", 0.20)
)  # selected rate; --sweep measures 0/10/20/30% (params.yaml#train.text_dropout_frac)


def augment_text_dropout(
    X: pd.DataFrame, y: pd.Series, frac: float, seed: int = SEED
) -> tuple[pd.DataFrame, pd.Series]:
    """Append full-stub twins of ``frac`` of train rows, STRATIFIED by class.

    Deterministic for a given seed, so comparison runs and
    register_champion's refit see identical training data. ``frac=0`` is a
    no-op (the control rate in the sweep). Train-only by construction:
    callers pass the train slice from a split made BEFORE this call.
    """
    if frac <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    y_arr = y.to_numpy()
    picks = []
    for cls in sorted(np.unique(y_arr)):  # every class represented, not
        cls_rows = np.flatnonzero(y_arr == cls)  # a uniform sample's luck
        take = min(len(cls_rows), max(1, int(len(cls_rows) * frac)))
        picks.append(rng.choice(cls_rows, size=take, replace=False))
    idx = np.concatenate(picks)

    stub = X.iloc[idx].copy()
    stub["ingredients_pseudo_text"] = ""
    stub["categories_tags"] = ""
    stub["text_was_missing"] = 1
    for c in COUNT_COLS:
        stub[c] = np.nan
    X_aug = pd.concat([X, stub], ignore_index=True)
    y_aug = pd.concat([y, y.iloc[idx]], ignore_index=True)
    return X_aug, y_aug


def simulate_stub_input(X: pd.DataFrame) -> pd.DataFrame:
    """Stub-shaped view of eval rows -- the SAME blanks augmentation makes.

    Applied to the untouched eval split so every run logs comparable
    sparse metrics alongside the baseline ones (the live population is
    ~76% thin records; one number would overstate real-world performance).
    """
    Xs = X.copy()
    Xs["ingredients_pseudo_text"] = ""
    Xs["categories_tags"] = ""
    Xs["text_was_missing"] = 1
    for c in COUNT_COLS:
        Xs[c] = np.nan
    return Xs


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
        n_svd_components=int(get("features", "n_svd_components", 25)),
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
    text_dropout_frac: float = 0.0,
    run_name: str | None = None,
    role: str | None = None,
) -> dict[str, float]:
    cfg = RUNS[run_no]
    # Train-only, post-split (reviewer condition): X_train is the train
    # slice from one split made BEFORE this call; the eval frame is never
    # augmented -- hashed here and asserted unchanged after the fit, so a
    # blanked twin leaking into eval fails loudly instead of silently.
    X_fit, y_fit = augment_text_dropout(X_train, y_train, text_dropout_frac)
    n_source = len(X_train)
    eval_hash = pd.util.hash_pandas_object(X_test).sum()
    pipe = build_full_pipeline(run_no)
    model = pipe.named_steps["model"]

    with mlflow.start_run(
        run_name=run_name or f"run{run_no}_{cfg['model']}"
    ) as active:
        mlflow.set_tags({
            "stage": "4",
            "run_no": str(run_no),
            "role": role
            or ("text-ablation" if run_no in (1, 2) else "model-comparison"),
        })
        params = {
            "run_no": run_no,
            "model_family": cfg["model"],
            "include_text": cfg["include_text"],
            "include_categorical": cfg["include_categorical"],
            "split_strategy": "stratified",
            "split_seed": SEED,
            "test_size": TEST_SIZE,
            "n_train": len(X_fit),
            "n_train_source": n_source,
            "text_dropout_frac": float(text_dropout_frac),
            "n_test": len(X_test),
            "data_sha256": data_sha256,
        }
        params.update(_scalar_params(model))

        t0 = time.time()
        pipe.fit(X_fit, y_fit)
        fit_sec = time.time() - t0
        params["fit_sec"] = round(fit_sec, 2)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        labels = np.asarray(model.classes_)
        metrics = evaluate(y_test, y_pred, y_proba, labels)
        metrics = {k: v for k, v in metrics.items() if k in EXPECTED_METRIC_KEYS}

        # Sparse metrics: the same eval rows through stub-shaped input.
        # Logged per run (reviewer condition) so every experiment reports
        # BOTH populations -- complete records and thin live stubs.
        stub_pred = pipe.predict(simulate_stub_input(X_test))
        metrics["sparse_accuracy"] = float(accuracy_score(y_test, stub_pred))
        metrics["sparse_f1_macro"] = float(
            f1_score(y_test, stub_pred, average="macro")
        )
        stub_per_class = f1_score(y_test, stub_pred, average=None, labels=labels)
        metrics["sparse_f1_class_3"] = float(stub_per_class[2])
        metrics["sparse_f1_class_4"] = float(stub_per_class[3])

        params["feature_count"] = int(len(feature_names(pipe)))
        assert pd.util.hash_pandas_object(X_test).sum() == eval_hash, (
            "eval split mutated — augmentation must only ever touch train rows"
        )

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as td:
            _log_confusion_matrix(y_test, y_pred, Path(td))
            if run_no in SHAP_RUNS:
                _log_shap_summary(
                    pipe.named_steps["feature_pipeline"],
                    model,
                    X_fit,  # background sampled from the ACTUAL training data
                    feature_names(pipe),
                    Path(td),
                )

        print(
            f"run{run_no:>2} {cfg['model']:<15} "
            f"f1_macro={metrics['f1_macro']:.4f} "
            f"sparse={metrics['sparse_f1_macro']:.4f} "
            f"s4={metrics['sparse_f1_class_4']:.4f} "
            f"drop={text_dropout_frac:.2f} "
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
            "params.text_dropout_frac",
            "metrics.f1_macro",
            "metrics.accuracy",
            "metrics.sparse_f1_macro",
            "metrics.sparse_f1_class_3",
            "metrics.sparse_f1_class_4",
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
        "params.text_dropout_frac": "drop",
        "metrics.f1_macro": "f1_macro",
        "metrics.accuracy": "accuracy",
        "metrics.sparse_f1_macro": "sparse_f1",
        "metrics.sparse_f1_class_3": "sparse_f1_c3",
        "metrics.sparse_f1_class_4": "sparse_f1_c4",
        "metrics.log_loss": "log_loss",
    })
    print("\n=== runs by macro-F1 (headline metric) + sparse companion ===")
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
    # re-runs must not pick a previous packaging run OR a sweep run as the
    # winner: sweeps measure rates, the matrix decides the champion
    if "tags.role" in df.columns:
        excluded = df["tags.role"].fillna("").isin(["champion", "text-dropout-sweep"])
        df = df[~excluded]
        assert not df.empty, "only packaging/sweep runs present — run --runs first"
    # only runs trained on the CURRENT dataset may win the registry slot:
    # earlier data generations keep their runs as comparison evidence, but
    # their metrics/data_sha256 describe a dataset the packaged model no
    # longer sees (seen after the 25 Sep 2026 count-fill regeneration).
    current_sha = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:16]
    if "params.data_sha256" in df.columns:
        on_data = df["params.data_sha256"].fillna("") == current_sha
        if on_data.any():
            df = df[on_data]
        assert not df.empty, "no runs on the current dataset — run --runs first"
    # only runs at the SELECTED text-dropout rate may win: sweep runs at
    # other rates (and pre-augmentation runs lacking the param entirely)
    # describe a different training distribution than the refit below.
    assert "params.text_dropout_frac" in df.columns, (
        "no text_dropout_frac param on any run — run --runs first"
    )
    fracs = pd.to_numeric(df["params.text_dropout_frac"], errors="coerce")
    on_rate = np.isclose(
        fracs.to_numpy(dtype=float), AUGMENT_TEXT_DROPOUT, equal_nan=False
    )
    assert on_rate.any(), (
        f"no runs at text_dropout_frac={AUGMENT_TEXT_DROPOUT} — "
        "run --runs first (set AUGMENT_TEXT_DROPOUT to the swept rate)"
    )
    df = df[on_rate]
    best = df.iloc[0]
    source_run_id = best["run_id"]
    run_no = int(best["params.run_no"])
    f1 = float(best["metrics.f1_macro"])
    print(f"winner: {best['tags.mlflow.runName']} "
          f"({best['params.model_family']}) f1_macro={f1:.4f} run={source_run_id}")

    # refit the winning config on the same train split (deterministic —
    # identical model to the comparison run, including the train-only
    # text-dropout augmentation at the selected rate) and log it
    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = make_split(X, y)
    X_fit, y_fit = augment_text_dropout(X_train, y_train, AUGMENT_TEXT_DROPOUT)
    pipe = build_full_pipeline(run_no)
    pipe.fit(X_fit, y_fit)

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
    parser.add_argument(
        "--sweep",
        nargs="*",
        type=float,
        metavar="FRAC",
        help="text-dropout rate sweep on the champion config (run 5): "
             "one tracked MLflow run per rate, e.g. --sweep 0 0.1 0.2 0.3",
    )
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--csv", type=Path, default=DATA_PATH)
    args = parser.parse_args(argv)
    if not args.runs and not args.register and not args.sweep:
        parser.error("pass --runs, --sweep and/or --register")

    mlflow.set_experiment(EXPERIMENT_NAME)

    if args.runs or args.sweep:
        data_sha256 = hashlib.sha256(args.csv.read_bytes()).hexdigest()[:16]
        X, y = load_dataset(args.csv)
        X_train, X_test, y_train, y_test = make_split(X, y)
        print(f"split: train={len(X_train)} test={len(X_test)} "
              f"classes={dict(y.value_counts().sort_index())}")
        if args.sweep:
            for frac in args.sweep:
                run_experiment(
                    5,
                    X_train, X_test, y_train, y_test,
                    data_sha256,
                    text_dropout_frac=float(frac),
                    run_name=f"sweep_drop{frac:g}_xgboost",
                    role="text-dropout-sweep",
                )
            print_comparison()
        if args.runs:
            for run_no in args.runs:
                run_experiment(
                    run_no,
                    X_train, X_test, y_train, y_test,
                    data_sha256,
                    text_dropout_frac=AUGMENT_TEXT_DROPOUT,
                )
            print_comparison()

    if args.register:
        register_champion(args.csv)
        print_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
