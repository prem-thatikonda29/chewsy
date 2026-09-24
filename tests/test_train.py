"""Stage 4 — tests for the training/MLflow utilities (PRD 4.x)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.features import COUNT_COLS, NUMERIC_COLS
from src.train import (
    EXPECTED_METRIC_KEYS,
    RUNS,
    build_full_pipeline,
    evaluate,
    load_dataset,
    make_model,
    make_split,
)

CSV = Path("data/processed/openfoodfacts_clean.csv")


def test_run_matrix_matches_prd_plan():
    assert set(RUNS) == {1, 2, 3, 4, 5, 6}
    for cfg in RUNS.values():
        assert {"include_text", "include_categorical", "model"} <= set(cfg)
    # PRD 4.2 / 4.3 ablation: only the text flag changes between runs 1 and 2
    assert RUNS[1]["include_text"] is False
    assert RUNS[2]["include_text"] is True
    assert RUNS[1]["model"] == RUNS[2]["model"] == "logreg"
    # model diversity across the full-feature runs
    families = {RUNS[n]["model"] for n in (3, 4, 5, 6)}
    assert len(families) == 4


def test_make_split_is_stratified_and_deterministic():
    n = 400
    y = pd.Series(np.tile([1, 2, 3, 4], n // 4), name="nova_group")
    X = pd.DataFrame({"a": np.arange(n, dtype=float)})

    X_train, X_test, y_train, y_test = make_split(X, y)
    X_train2, X_test2, y_train2, y_test2 = make_split(X, y)

    assert list(X_train.index) == list(X_train2.index)
    assert list(X_test.index) == list(X_test2.index)
    assert len(X_test) == n // 5
    for split_y in (y_train, y_test):
        props = split_y.value_counts(normalize=True).sort_index()
        assert np.allclose(props.values, 0.25, atol=1e-6)


def test_evaluate_reports_headline_metrics():
    y_true = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    y_pred = np.array([1, 1, 2, 4, 3, 3, 4, 2])
    y_proba = np.full((8, 4), 0.25)

    metrics = evaluate(y_true, y_pred, y_proba, labels=np.array([1, 2, 3, 4]))

    assert set(metrics) == EXPECTED_METRIC_KEYS
    assert metrics["f1_macro"] == pytest.approx(
        f1_score(y_true, y_pred, average="macro")
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["log_loss"] > 0.0


def make_synthetic_frame(n=120, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "product_name": [f"item {i}" for i in range(n)],
        "brands": rng.choice(["acme", "globex", "initech", None], n),
        "categories_tags": rng.choice(
            ["en:snacks,en:crackers", "en:beverages,en:teas",
             "en:dairies,en:milks", "en:condiments,en:sauces"], n),
        "ingredients_pseudo_text": [
            "sugar salt oil" if i % 3 else None for i in range(n)],
        "sodium_100g": rng.normal(0.4, 0.2, n),
    }
    for col in NUMERIC_COLS:
        data[col] = np.abs(rng.normal(150.0, 80.0, n))
    for col in COUNT_COLS:
        data[col] = rng.integers(0, 12, n).astype(float)
    frame = pd.DataFrame(data)
    # a few NaNs so the imputer path is exercised
    frame.loc[frame.sample(frac=0.1, random_state=1).index, "fiber_100g"] = np.nan
    return frame


def test_full_pipeline_smoke_fits_and_predicts():
    frame = make_synthetic_frame()
    y = pd.Series(np.tile([1, 2, 3, 4], len(frame) // 4 + 1)[: len(frame)])
    pipe = build_full_pipeline(1)

    pipe.fit(frame, y)
    y_pred = pipe.predict(frame)
    proba = pipe.predict_proba(frame)

    assert set(np.unique(y_pred)) <= {1, 2, 3, 4}
    assert proba.shape == (len(frame), 4)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_make_model_families():
    for family in {cfg["model"] for cfg in RUNS.values()}:
        model = make_model(family)
        assert hasattr(model, "predict_proba")


@pytest.mark.skipif(not CSV.exists(), reason="DVC data not pulled")
def test_load_dataset_leakage_guard():
    X, y = load_dataset()
    forbidden = [
        c for c in X.columns
        if c.startswith("nutriscore") or c in {"nutrition_grade_fr", "nutrition_grades"}
    ]
    assert forbidden == []
    assert "nova_group" not in X.columns
    assert set(y.unique()) == {1, 2, 3, 4}
