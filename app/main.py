"""Stage 6.3 -- FastAPI serving layer (PRD 6.3).

Run from the repo root:

    uvicorn app.main:app --reload --port 8000

Design points:
- ``model.joblib`` is loaded ONCE at startup (lifespan), never per request.
- The model computes the verdict on EVERY scan (Hard rule 11): the flow is
  barcode -> live OFF fetch -> Stage 2 normalization -> pipeline -> SHAP.
  OFF's stored ``nova_group`` is stripped in off_client and asserted absent
  here before the frame reaches the model.
- SHAP runs on the transformed feature row against a TreeExplainer built
  once at startup; features are the real engineered columns (PCA is
  analysis-only, never in the model).
- CORS is enabled for the Next.js origin (env ``FRONTEND_ORIGINS``) --
  without it the browser blocks Stage 7's fetch calls.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import off_client
from app.schemas import (
    NOVA_LABELS,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    ShapFeature,
)
from src.clean import FORBIDDEN_LEAKAGE_FIELDS, normalize_live_row

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "model.joblib"

# in-memory counters (PRD 6.3 -- simple is fine for this scope)
METRICS = {"requests": 0, "errors": 0, "latency_sum_ms": 0.0}

# loaded once at startup, read on every request
STATE: dict = {"pipeline": None, "explainer": None, "feature_names": None}


def load_pipeline(path: Path = MODEL_PATH):
    """Load the packaged champion (Stage 5 artifact)."""
    assert path.exists(), (
        f"missing model artifact: {path} -- run: python src/package.py "
        "(or dvc pull for a fresh clone)"
    )
    return joblib.load(path)


def _feature_names(pipeline) -> list[str]:
    """Names for the 44 transformed columns -- same navigation as
    src/train.py::feature_names (SHAP needs them for readable labels)."""
    ct = pipeline.named_steps["feature_pipeline"].named_steps["features"]
    return [str(n) for n in ct.get_feature_names_out()]


def _build_explainer(pipeline) -> shap.TreeExplainer:
    """TreeExplainer over the raw xgboost estimator (unwrap NovaLabelAdapter)."""
    model = pipeline.named_steps["model"]
    tree_model = getattr(model, "estimator_", model)
    return shap.TreeExplainer(tree_model)


def score_row(pipeline, df: pd.DataFrame) -> tuple[int, np.ndarray]:
    """One scoring path shared by POST /predict and batch_predict.

    Returns ``(predicted_nova, class_probabilities)`` -- both straight from
    the model. Asserts the Hard rule 11 guards on the frame it is about to
    score: no ``nova_group``, no Nutri-Score fields.
    """
    leaked = (FORBIDDEN_LEAKAGE_FIELDS | {"nova_group"}) & set(df.columns)
    assert not leaked, f"leakage field(s) in feature frame: {leaked}"
    pred = int(pipeline.predict(df)[0])
    proba = np.asarray(pipeline.predict_proba(df)[0], dtype=float)
    assert pred in (1, 2, 3, 4), f"model returned non-NOVA label: {pred}"
    return pred, proba


def shap_top_features(pipeline, df: pd.DataFrame, pred: int, k: int = 5) -> list[ShapFeature]:
    """Top-k SHAP contributions to THIS prediction's predicted class.

    Works on the transformed row (44 engineered features -- real nutrients,
    frequencies, SVD components; never PCA components, Hard rule from
    Stage 3). Class index: NovaLabelAdapter maps NOVA {1,2,3,4} -> {0,1,2,3},
    so NOVA ``pred`` lives at estimator column ``pred - 1``.
    """
    feature_pipeline = pipeline.named_steps["feature_pipeline"]
    transformed = feature_pipeline.transform(df)
    explanation = STATE["explainer"](transformed)
    values = explanation.values
    if isinstance(values, list):  # older shap returns one array per class
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    contrib = values[0, :, pred - 1] if values.ndim == 3 else values[0]

    names = STATE["feature_names"]
    order = np.argsort(-np.abs(contrib))[:k]
    return [
        ShapFeature(feature=names[i], shap_value=round(float(contrib[i]), 4))
        for i in order
    ]


def build_feature_frame(barcode: str, product: dict) -> pd.DataFrame:
    """OFF product -> normalized one-row feature frame (no ``code`` col).

    Raises nothing itself; exceptions from the client propagate to the route
    which maps them to 404/503.
    """
    raw = off_client.extract_feature_input(barcode, product)
    row = normalize_live_row(raw)  # asserts nova_group / Nutri-Score absent
    assert "nova_group" not in row, "Hard rule 11: nova_group reached the feature row"
    return pd.DataFrame([row]).drop(columns=["code"], errors="ignore")


@asynccontextmanager
async def lifespan(_: FastAPI) -> Iterator[None]:
    t0 = time.perf_counter()
    STATE["pipeline"] = load_pipeline()
    STATE["explainer"] = _build_explainer(STATE["pipeline"])
    STATE["feature_names"] = _feature_names(STATE["pipeline"])
    print(
        f"model loaded in {time.perf_counter() - t0:.1f}s "
        f"({len(STATE['feature_names'])} features, explainer ready)"
    )
    yield
    STATE.update(pipeline=None, explainer=None, feature_names=None)


app = FastAPI(
    title="Chewsy API",
    description=(
        "Barcode in -> live Open Food Facts nutrients -> MODEL-computed NOVA "
        "class (1-4) + SHAP explanation. The verdict is never read from OFF."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Stage 7's Next.js dev server is a different origin -- without this the
# browser blocks the fetch ("works with curl, fails in the browser").
_origins = [
    o.strip()
    for o in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check: 200 + whether the model finished loading."""
    return HealthResponse(
        status="ok", model_loaded=STATE["pipeline"] is not None
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Request count, error count, average /predict latency (in-memory)."""
    n = METRICS["requests"]
    avg = METRICS["latency_sum_ms"] / n if n else 0.0
    return MetricsResponse(
        requests=n, errors=METRICS["errors"], avg_latency_ms=round(avg, 2)
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Scan one barcode: live fetch -> normalize -> MODEL verdict -> SHAP.

    The model runs on every request -- never a passthrough of OFF's stored
    label, never conditioned on whether OFF already knows the NOVA class.
    """
    start = time.perf_counter()
    METRICS["requests"] += 1
    try:
        try:
            product = off_client.fetch_product(req.barcode)
        except off_client.ProductNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except off_client.OffAPIUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        df = build_feature_frame(req.barcode, product)
        pred, proba = score_row(STATE["pipeline"], df)
        top = shap_top_features(STATE["pipeline"], df, pred)
        confidence = float(proba[pred - 1])

        return PredictResponse(
            barcode=req.barcode,
            product_name=(product.get("product_name") or "").strip(),
            image_url=product.get("image_front_url") or product.get("image_url"),
            predicted_nova=pred,
            nova_label=NOVA_LABELS[pred],
            confidence=round(confidence, 4),
            class_probabilities={
                str(i + 1): round(float(p), 4) for i, p in enumerate(proba)
            },
            shap_top_features=top,
        )
    except HTTPException:
        METRICS["errors"] += 1
        raise
    except Exception:
        METRICS["errors"] += 1
        raise
    finally:
        METRICS["latency_sum_ms"] += (time.perf_counter() - start) * 1000
