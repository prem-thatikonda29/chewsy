"""Stage 6.2 -- Pydantic request/response models for the Chewsy API.

Naming is load-bearing (PRD 6.2): the response field is ``predicted_nova``
because the class is COMPUTED by the model on every scan -- it is never
fetched from Open Food Facts (Hard rule 11). ``nova_label`` is just the
human-readable name of that predicted class.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# PRD: NOVA classes 1..4
NOVA_LABELS = {
    1: "Unprocessed / minimally processed",
    2: "Processed culinary ingredient",
    3: "Processed food",
    4: "Ultra-processed food",
}


class PredictRequest(BaseModel):
    """Input to POST /predict: a barcode only.

    The API fetches the product's raw nutrients/ingredients from Open Food
    Facts and scores them with the model -- no target/label fields are ever
    requested or accepted.
    """

    barcode: str = Field(
        pattern=r"^\d{6,14}$",
        examples=["3017620422003"],
        description="Product barcode (EAN-8/12/13/UPC-A digits).",
    )


class ShapFeature(BaseModel):
    """One SHAP contribution for this prediction."""

    feature: str = Field(description="Engineered feature name (real nutrient / "
                                     "frequency / SVD component, never PCA).")
    shap_value: float = Field(
        description="SHAP contribution toward the predicted NOVA class; "
                    "positive = pushes toward it, negative = pushes away."
    )


class PredictResponse(BaseModel):
    """Model-computed NOVA verdict + explanation for one scanned product.

    ``predicted_nova`` is produced by the frozen model (models/model.joblib)
    from raw nutrients and ingredients ONLY. Open Food Facts' own stored
    ``nova_group`` is stripped in off_client and asserted absent before
    scoring -- this response never contains it, even when OFF has a label.
    """

    barcode: str
    product_name: str = ""
    image_url: str | None = None
    predicted_nova: int = Field(
        description="MODEL-COMPUTED NOVA class (1-4). Not fetched from OFF."
    )
    nova_label: str = Field(description="Human-readable name of predicted_nova.")
    confidence: float = Field(
        description="Model's probability for predicted_nova (from "
                    "predict_proba -- the model's own confidence, never OFF's)."
    )
    class_probabilities: dict[str, float] = Field(
        description='Model probability per NOVA class, keys "1".."4".'
    )
    shap_top_features: list[ShapFeature] = Field(
        description="Top SHAP features driving this specific prediction."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    """Simple in-memory counters (PRD 6.3 -- project-scope, no Prometheus)."""

    requests: int
    errors: int
    avg_latency_ms: float
