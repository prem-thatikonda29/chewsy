"""Stage 6.2 -- Pydantic request/response models for the Chewsy API.

Naming is load-bearing (PRD 6.2): the response field is ``predicted_nova``
because the class is COMPUTED by the model on every scan -- it is never
fetched from Open Food Facts (Hard rule 11). ``nova_label`` is just the
human-readable name of that predicted class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# PRD: NOVA classes 1..4
NOVA_LABELS = {
    1: "Unprocessed / minimally processed",
    2: "Processed culinary ingredient",
    3: "Processed food",
    4: "Ultra-processed food",
}

# Stage 6.7 -- band vocabularies (source of truth for the values lives in
# src/nutrition_flags.py; these Literals just make the schema self-document)
TrafficBand = Literal["low", "medium", "high"]
PositiveBand = Literal["good", "moderate", "low"]


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
    data_sparse: bool = Field(
        description="True when the OFF record is a STUB -- no ingredient "
                    "list, no category tags, counts unpublished -- so the "
                    "row the model scored was thin. The verdict is still "
                    "MODEL-computed either way; the UI must present it as "
                    "low-information when this is true (never shout a "
                    "confidence the input cannot support)."
    )
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

    # --- Stage 6.7: the "what's in it" axis + facts the result screen needs.
    # Computed LOCALLY from the normalized row already in hand (Hard rule 12:
    # threshold lookups on nutrition_100g, never fetched OFF grades) --
    # zero extra API calls, no Nutri-Score family anywhere in this model.
    nutrition_100g: dict[str, float | None] = Field(
        description="Per-100g facts panel: energy (kcal), fat, "
                    "saturated_fat, carbohydrates, sugars, fiber, proteins, "
                    "salt -- Stage 2-normalized values; null = not published "
                    "(frontend renders null as '—', never a fabricated number)."
    )
    traffic_lights: dict[str, TrafficBand | None] = Field(
        description="NHS nutrient traffic lights for sugars/fat/"
                    "saturated_fat/salt: 'low'|'medium'|'high', null when the "
                    "nutrient is missing (band never guessed). Drinks use the "
                    "per-100ml thresholds. Computed locally -- not OFF grades."
    )
    positives: dict[str, PositiveBand | None] = Field(
        description="CUSTOM fibre/protein bands ('good'|'moderate'|'low', "
                    "null when missing) -- these are NOT traffic lights; the "
                    "FSA scheme has no official thresholds for them."
    )
    headline: str = Field(
        description="Combined two-axis one-liner (processing x nutrients). "
                    "NOVA is a descriptor, never a verdict: the headline "
                    "never calls a class good/bad, and states 'nutrition data "
                    "unavailable' when no nutrients are published."
    )
    additives_n: int = Field(
        description="Number of additives reported on the package."
    )
    ingredients_n: int = Field(
        description="Number of ingredients reported on the package."
    )
    ingredients_text: str = Field(
        description="Full ingredient list as printed on the package (raw OFF "
                    "text, HTML-unescaped) -- what's actually in it."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    """Simple in-memory counters (PRD 6.3 -- project-scope, no Prometheus).

    Stage 12.1 adds per-class prediction counts and a sparse-scan counter
    so the endpoint can back the monitoring story, not just traffic totals.
    """

    requests: int
    errors: int
    avg_latency_ms: float
    per_class: dict[str, int]
    sparse: int
