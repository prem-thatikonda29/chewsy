// Stage 7.10 — mirrors the FastAPI Pydantic schema (app/schemas.py)
// exactly. The frontend is a pure view of the API response: it never
// recomputes a score, band, or headline (one source of truth, PRD 7.4).

export type NovaClass = 1 | 2 | 3 | 4;

export type TrafficBand = "low" | "medium" | "high";
export type PositiveBand = "good" | "moderate" | "low";

export type TrafficKey = "sugars" | "fat" | "saturated_fat" | "salt";
export type PositiveKey = "fiber" | "proteins";

export type NutritionKey =
  | "energy"
  | "fat"
  | "saturated_fat"
  | "carbohydrates"
  | "sugars"
  | "fiber"
  | "proteins"
  | "salt";

export interface ShapFeature {
  feature: string;
  shap_value: number;
}

export interface PredictResponse {
  barcode: string;
  product_name: string;
  image_url: string | null;
  /** MODEL-COMPUTED NOVA class (1-4). Never fetched from OFF. */
  predicted_nova: NovaClass;
  nova_label: string;
  confidence: number;
  class_probabilities: Record<"1" | "2" | "3" | "4", number>;
  shap_top_features: ShapFeature[];
  nutrition_100g: Record<NutritionKey, number | null>;
  traffic_lights: Record<TrafficKey, TrafficBand | null>;
  positives: Record<PositiveKey, PositiveBand | null>;
  headline: string;
  additives_n: number;
  ingredients_n: number;
  ingredients_text: string;
}

/** Same names as the API's NOVA_LABELS — used for probability bars. */
export const NOVA_LABELS: Record<NovaClass, string> = {
  1: "Unprocessed / minimally processed",
  2: "Processed culinary ingredient",
  3: "Processed food",
  4: "Ultra-processed food",
};

export const TRAFFIC_KEYS: TrafficKey[] = [
  "sugars",
  "fat",
  "saturated_fat",
  "salt",
];

export const NUTRITION_ROWS: { key: NutritionKey; label: string; unit: string }[] = [
  { key: "energy", label: "Energy", unit: "kcal" },
  { key: "fat", label: "Fat", unit: "g" },
  { key: "saturated_fat", label: "— of which saturates", unit: "g" },
  { key: "carbohydrates", label: "Carbohydrate", unit: "g" },
  { key: "sugars", label: "— of which sugars", unit: "g" },
  { key: "fiber", label: "Fibre", unit: "g" },
  { key: "proteins", label: "Protein", unit: "g" },
  { key: "salt", label: "Salt", unit: "g" },
];
