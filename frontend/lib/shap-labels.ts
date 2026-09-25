// PRD 7.5 — SHAP feature names are engineered column names
// (num__additives_n, text__truncatedsvd9, ...); the chart shows humans
// what the model actually reads, so they get humanized labels.

const NUM_LABELS: Record<string, string> = {
  energy_100g: "Energy (kcal/100g)",
  fat_100g: "Fat",
  saturated_fat_100g: "Saturated fat",
  carbohydrates_100g: "Carbohydrates",
  sugars_100g: "Sugars",
  fiber_100g: "Fibre",
  proteins_100g: "Protein",
  salt_100g: "Salt",
  sodium_100g: "Sodium",
  additives_n: "Number of additives",
  ingredients_n: "Number of ingredients",
  unknown_ingredients_n: "Unrecognised ingredients (count)",
  fiber_100g_was_missing: "Fibre not reported",
  sodium_100g_was_missing: "Sodium not reported",
  text_was_missing: "No ingredient text",
  nutrients_all_missing: "No nutrients reported",
  sugar_fiber_ratio: "Sugar-to-fibre ratio",
  sat_fat_fat_ratio: "Saturated-to-total fat ratio",
};

const SVD_RE = /^text__truncatedsvd(\d+)$/;

export function humanizeFeature(name: string): string {
  const num = name.startsWith("num__") ? name.slice(5) : name;
  if (NUM_LABELS[num]) return NUM_LABELS[num];

  const svd = name.match(SVD_RE);
  if (svd) return `Ingredient-text signal #${svd[1]}`;

  if (name.startsWith("brand_freq__")) return "Brand commonness (training data)";
  if (name.startsWith("cat_tagfreq__")) return "Category commonness (training data)";

  // fallback: strip a prefix, spaces for underscores
  const bare = name.replace(/^[a-z_]+__/, "");
  return bare
    .replace(/_100g$/, "")
    .replace(/_/g, " ")
    .replace(/^./, (c) => c.toUpperCase());
}
