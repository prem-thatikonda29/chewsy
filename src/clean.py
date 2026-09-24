"""
clean.py

Stage 2 -- Data Verification & Cleaning.

Loads the raw live-API training pull, prints a diagnostic report (PRD 2.1
checklist plus independent consistency analysis: macro-sum sanity,
energy-vs-Atwater cross-check, near-duplicate names, brand/text hygiene),
applies a documented drop/cap decision to every flagged anomaly,
and writes data/processed/openfoodfacts_clean.csv for DVC tracking.

Stage 2 is deliberately ROW-LOCAL ONLY. Group-median imputation of nutrient
columns was REMOVED from here (it used to run on the full dataset before any
train/test split -- a train/test leakage: test rows influenced the medians
filled into train rows). All learned imputation now lives inside the sklearn
pipeline in src/features.py (GroupMedianImputer), fit on train rows only.
This script therefore INTENTIONALLY retains NaNs in nutrient columns; the
`*_was_missing` indicator columns still record the original missingness.

Usage:
    python src/clean.py
"""

import html
import os

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/openfoodfacts_training_set.csv"
OUT_PATH = "data/processed/openfoodfacts_clean.csv"

# Mass-basis nutrient columns: values must live in [0, 100] g per 100 g.
# (energy_100g is kcal/100g -- a different unit, bounded by ~900 kcal for
# pure fat, not by 100.)
MASS_COLS = [
    "fat_100g",
    "saturated_fat_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fiber_100g",
    "proteins_100g",
    "salt_100g",
    "sodium_100g",
]
ENERGY_COL = "energy_100g"
MAX_ENERGY_KCAL = 900.0   # pure fat is ~900 kcal/100g; above = impossible
MACRO_COLS = ["fat_100g", "carbohydrates_100g", "proteins_100g", "salt_100g"]
NUTRIENT_COLS = [ENERGY_COL] + MASS_COLS

# Atwater cross-check bounds: kcal / (4*carbs + 4*protein + 9*fat + 2*fiber).
# Real foods sit near 1.0; anything outside [0.3, 3.0] means one of the two
# independently-reported fields is wrong (measured in analysis: 41 rows).
ENERGY_RATIO_LO, ENERGY_RATIO_HI = 0.3, 3.0

# Leakage guard -- same fields fetch_training_set.py refuses to request, but
# asserted here too (PRD 2.5: don't just trust the fetch script). Nutri-Score
# is a computed label sitting next to the target, not a legitimate predictor.
# Old bulk export called it nutrition_grade_fr; these are the live-API names.
FORBIDDEN_LEAKAGE_FIELDS = {
    "nutrition_grades",
    "nutriscore_grade",
    "nutriscore_score",
    "nutriscore_data",
    "nutrition_grade_fr",
}

# Indicator columns PRD 2.3 requires -- missingness itself is informative
# (less-documented, often smaller-brand products under-report nutrients).
# text_was_missing is an analysis-driven addition: 1278 rows have no
# ingredient text and 1228 of those are NOVA 2 (oils/sugars/salts whose
# ingredient list is trivial/omitted) -- that missingness is real signal.
INDICATOR_COLS = ["fiber_100g", "sodium_100g"]
TEXT_COL = "ingredients_pseudo_text"


def report(df: pd.DataFrame) -> None:
    """PRD 2.1: printed diagnostic report before any cleaning decision."""
    print("=" * 70)
    print("DIAGNOSTIC REPORT (raw)")
    print("=" * 70)
    print(f"rows: {len(df)}  cols: {len(df.columns)}")
    print(f"duplicate code (barcode): {int(df['code'].duplicated().sum())}")

    print("\nnull % per column:")
    null_pct = (df.isna().mean() * 100).round(1)
    for col, pct in null_pct.items():
        print(f"  {col:28s} {pct:5.1f}%")

    print("\nnova_group value counts:")
    print(df["nova_group"].value_counts().sort_index().to_string())

    print("\n*_100g value ranges:")
    for col in MASS_COLS:
        s = df[col]
        print(
            f"  {col:22s} min={s.min():>8.2f} max={s.max():>8.2f} "
            f"neg={int((s < 0).sum()):>4d} over100={int((s > 100).sum()):>4d}"
        )
    # energy is kcal/100g, not g/100g -- the 0..100 bound does not apply,
    # so it gets its own line with the correct physical bound (~900 kcal).
    e = df[ENERGY_COL]
    print(
        f"  {ENERGY_COL:22s} min={e.min():>8.2f} max={e.max():>8.2f} "
        f"kcal_over_{MAX_ENERGY_KCAL:.0f}={int((e > MAX_ENERGY_KCAL).sum()):>4d}"
    )

    # ---- anomaly flags (each gets a documented decision below) ----
    print("\nFLAGGED ANOMALIES:")
    neg = {c: int((df[c] < 0).sum()) for c in MASS_COLS if (df[c] < 0).any()}
    over = {c: int((df[c] > 100).sum()) for c in MASS_COLS if (df[c] > 100).any()}
    hot = int((df[ENERGY_COL] > MAX_ENERGY_KCAL).sum())
    sat_gt_fat = int(
        ((df["saturated_fat_100g"] > df["fat_100g"]) & df["saturated_fat_100g"].notna()).sum()
    )
    print(f"  negative mass values      : {neg or 'none'}")
    print(f"  mass values > 100g        : {over or 'none'}")
    print(f"  energy_100g > {MAX_ENERGY_KCAL:.0f} kcal       : {hot}")
    print(f"  saturated_fat > fat       : {sat_gt_fat}")

    # independent consistency analysis (beyond the PRD checklist)
    macro_sum = df[MACRO_COLS].sum(axis=1, min_count=1)
    over100 = int((macro_sum > 100).sum())
    print(f"  macro sum (fat+carb+pro+salt) > 100 : {over100} "
          f"(most are <=102 rounding slack; OFF counts fiber inside carbs, "
          f"so fiber is excluded here to avoid double-counting)")

    est = (
        4 * df["carbohydrates_100g"] + 4 * df["proteins_100g"]
        + 9 * df["fat_100g"] + 2 * df["fiber_100g"].fillna(0)
    )
    ratio = df[ENERGY_COL] / est.replace(0, np.nan)
    zero_energy = int(((df[ENERGY_COL] == 0) & (est >= 20)).sum())
    inconsistent = int(
        (df[ENERGY_COL].notna() & ((ratio > ENERGY_RATIO_HI) | (ratio < ENERGY_RATIO_LO))).sum()
    )
    print(f"  energy==0 despite >=20kcal of macros : {zero_energy}")
    print(f"  energy/Atwater outside [{ENERGY_RATIO_LO}, {ENERGY_RATIO_HI}] : {inconsistent}")

    named = df[df["product_name"].fillna("").str.strip() != ""]
    dup_named = named[named.duplicated(subset=["product_name", "brands"], keep=False)]
    if len(dup_named):
        groups = dup_named.groupby(["product_name", "brands"], dropna=False)
        max_size = int(groups.size().max())
        conflicts = int((groups["nova_group"].nunique() > 1).sum())
        print(
            f"  near-dup name+brand rows (diff barcode) : {len(dup_named)} in "
            f"{groups.ngroups} groups (max {max_size}/group, {conflicts} label conflicts) "
            f"-> keep: generic names + distinct barcodes, not duplicate records"
        )
    else:
        print("  near-dup name+brand rows (diff barcode) : none")

    all_missing = int(df[NUTRIENT_COLS].isna().all(axis=1).sum())
    print(f"  rows with ALL nutrients null   : {all_missing} "
          f"(no nutrient report at all -- rows kept + flagged via "
          f"nutrients_all_missing; NaNs retained, see clean())")

    brands = df["brands"].fillna("")
    print(f"  brands stored as list-repr ['X'] : {int(brands.str.startswith('[').sum())} rows")
    text = df[TEXT_COL].fillna("")
    print(f"  empty ingredient text           : {int(text.str.strip().eq('').sum())} rows")
    print(
        f"  product_name w/ HTML entities   : "
        f"{int(df['product_name'].fillna('').str.contains(r'&[a-z]+;|&#\\d+;').sum())} rows"
    )
    print(
        "\n  decisions -> negatives: NaN (imputed at train time) | >100g: cap at 100 | "
        "energy bad: NaN (imputed at train time) | sat>fat: NaN (imputed at "
        "train time) | dup code: drop keep-first | "
        "near-dup names: keep | brands: un-list + lowercase | "
        "NO group-median fill here -- NaNs are preserved for the Stage 3 "
        "GroupMedianImputer (fit on train rows only)"
    )
    print()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # PRD 2.5 -- leakage check as an actual code assertion (see module docstring).
    leaked = FORBIDDEN_LEAKAGE_FIELDS & set(df.columns)
    assert not leaked, f"Nutri-Score leakage field(s) in dataset: {leaked}"

    # PRD 2.2 -- drop exact-duplicate barcodes, keep first.
    n_before = len(df)
    df = df.drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    print(f"dropped {n_before - len(df)} duplicate-code rows")

    # --- missingness indicators (PRD 2.3) -------------------------------
    # Captured FIRST, on the original nulls: the indicator is about what the
    # source did/didn't report, so it must be recorded before any cleaning
    # step below can introduce new NaNs (a capped/invalidated value is not
    # the same signal as "never reported").
    for col in INDICATOR_COLS:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)
    df["text_was_missing"] = df[TEXT_COL].fillna("").str.strip().eq("").astype(int)
    # Analysis-driven: ~17% of rows report NO nutrient at all (class-skewed,
    # NOVA 4 the least represented). Decision: KEEP the rows (they still carry
    # ingredients/brand text and a valid label) but flag them: every nutrient
    # on such a row is filled at train time (group median -> global median ->
    # SimpleImputer zero), i.e. synthetic, and the model should be able to
    # tell a measured value from an imputed one. Also explains why
    # post-imputation macro sums can exceed 100 -- the per-column medians are
    # drawn independently, so their sum is not constrained to <= 100.
    df["nutrients_all_missing"] = df[NUTRIENT_COLS].isna().all(axis=1).astype(int)

    # --- categorical hygiene (analysis-driven) --------------------------
    # brands arrives as a Python list-repr ("['Andros']") on 5006/6000 rows
    # because the fetch script joined a list field. Decision: parse to a
    # plain name (first entry) so Stage 3 frequency encoding groups real
    # duplicates instead of treating the brackets as part of the brand.
    # Lowercasing merges case variants (3035 -> ~2897 unique).
    # (.map, not html.unescape(series) -- the latter silently no-ops because
    # html.unescape checks "'&' in s", which on a Series tests the INDEX.)
    b = df["brands"].fillna("").str.strip()
    b = b.str.strip("[]").str.split(r"',\s*'").str[0].str.strip("'\"")
    df["brands"] = (
        b.map(html.unescape).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    df.loc[df["brands"] == "", "brands"] = "unknown"
    # API sentinel strings ("null", "none", "nan") are not brand names --
    # they'd otherwise survive the CSV round-trip as literal tokens (and
    # pandas re-reads "null" as NaN, which is exactly what we're avoiding).
    df.loc[df["brands"].isin({"null", "none", "nan", "n/a"}), "brands"] = "unknown"

    # product_name: 28 rows carry HTML entities (&quot; etc.) from the API;
    # 16 are blank/whitespace. Decision: unescape + strip, blanks -> "".
    name = df["product_name"].fillna("").map(html.unescape).str.strip()
    df["product_name"] = name

    # --- documented anomaly decisions (PRD 2.1) -------------------------
    # Negative mass is physically impossible (OFF data-entry / sign error):
    # decision = set to NaN so it flows into the normal imputation path
    # at train time rather than poisoning medians with a bogus -1.
    for col in MASS_COLS:
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            print(f"{col}: {n_neg} negative value(s) -> NaN (kept NaN; imputed at train time)")
            df.loc[df[col] < 0, col] = np.nan

    # Mass-basis value above 100 g/100g: decision = CAP at 100, not drop.
    # Observed only marginally over (salt_100g 105 -- a Himalayan-salt row,
    # almost certainly a unit/rounding slip on an otherwise valid, labeled
    # row); dropping the row would throw away the NOVA label for one
    # fat-fingered number.
    for col in MASS_COLS:
        n_over = int((df[col] > 100).sum())
        if n_over:
            print(f"{col}: {n_over} value(s) > 100g -> capped at 100")
            df.loc[df[col] > 100, col] = 100.0

    # Energy above ~900 kcal/100g exceeds the physical maximum for any food
    # (pure fat): decision = NaN (kept NaN; imputed at train time), same
    # rationale as negatives.
    n_hot = int((df[ENERGY_COL] > MAX_ENERGY_KCAL).sum())
    if n_hot:
        print(f"{ENERGY_COL}: {n_hot} value(s) > {MAX_ENERGY_KCAL:.0f} kcal -> NaN (kept NaN; imputed at train time)")
        df.loc[df[ENERGY_COL] > MAX_ENERGY_KCAL, ENERGY_COL] = np.nan

    # saturated fat cannot exceed total fat: decision = NaN the sat-fat side
    # only (fat_100g itself is the more trustworthy field) -- a single-cell,
    # row-local fix. The NaN is left in place for train-time imputation.
    bad = (df["saturated_fat_100g"] > df["fat_100g"]) & df["saturated_fat_100g"].notna()
    if bad.any():
        print(f"saturated_fat_100g: {int(bad.sum())} row(s) > fat_100g -> NaN (kept NaN; imputed at train time)")
        df.loc[bad, "saturated_fat_100g"] = np.nan

    # --- energy cross-consistency (analysis-driven) ---------------------
    # Rule 1: energy_100g == 0 while macros imply >= 20 kcal (e.g. a row
    # with 66.8 g carbs + 12.4 g protein reported 0 kcal) -- energy was
    # simply never entered. Decision: NaN (imputed at train time).
    est = (
        4 * df["carbohydrates_100g"] + 4 * df["proteins_100g"]
        + 9 * df["fat_100g"] + 2 * df["fiber_100g"].fillna(0)
    )
    dead = (df[ENERGY_COL] == 0) & (est >= 20) & df[ENERGY_COL].notna()
    if dead.any():
        print(f"{ENERGY_COL}: {int(dead.sum())} zero(s) contradicting macros -> NaN (kept NaN; imputed at train time)")
        df.loc[dead, ENERGY_COL] = np.nan
    # Rule 2: reported energy vs Atwater estimate outside [0.3, 3.0] --
    # one of the two independently-reported fields is wrong and we cannot
    # tell which, so neither is trusted. Decision: NaN (imputed at train time).
    ratio = df[ENERGY_COL] / est.replace(0, np.nan)
    wild = df[ENERGY_COL].notna() & ((ratio > ENERGY_RATIO_HI) | (ratio < ENERGY_RATIO_LO))
    if wild.any():
        print(
            f"{ENERGY_COL}: {int(wild.sum())} value(s) inconsistent with macros "
            f"(ratio outside [{ENERGY_RATIO_LO}, {ENERGY_RATIO_HI}]) -> NaN "
            f"(kept NaN; imputed at train time)"
        )
        df.loc[wild, ENERGY_COL] = np.nan

    # NOTE: no imputation here. Missing nutrient cells are left as NaN on
    # purpose -- see module docstring. Group-median imputation was moved to
    # src/features.py::GroupMedianImputer so it is fit on train rows only.

    # Count/int columns: fill with 0 (absence of an additive count reads
    # as "no additives reported" -> same as none) + keep Stage 3 numeric.
    # Row-local and constant -- no statistics are learned from other rows,
    # so this is not a leakage source and stays in Stage 2.
    for col in ["additives_n", "ingredients_n", "unknown_ingredients_n"]:
        df[col] = df[col].fillna(0)

    # --- text normalization (PRD 2.4) ----------------------------------
    # Lowercase, strip punctuation noise, collapse whitespace, unescape any
    # HTML entities. Deliberately NOT removing stopwords -- TF-IDF in
    # Stage 3 does that via stop_words='english'; doing it here would
    # double-process the text. Known limitation, documented for Stage 3:
    # ~80 rows mix non-English ingredient tokens, so an english-only
    # stop word list only partially applies -- acceptable for a demo model.
    text = (
        df[TEXT_COL]
        .fillna("")
        .map(html.unescape)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df[TEXT_COL] = text

    df["categories_tags"] = df["categories_tags"].fillna("").str.strip()

    # Re-run the leakage guard on the final frame: assert it's absent from
    # the clean CSV itself, not just the raw pull (PRD 2.5).
    leaked = FORBIDDEN_LEAKAGE_FIELDS & set(df.columns)
    assert not leaked, f"Nutri-Score leakage field(s) after cleaning: {leaked}"

    return df


def main() -> None:
    df = pd.read_csv(RAW_PATH)
    report(df)
    df = clean(df)

    print("\nPOST-CLEAN SUMMARY:")
    print(f"rows: {len(df)}")
    print("nova_group balance:", df["nova_group"].value_counts().sort_index().to_dict())
    macro_sum = df[MACRO_COLS].sum(axis=1, min_count=1)
    print(f"macro sums > 100: {int((macro_sum > 100).sum())} (rounding slack only)")

    # Honest null accounting: NaNs in nutrient columns are RETAINED here on
    # purpose. They are imputed at train time by features.GroupMedianImputer
    # (fit on train rows only) -- never in this Stage 2 pre-split step.
    print("\nNaNs retained at write (imputed at train time by GroupMedianImputer):")
    nulls = df.isna().sum()
    print(f"  {'column':28s} {'nulls':>7s} {'pct':>7s}")
    for col, n in nulls.items():
        print(f"  {col:28s} {int(n):>7d} {100 * n / len(df):>6.1f}%")
    print(f"  {'TOTAL':28s} {int(nulls.sum()):>7d}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"written -> {OUT_PATH}")


if __name__ == "__main__":
    main()
