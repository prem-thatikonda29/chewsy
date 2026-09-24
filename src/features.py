"""Stage 3 — feature engineering for Chewsy.

Evidence-backed decisions (full writeup: docs/data_quality_report.md §6):

- ``sodium_100g`` is excluded from NUMERIC_COLS: corr(salt, sodium) = 0.967
  (0.982 before the leak fix, when both columns were group-median-filled)
  with an exact 2.5x unit relation (salt = 2.5 * sodium) — pure redundancy,
  so one side is dropped (PRD 3.5). The pair PRD suggested checking,
  fat vs saturated_fat, measures only 0.626 (0.648 pre-fix) — both kept.
- energy_100g uses sqrt, NOT the PRD's suggested log1p: measured skew
  0.91 -> -1.93 with log1p (overshoot) vs 0.91 -> -0.17 with sqrt.
  sugars_100g uses log1p as planned (1.85 -> 0.56). Those numbers were
  measured before the leak fix, on the fully-filled Stage 2 frame; the
  same run prints now (NaNs retained, pandas skips them): energy
  0.83 -> -0.07 sqrt, sugars 2.18 -> 0.53 log1p — same conclusion.
- categories_tags is a comma-joined hierarchy (avg 6.8 tags/row, 5,528
  unique exact combos; the top-50 combos cover only 34% of rows), so it is
  encoded per individual tag, not per exact combo.
- PCA is intentionally NOT part of the pipeline: Stage 4's SHAP (the demo's
  "why" moment) must explain real nutrients, not uninterpretable components.
  PCA runs in run_diagnostics() instead. TruncatedSVD on TF-IDF *is* in the
  pipeline (Stage 4 Run 2 wants it).
- Every custom transformer lives in this module so joblib can pickle the
  fitted pipeline — never define them inline (the "can't get attribute
  FeatureCreator" bug).
- Frequency maps are built in fit() only. Stage 4 fits the pipeline on
  train rows, so encodings are train-only by construction — and unlike
  target encoding they never look at the target at all (PRD 3.3).
- Group-median imputation lives INSIDE the pipeline (GroupMedianImputer,
  step "impute"), fit on train rows only. It was moved out of Stage 2
  (clean.py used to fill nutrient NaNs with group medians computed on the
  full dataset before the train/test split — test rows were leaking into
  the values filled into train rows). clean.py now preserves NaNs.
- Inference robustness (beyond the PRD, needed for Stage 6): missing
  nutrients -> GroupMedianImputer (group median -> global train median)
  then the numeric branch's SimpleImputer, missing text/cats -> filled,
  unseen brand/tag -> mean training frequency, negatives -> NaN -> impute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

NUMERIC_COLS = [
    "energy_100g", "fat_100g", "saturated_fat_100g", "carbohydrates_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "salt_100g",
]  # sodium_100g dropped: exact 2.5x unit duplicate of salt_100g (corr 0.967)

# All 9 nutrient fields as clean.py defines them (NUMERIC_COLS + the
# redundant sodium side) — used only for the "all nutrients missing"
# indicator, where the redundancy is part of the signal.
NUTRIENT_COLS = NUMERIC_COLS + ["sodium_100g"]

COUNT_COLS = ["additives_n", "ingredients_n", "unknown_ingredients_n"]

INDICATOR_COLS = [
    "fiber_100g_was_missing", "sodium_100g_was_missing",
    "text_was_missing", "nutrients_all_missing",
]

RATIO_COLS = ["sugar_fiber_ratio", "sat_fat_fat_ratio"]

NUMERIC_FEATURE_COLS = NUMERIC_COLS + COUNT_COLS + INDICATOR_COLS + RATIO_COLS

BRAND_COL = "brands"
CATEGORY_COL = "categories_tags"
INGREDIENT_TEXT_COL = "ingredients_pseudo_text"
COMBINED_TEXT_COL = "text_all"

# Measured skew correction (see module docstring)
SKEW_COLS = {"energy_100g": "sqrt", "sugars_100g": "log1p"}


def _apply_skew(series: pd.Series, how: str) -> pd.Series:
    with np.errstate(invalid="ignore"):
        if how == "sqrt":
            return np.sqrt(series)
        if how == "log1p":
            return np.log1p(series)
    raise ValueError(f"unknown skew transform: {how}")


def _squeeze_column(X):
    """Module-level (picklable) — TfidfVectorizer needs a 1-D input."""
    return X.iloc[:, 0] if hasattr(X, "iloc") else np.asarray(X).ravel()


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Head step: fill categorical/text NaNs, build combined text, add ratios.

    Requires DataFrame input (column selection by name downstream depends
    on it). Missing nutrients are left for the imputer steps (GroupMedian
    imputer, then the numeric branch's SimpleImputer) — only text/
    categorical NaNs are filled here. Missingness indicators are computed
    ONLY when absent: training rows come from clean.csv which already
    carries them (captured there before any Stage 2 anomaly NaNs), while
    inference rows arrive raw from the API with no indicator columns.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()

        # --- missingness indicators (computed only if absent) -----------
        # Must run before anything fills the nutrients/text — these derive
        # from the CURRENT NaN state. When already present (clean.csv) the
        # stored values are preserved verbatim, never recomputed.
        if "fiber_100g_was_missing" not in out.columns and "fiber_100g" in out.columns:
            out["fiber_100g_was_missing"] = out["fiber_100g"].isna().astype(int)
        if "sodium_100g_was_missing" not in out.columns and "sodium_100g" in out.columns:
            out["sodium_100g_was_missing"] = out["sodium_100g"].isna().astype(int)
        if "text_was_missing" not in out.columns:
            raw_text = (
                out[INGREDIENT_TEXT_COL]
                if INGREDIENT_TEXT_COL in out.columns
                else pd.Series("", index=out.index)
            )
            out["text_was_missing"] = (
                raw_text.fillna("").astype(str).str.strip().eq("").astype(int)
            )
        if "nutrients_all_missing" not in out.columns:
            present = [c for c in NUTRIENT_COLS if c in out.columns]
            if present:
                out["nutrients_all_missing"] = out[present].isna().all(axis=1).astype(int)

        if BRAND_COL in out.columns:
            out[BRAND_COL] = out[BRAND_COL].fillna("unknown")
        if CATEGORY_COL in out.columns:
            out[CATEGORY_COL] = out[CATEGORY_COL].fillna("unknown")

        if INGREDIENT_TEXT_COL in out.columns:
            out[INGREDIENT_TEXT_COL] = (
                out[INGREDIENT_TEXT_COL].fillna("").astype(str)
            )
            ingredients = out[INGREDIENT_TEXT_COL]
        else:
            ingredients = pd.Series("", index=out.index)
        if "product_name" in out.columns:
            out[COMBINED_TEXT_COL] = (
                out["product_name"].fillna("").astype(str) + " " + ingredients
            ).str.strip()
        else:
            out[COMBINED_TEXT_COL] = ingredients

        # sugar per unit of fiber; fiber floor of 0.5 g keeps the ratio
        # finite and preserves the sugar signal for fiber-free products
        out["sugar_fiber_ratio"] = (
            out["sugars_100g"] / out["fiber_100g"].clip(lower=0.5)
        )

        # saturated share of total fat; fat==0 -> 0.0, fat missing -> NaN
        # (the branch imputer fills it), fat>0 -> normal ratio
        fat = out["fat_100g"]
        denom = fat.where(fat > 0)
        ratio = out["saturated_fat_100g"] / denom
        out["sat_fat_fat_ratio"] = ratio.mask(denom.isna() & fat.notna(), 0.0)

        return out

    def get_feature_names_out(self, input_features=None):
        base = np.asarray(input_features, dtype=object)
        extra = np.asarray([COMBINED_TEXT_COL] + RATIO_COLS, dtype=object)
        return np.concatenate([base, extra])


class SkewCorrect(BaseEstimator, TransformerMixin):
    """Per-column skew correction (transform map in SKEW_COLS).

    Runs on the DataFrame before the ColumnTransformer (which hands the
    numeric branch arrays). Values that make the transform invalid
    (e.g. negatives) become NaN and are median-imputed downstream.
    """

    def __init__(self, transforms=None):
        self.transforms = transforms  # kept unmodified for sklearn.clone

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        transforms = (
            dict(SKEW_COLS) if self.transforms is None else dict(self.transforms)
        )
        for col, how in transforms.items():
            if col in out.columns:
                out[col] = _apply_skew(out[col].astype(float), how)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Exact-value frequency encoding (single-value columns, e.g. brands).

    Maps are built in fit() only — see module docstring on train-only
    encodings (PRD 3.3). Unseen values at transform time get the mean
    training value frequency (a small nonzero, i.e. "rarer than
    anything seen").
    """

    def __init__(self, columns=None):
        self.columns = columns  # kept unmodified for sklearn.clone

    def fit(self, X, y=None):
        cols = list(X.columns) if self.columns is None else list(self.columns)
        self.columns_ = cols
        self.maps_ = {}
        self.fallbacks_ = {}
        for c in cols:
            vc = X[c].value_counts(normalize=True)
            self.maps_[c] = vc.to_dict()
            self.fallbacks_[c] = float(vc.mean()) if len(vc) else 0.0
        return self

    def transform(self, X):
        cols = []
        for c in self.columns_:
            mapped = X[c].map(self.maps_[c])
            cols.append(mapped.fillna(self.fallbacks_[c]).to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        cols = self.columns if self.columns is not None else (input_features or [])
        return np.asarray([f"{c}_freq" for c in cols], dtype=object)


class CategoryTagFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Multi-tag frequency encoding for comma-joined tag strings.

    A row like 'en:condiments,en:vinegars,en:cider-vinegars' is scored as
    the mean document-frequency of its individual tags. Exact-combo
    frequency encoding was rejected: 5,528 unique combos and the top-50
    combos cover only 34% of rows, leaving 2/3 with ~zero signal. Tag
    dictionaries are built in fit() only (train rows — PRD 3.3).
    """

    def __init__(self, columns=None):
        self.columns = columns  # kept unmodified for sklearn.clone

    def fit(self, X, y=None):
        cols = list(X.columns) if self.columns is None else list(self.columns)
        self.columns_ = cols
        self.maps_ = {}
        self.fallbacks_ = {}
        for c in cols:
            exploded = X[c].fillna("").astype(str).str.split(",").explode()
            exploded = exploded[exploded != ""]
            doc_freq = exploded.value_counts() / max(len(X), 1)
            self.maps_[c] = doc_freq.to_dict()
            self.fallbacks_[c] = float(doc_freq.mean()) if len(doc_freq) else 0.0
        return self

    def transform(self, X):
        cols = []
        for c in self.columns_:
            tag_lists = X[c].fillna("").astype(str).str.split(",")

            def score(tags, _map=self.maps_[c], _fb=self.fallbacks_[c]):
                vals = [_map[t] for t in tags if t in _map and t != ""]
                return sum(vals) / len(vals) if vals else _fb

            cols.append(tag_lists.map(score).to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        cols = self.columns if self.columns is not None else (input_features or [])
        return np.asarray([f"{c}_tagfreq" for c in cols], dtype=object)


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Train-time group-median imputation (moved from Stage 2 for leak fix).

    fit() computes, per column, medians within each categories_tags group
    plus a global median fallback — fitted on whatever rows the pipeline is
    fit on (train only). transform() fills NaNs group-first, then global.
    Unseen group at inference -> global train median.
    """

    def __init__(self, columns=None, group_col=CATEGORY_COL):
        # kept unmodified for sklearn.clone — no list()/dict() copies here
        self.columns = columns
        self.group_col = group_col

    @staticmethod
    def _group_keys(X, group_col):
        """Group labels with NaN -> "unknown" (FeatureCreator fills them,
        but a raw frame may still carry one)."""
        keys = X[group_col]
        return keys.astype(object).where(keys.notna(), "unknown")

    def fit(self, X, y=None):
        cols = list(X.columns) if self.columns is None else list(self.columns)
        # tolerate a frame missing an optional column (defensive only)
        self.columns_ = [c for c in cols if c in X.columns]
        use_group = (
            self.group_col is not None
            and self.group_col in X.columns
            and self.group_col not in self.columns_
        )
        keys = self._group_keys(X, self.group_col) if use_group else None

        self.global_medians_ = {}
        self.group_medians_ = {}
        for c in self.columns_:
            s = X[c]
            # all-NaN column -> global median is NaN on purpose: leave it
            # for the downstream SimpleImputer(keep_empty_features=True).
            self.global_medians_[c] = s.median()
            if keys is None:
                self.group_medians_[c] = {}
            else:
                # groups with no observed value for c are absent from the
                # dict -> map yields NaN -> transform falls back to global
                self.group_medians_[c] = s.groupby(keys).median().to_dict()
        return self

    def transform(self, X, y=None):
        out = X.copy()
        if not hasattr(self, "columns_"):
            raise RuntimeError("GroupMedianImputer.transform called before fit")
        use_group = bool(self.group_medians_) and any(
            bool(m) for m in self.group_medians_.values()
        )
        keys = (
            self._group_keys(X, self.group_col)
            if use_group and self.group_col in X.columns
            else None
        )
        for c in self.columns_:
            if c not in out.columns:
                continue
            na_mask = out[c].isna().to_numpy()
            if not na_mask.any():
                # no-op: never-missing column keeps its values AND dtype
                continue
            # work positionally (numpy) so duplicate/odd indexes can't
            # produce an alignment blow-up in fillna(Series)
            vals = out[c].to_numpy(dtype=float, copy=True)
            if keys is not None and self.group_medians_.get(c):
                group_vals = keys.map(self.group_medians_[c]).to_numpy(dtype=float)
                use = na_mask & ~np.isnan(group_vals)
                vals[use] = group_vals[use]
            # group median missing (unseen group / no observed values in
            # that group) -> global train median; all-NaN column -> global
            # is NaN -> left for the downstream SimpleImputer
            vals[np.isnan(vals)] = self.global_medians_[c]
            out[c] = pd.Series(vals, index=out.index, name=c)
        return out

    def get_feature_names_out(self, input_features=None):
        # one-to-one: same columns in, same columns out
        return np.asarray(input_features, dtype=object)


def _build_text_branch(n_components, max_features, min_df, max_df) -> Pipeline:
    return Pipeline([
        ("flatten", FunctionTransformer(
            _squeeze_column, validate=False, feature_names_out="one-to-one")),
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )),
        ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
    ])


def build_feature_pipeline(
    include_text: bool = True,
    include_categorical: bool = True,
    n_svd_components: int = 25,
    tfidf_max_features: int = 500,
    tfidf_min_df: int = 5,
    tfidf_max_df: float = 0.95,
) -> Pipeline:
    """Assemble the full feature pipeline (PRD 3.9).

    include_text / include_categorical exist for Stage 4's runs:
    Run 1 = nutrients + ratios only (both False), Run 2 = + text,
    Run 3 = full feature set. Pipeline input must be a DataFrame.

    Numeric branch order (PRD 3.5 + 3.6): filter-method selection
    (VarianceThreshold) on unscaled, imputed values, THEN StandardScaler.
    Skew correction happens in the head step — it is a monotone map, so it
    cannot turn a varying column constant or vice versa: the selected set
    is identical to running the filter on raw values.
    """
    transformers = [("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("variance", VarianceThreshold(threshold=1e-4)),
        ("scale", StandardScaler()),
    ]), NUMERIC_FEATURE_COLS)]

    if include_categorical:
        transformers += [
            ("brand_freq", FrequencyEncoder([BRAND_COL]), [BRAND_COL]),
            ("cat_tagfreq", CategoryTagFrequencyEncoder([CATEGORY_COL]),
             [CATEGORY_COL]),
        ]
    if include_text:
        transformers.append((
            "text",
            _build_text_branch(n_svd_components, tfidf_max_features,
                               tfidf_min_df, tfidf_max_df),
            [COMBINED_TEXT_COL],
        ))

    return Pipeline([
        ("create", FeatureCreator()),
        # impute AFTER create (needs the ratio columns + filled categories
        # it produces) and BEFORE skew (impute-then-skew, same order at
        # fit and inference). fit() sees only the rows the pipeline is fit
        # on -> train-only medians (the Stage 2 leak fix).
        ("impute", GroupMedianImputer(NUMERIC_FEATURE_COLS)),
        ("skew", SkewCorrect()),
        ("features", ColumnTransformer(transformers, remainder="drop")),
    ])


def run_diagnostics(
    csv_path: str = "data/processed/openfoodfacts_clean.csv",
    n_svd_components: int = 25,
    tfidf_min_df: int = 5,
    tfidf_max_df: float = 0.95,
) -> dict:
    """Print the Stage 3 evidence: skew (3.4), filter-method pass (3.5),
    PCA explained variance + PC1 loadings (3.8), SVD variance, coverage.

    Returns the key numbers as a dict (also used by tests).
    """
    df = pd.read_csv(csv_path)
    evidence: dict = {}

    print("=" * 70)
    print("NaNs in input: per-column null counts "
          "(imputed at train time by GroupMedianImputer)")
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        for col, n in nulls.items():
            print(f"  {col:28s} {int(n):>7d} ({100 * n / len(df):.1f}%)")
    else:
        print("  none")
    evidence["nan_counts"] = {c: int(n) for c, n in nulls.items()}

    print("=" * 70)
    print("3.4 SKEW CORRECTION EVIDENCE (before -> after)")
    for col, how in SKEW_COLS.items():
        before = float(df[col].skew())
        after = float(_apply_skew(df[col].astype(float), how).skew())
        print(f"  {col:18s} {before:+.3f} -> {after:+.3f}  ({how})")
        evidence[f"skew_{col}"] = (before, after)

    print("=" * 70)
    print("3.5 FILTER-METHOD PASS (VarianceThreshold + correlation)")
    variances = df[NUMERIC_COLS].var()
    dropped = [c for c in variances.index if float(variances[c]) < 1e-4]
    print(f"  variances: {variances.round(2).to_dict()}")
    print(f"  near-zero-variance drops (threshold=1e-4): {dropped or 'none'}")
    evidence["variance_drops"] = dropped

    corr = df[NUMERIC_COLS].corr()
    high_pairs = []
    for i, a in enumerate(NUMERIC_COLS):
        for b in NUMERIC_COLS[i + 1:]:
            if abs(corr.loc[a, b]) >= 0.9:
                high_pairs.append((a, b, round(float(corr.loc[a, b]), 3)))
    print(f"  pairs |corr| >= 0.9 among kept nutrients: {high_pairs or 'none'}")
    # the dropped column, measured against its kept twin:
    full_corr = float(df["salt_100g"].corr(df["sodium_100g"]))
    ratio = (df["salt_100g"] / df["sodium_100g"].replace(0, np.nan)).mode()
    print(f"  dropped sodium_100g vs salt_100g: corr={full_corr:.3f}, "
          f"modal ratio={float(ratio.iloc[0]) if len(ratio) else 'n/a'}")
    print(f"  kept fat vs saturated_fat: corr="
          f"{float(df['fat_100g'].corr(df['saturated_fat_100g'])):.3f} "
          f"(PRD-suggested pair — below threshold, both kept)")
    evidence["corr_salt_sodium"] = full_corr
    evidence["corr_fat_satfat"] = float(df["fat_100g"].corr(df["saturated_fat_100g"]))

    print("=" * 70)
    print("3.8 PCA ON THE SCALED NUTRIENT BLOCK (analysis only, not in pipeline)")
    # Evidence print only, NOT model input: clean.csv now (correctly)
    # retains NaNs, and StandardScaler can't take them. Median-fill here
    # just so the printout runs — the model path is the pipeline's
    # GroupMedianImputer + SimpleImputer, fit on train rows.
    pca_block = df[NUMERIC_COLS].fillna(df[NUMERIC_COLS].median())
    scaled = StandardScaler().fit_transform(pca_block)
    pca = PCA().fit(scaled)
    evr = pca.explained_variance_ratio_
    loadings = dict(zip(NUMERIC_COLS, np.round(pca.components_[0], 3)))
    print(f"  explained variance ratio: {np.round(evr, 3).tolist()}")
    print(f"  cumulative: {np.round(np.cumsum(evr), 3).tolist()}")
    print(f"  PC1 loadings: "
          f"{dict(sorted(loadings.items(), key=lambda kv: -abs(kv[1])))}")
    print(f"  PC1 explains {evr[0] * 100:.1f}% of nutrient variance")
    evidence["pca_evr"] = evr.tolist()
    evidence["pc1_loadings"] = loadings

    print("=" * 70)
    print("3.7/3.8 FULL PIPELINE + TF-IDF -> SVD")
    pipe = build_feature_pipeline(
        n_svd_components=n_svd_components, tfidf_min_df=tfidf_min_df,
        tfidf_max_df=tfidf_max_df,
    )
    pipe.fit(df)
    text_pipe = pipe.named_steps["features"].named_transformers_["text"]
    svd = text_pipe.named_steps["svd"]
    tfidf = text_pipe.named_steps["tfidf"]
    svd_evr = svd.explained_variance_ratio_
    print(f"  TF-IDF vocab size: {len(tfidf.get_feature_names_out())}")
    print(f"  SVD components: {len(svd_evr)}, "
          f"cumulative explained variance: {svd_evr.sum() * 100:.1f}%")
    print(f"  SVD EVR per component: {np.round(svd_evr, 3).tolist()}")
    evidence["tfidf_vocab"] = int(len(tfidf.get_feature_names_out()))
    evidence["svd_evr_total"] = float(svd_evr.sum())

    Z = pipe.transform(df)
    n_nonfinite = int((~np.isfinite(Z)).sum())
    print(f"  final matrix: {Z.shape[0]} rows x {Z.shape[1]} cols, "
          f"non-finite values: {n_nonfinite}")
    evidence["shape"] = tuple(Z.shape)
    evidence["nonfinite"] = n_nonfinite

    print("=" * 70)
    print("CATEGORICAL ENCODING COVERAGE (why tag-level, not exact-combo)")
    exact = df[CATEGORY_COL].value_counts(normalize=True)
    top50 = float(exact.head(50).sum())
    print(f"  unique exact combos: {len(exact)}; top-50 combos cover "
          f"{top50 * 100:.1f}% of rows")
    print(f"  -> per-tag mean-frequency encoding is used instead")
    evidence["exact_combo_top50_coverage"] = top50

    print("=" * 70)
    return evidence


if __name__ == "__main__":
    run_diagnostics()
