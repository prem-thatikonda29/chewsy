"""Stage 3 — tests for the feature engineering pipeline (PRD 3.10)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import (
    INDICATOR_COLS,
    NUTRIENT_COLS,
    CategoryTagFrequencyEncoder,
    FeatureCreator,
    FrequencyEncoder,
    GroupMedianImputer,
    NUMERIC_FEATURE_COLS,
    build_feature_pipeline,
    run_diagnostics,
)


def make_sample(n=60, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    brands = rng.choice(["acme", "globex", "initech", "unknown"], n)
    cats = rng.choice(
        [
            "en:snacks,en:crackers",
            "en:beverages,en:teas",
            "en:dairies,en:milks",
            "en:condiments,en:sauces",
        ],
        n,
    )
    words = ["sugar", "salt", "flour", "oil", "cocoa", "milk", "lecithin"]
    text = [
        " ".join(rng.choice(words, rng.integers(3, 7))) for _ in range(n)
    ]
    names = rng.choice(["Choco Bar", "Herbal Tea", "Salted Chips", "Plain Milk"], n)
    # two sparse rows (25 Sep 2026): no ingredient list -> counts stay
    # NaN (clean.py's text-conditioned fill leaves them unknown), so the
    # pipeline must impute them; also keeps the counts non-degenerate.
    additives = rng.integers(0, 10, n).astype(float)
    ingredients = rng.integers(1, 20, n).astype(float)
    additives[0] = np.nan
    ingredients[1] = np.nan
    return pd.DataFrame({
        "product_name": names,
        "brands": brands,
        "categories_tags": cats,
        "ingredients_pseudo_text": text,
        "energy_100g": rng.uniform(50, 890, n),
        "fat_100g": rng.uniform(0, 80, n),
        "saturated_fat_100g": rng.uniform(0, 40, n),
        "carbohydrates_100g": rng.uniform(0, 90, n),
        "sugars_100g": rng.uniform(0, 60, n),
        "fiber_100g": rng.uniform(0, 20, n),
        "proteins_100g": rng.uniform(0, 30, n),
        "salt_100g": rng.uniform(0, 5, n),
        "sodium_100g": rng.uniform(0, 2, n),  # present in raw CSV, not a feature
        "additives_n": additives,
        "ingredients_n": ingredients,
        "unknown_ingredients_n": rng.integers(0, 5, n),
        "fiber_100g_was_missing": rng.integers(0, 2, n),
        "sodium_100g_was_missing": rng.integers(0, 2, n),
        "text_was_missing": rng.integers(0, 2, n),
        "nutrients_all_missing": rng.integers(0, 2, n),
    })


def small_pipeline(**kw):
    kw.setdefault("n_svd_components", 3)
    kw.setdefault("tfidf_min_df", 2)
    return build_feature_pipeline(**kw)


class TestFeatureCreator:
    def test_adds_ratio_columns_with_correct_values(self):
        df = pd.DataFrame({
            "sugars_100g": [10.0, 20.0, 0.0],
            "fiber_100g": [2.0, 0.0, 5.0],
            "fat_100g": [10.0, 0.0, np.nan],
            "saturated_fat_100g": [5.0, 0.0, 1.0],
        })
        out = FeatureCreator().fit_transform(df)
        # sugar_fiber_ratio: fiber floor 0.5g keeps ratios finite
        assert out["sugar_fiber_ratio"].iloc[0] == pytest.approx(5.0)
        assert out["sugar_fiber_ratio"].iloc[1] == pytest.approx(40.0)
        # sat_fat_fat_ratio: 0 when fat==0, NaN-fat -> NaN (imputed later)
        assert out["sat_fat_fat_ratio"].iloc[0] == pytest.approx(0.5)
        assert out["sat_fat_fat_ratio"].iloc[1] == 0.0
        assert np.isnan(out["sat_fat_fat_ratio"].iloc[2])

    def test_fills_text_and_categorical_nans(self):
        df = pd.DataFrame({
            "product_name": [None],
            "brands": [None],
            "categories_tags": [None],
            "ingredients_pseudo_text": [None],
            "sugars_100g": [1.0],
            "fiber_100g": [1.0],
            "fat_100g": [1.0],
            "saturated_fat_100g": [1.0],
        })
        out = FeatureCreator().fit_transform(df)
        assert out["brands"].iloc[0] == "unknown"
        assert out["categories_tags"].iloc[0] == "unknown"
        assert out["ingredients_pseudo_text"].iloc[0] == ""
        assert out["text_all"].iloc[0] == ""


    def test_computes_indicator_columns_when_absent(self):
        """Raw API rows have no `*_was_missing` columns: derive them from
        the current NaN state, before anything fills the nutrients."""
        df = make_sample(n=4).drop(columns=list(INDICATOR_COLS))
        df.loc[0, "fiber_100g"] = np.nan
        df.loc[1, "sodium_100g"] = np.nan
        df.loc[2, "ingredients_pseudo_text"] = "   "
        df.loc[3, list(NUTRIENT_COLS)] = np.nan
        out = FeatureCreator().fit_transform(df)
        assert out["fiber_100g_was_missing"].tolist() == [1, 0, 0, 1]
        assert out["sodium_100g_was_missing"].tolist() == [0, 1, 0, 1]
        assert out["text_was_missing"].tolist() == [0, 0, 1, 0]
        assert out["nutrients_all_missing"].tolist() == [0, 0, 0, 1]

    def test_preserves_indicator_columns_when_present(self):
        """Training rows come from clean.csv with the indicators already
        captured there — they must survive verbatim, never be recomputed."""
        df = make_sample(n=4)
        for col in INDICATOR_COLS:
            df[col] = 7
        out = FeatureCreator().fit_transform(df)
        for col in INDICATOR_COLS:
            assert (out[col] == 7).all(), f"{col} was recomputed"


class TestGroupMedianImputer:
    def test_fills_from_group_median_only(self):
        """NaN in a group with observed [10, 12] -> 11; a different group's
        much larger values must not influence it."""
        df = pd.DataFrame({
            "categories_tags": ["g1", "g1", "g1", "g2"],
            "col_a": [10.0, 12.0, np.nan, 100.0],
        })
        out = GroupMedianImputer(["col_a"]).fit_transform(df)
        assert out["col_a"].iloc[2] == pytest.approx(11.0)
        # observed values untouched
        np.testing.assert_allclose(
            out["col_a"].to_numpy(), [10.0, 12.0, 11.0, 100.0]
        )

    def test_unseen_group_at_transform_uses_global_train_median(self):
        fit_df = pd.DataFrame({
            "categories_tags": ["g1", "g1", "g2", "g2"],
            "col_a": [10.0, 12.0, 100.0, 200.0],
        })
        imp = GroupMedianImputer(["col_a"]).fit(fit_df)
        # global train median = median(10, 12, 100, 200) = 56.0
        new = pd.DataFrame({"categories_tags": ["g3"], "col_a": [np.nan]})
        out = imp.transform(new)
        assert out["col_a"].iloc[0] == pytest.approx(56.0)

    def test_group_with_no_observed_values_falls_back_to_global(self):
        df = pd.DataFrame({
            "categories_tags": ["g1", "g1", "g2", "g2"],
            "col_a": [10.0, 12.0, np.nan, np.nan],
        })
        out = GroupMedianImputer(["col_a"]).fit_transform(df)
        # g2 has no observed value for col_a -> global median = 11.0
        assert out["col_a"].iloc[2] == pytest.approx(11.0)
        assert out["col_a"].iloc[3] == pytest.approx(11.0)

    def test_all_nan_column_left_for_downstream_simple_imputer(self):
        df = pd.DataFrame({
            "categories_tags": ["g1", "g2"],
            "col_a": [np.nan, np.nan],
        })
        out = GroupMedianImputer(["col_a"]).fit_transform(df)
        # global median of an all-NaN column is NaN -> SimpleImputer's job
        assert out["col_a"].isna().all()

    def test_never_missing_column_is_a_noop(self):
        df = pd.DataFrame({
            "categories_tags": ["g1", "g2"],
            "col_a": [1.0, 2.0],
            "col_b": [3, 4],
        })
        out = GroupMedianImputer(["col_a", "col_b"]).fit_transform(df)
        np.testing.assert_allclose(out["col_a"].to_numpy(), [1.0, 2.0])
        np.testing.assert_allclose(out["col_b"].to_numpy(), [3, 4])

    def test_get_feature_names_out_is_one_to_one(self):
        cols = ["col_a", "col_b"]
        names = GroupMedianImputer(cols).fit(
            pd.DataFrame({"categories_tags": ["g1"], "col_a": [1.0], "col_b": [2.0]})
        ).get_feature_names_out(cols)
        assert list(names) == cols


class TestFrequencyEncoders:
    def test_brand_frequencies_and_unseen_fallback(self):
        df = pd.DataFrame({"brands": ["a", "a", "a", "b"]})
        enc = FrequencyEncoder(["brands"]).fit(df)
        out = enc.transform(pd.DataFrame({"brands": ["a", "b", "zzz"]}))
        assert out[0, 0] == pytest.approx(0.75)
        assert out[1, 0] == pytest.approx(0.25)
        # unseen brand -> mean training frequency
        assert out[2, 0] == pytest.approx(np.mean([0.75, 0.25]))

    def test_tag_level_encoding_averages_member_tags(self):
        df = pd.DataFrame({"categories_tags": [
            "en:snacks,en:chips",
            "en:snacks",
            "en:milks",
        ]})
        enc = CategoryTagFrequencyEncoder(["categories_tags"]).fit(df)
        # doc frequencies: snacks=2/3, chips=1/3, milks=1/3
        out = enc.transform(pd.DataFrame({"categories_tags": [
            "en:snacks,en:chips",   # mean(2/3, 1/3) = 0.5
            "en:milks",             # 1/3
            "en:unseen-tag",        # fallback = mean of tag freqs
        ]}))
        assert out[0, 0] == pytest.approx(0.5)
        assert out[1, 0] == pytest.approx(1 / 3)
        assert out[2, 0] == pytest.approx(np.mean([2 / 3, 1 / 3, 1 / 3]))


class TestPipeline:
    def test_numeric_only_shape(self):
        pipe = small_pipeline(include_text=False, include_categorical=False)
        Z = pipe.fit_transform(make_sample())
        assert Z.shape == (60, len(NUMERIC_FEATURE_COLS))
        assert np.isfinite(Z).all()

    def test_full_pipeline_shape(self):
        pipe = small_pipeline()
        Z = pipe.fit_transform(make_sample())
        # 17 numeric + 1 brand + 1 category + 3 SVD components
        assert Z.shape == (60, len(NUMERIC_FEATURE_COLS) + 2 + 3)
        assert np.isfinite(Z).all()

    def test_text_branch_can_be_toggled_for_stage4_runs(self):
        with_text = small_pipeline(include_text=True)
        without_text = small_pipeline(include_text=False)
        n_with = with_text.fit_transform(make_sample()).shape[1]
        n_without = without_text.fit_transform(make_sample()).shape[1]
        assert n_with - n_without == 3  # SVD component count

    def test_survives_missing_unseen_and_invalid_inference_input(self):
        """Stage 6 sends raw API rows: NaN nutrients, empty text, unseen
        brand/tag, negative energy, NaN counts (sparse OFF entry) must all
        come out finite."""
        df = make_sample(n=40)
        df.loc[0, "energy_100g"] = np.nan
        df.loc[1, "brands"] = "brand-new-at-inference"
        df.loc[2, "categories_tags"] = "en:brand-new-tag"
        df.loc[3, "ingredients_pseudo_text"] = ""
        df.loc[4, "product_name"] = None
        df.loc[5, "sugars_100g"] = -3.0  # invalid -> skew step -> imputer
        # sparse entry: no ingredient list -> counts unknown (NaN, not 0)
        df.loc[6, ["additives_n", "ingredients_n", "unknown_ingredients_n"]] = np.nan
        pipe = small_pipeline()
        Z = pipe.fit_transform(df)
        assert np.isfinite(Z).all()

    def test_group_median_imputer_makes_shared_group_nans_finite(self):
        """Nutrient NaNs concentrated in one categories_tags group: the
        GroupMedianImputer step (not just the global fallback) must turn
        them into a finite matrix."""
        df = make_sample(n=40)
        shared = df["categories_tags"].iloc[0]
        idx = df.index[df["categories_tags"] == shared][:6]
        df.loc[idx, "fiber_100g"] = np.nan
        df.loc[idx[:3], "energy_100g"] = np.nan
        df.loc[idx[3], "sugars_100g"] = np.nan
        pipe = small_pipeline()
        Z = pipe.fit_transform(df)
        assert np.isfinite(Z).all()
        # the step is really in the pipeline and was fit on these rows
        imputer = pipe.named_steps["impute"]
        assert "en:" in str(set(imputer.group_medians_["fiber_100g"]))
        assert np.isfinite(imputer.global_medians_["fiber_100g"])

    def test_joblib_roundtrip_reproduces_transform(self):
        """Guards the Stage 5 'can't get attribute FeatureCreator' bug."""
        pipe = small_pipeline()
        df = make_sample()
        Z1 = pipe.fit_transform(df)
        joblib.dump(pipe, "/tmp/chewsy_test_pipeline.joblib")
        loaded = joblib.load("/tmp/chewsy_test_pipeline.joblib")
        Z2 = loaded.transform(df)
        np.testing.assert_allclose(Z1, Z2)

    def test_feature_names_available_for_shap(self):
        pipe = small_pipeline()
        pipe.fit(make_sample())
        names = pipe.named_steps["features"].get_feature_names_out()
        assert len(names) == len(NUMERIC_FEATURE_COLS) + 2 + 3
        assert "num__energy_100g" in names
        assert "brand_freq__brands_freq" in names


class TestDiagnostics:
    def test_diagnostics_runs_and_returns_evidence(self, tmp_path):
        csv = tmp_path / "sample.csv"
        make_sample(n=50).to_csv(csv, index=False)
        evidence = run_diagnostics(
            csv_path=str(csv), n_svd_components=3,
            tfidf_min_df=2, tfidf_max_df=1.0,
        )
        for key in [
            "skew_energy_100g", "skew_sugars_100g", "variance_drops",
            "corr_salt_sodium", "corr_fat_satfat", "pca_evr",
            "pc1_loadings", "svd_evr_total", "shape", "nonfinite",
            "exact_combo_top50_coverage", "tfidf_vocab",
        ]:
            assert key in evidence, f"missing evidence key: {key}"
        assert evidence["nonfinite"] == 0
        assert len(evidence["pca_evr"]) == 8
