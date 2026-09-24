"""Stage 3 — tests for the feature engineering pipeline (PRD 3.10)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import (
    CategoryTagFrequencyEncoder,
    FeatureCreator,
    FrequencyEncoder,
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
        "additives_n": rng.integers(0, 10, n),
        "ingredients_n": rng.integers(1, 20, n),
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
        brand/tag, negative energy must all come out finite."""
        df = make_sample(n=40)
        df.loc[0, "energy_100g"] = np.nan
        df.loc[1, "brands"] = "brand-new-at-inference"
        df.loc[2, "categories_tags"] = "en:brand-new-tag"
        df.loc[3, "ingredients_pseudo_text"] = ""
        df.loc[4, "product_name"] = None
        df.loc[5, "sugars_100g"] = -3.0  # invalid -> skew step -> imputer
        pipe = small_pipeline()
        Z = pipe.fit_transform(df)
        assert np.isfinite(Z).all()

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
