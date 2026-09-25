"""Stage 6.7 — unit tests for src/nutrition_flags.py (PRD 6.7).

Pure-function tests: threshold edges, the four headline quadrants (incl.
the class-3 wording refinement and the all-null fallback), beverage
detection, and the custom positives bands — no API, model, or network.
The API-level cases live in tests/test_api.py::TestStage67TwoAxisResponse.
"""

import pytest

from src.nutrition_flags import (
    NUTRITION_KEYS,
    build_scan_facts,
    headline,
    is_beverage,
    nutrition_100g,
    positives,
    traffic_band,
    traffic_lights,
)

NO_REDS = {"sugars": "low", "fat": "low", "saturated_fat": "low", "salt": "low"}
ALL_REDS = {"sugars": "high", "fat": "high", "saturated_fat": "high", "salt": "high"}
ALL_NULL = {"sugars": None, "fat": None, "saturated_fat": None, "salt": None}


class TestHeadlineQuadrants:
    """Processing (NOVA <= 2 vs >= 3) x nutrients (any red vs none)."""

    @pytest.mark.parametrize(
        "nova,lights,expected",
        [
            # Low processing, no reds
            (1, NO_REDS, "Minimally processed and nutritionally solid"),
            (2, NO_REDS, "Minimally processed and nutritionally solid"),
            # Low processing, has reds
            (1, ALL_REDS, "Minimally processed, but high in sugar, fat, saturated fat and salt"),
            (2, ALL_REDS, "Minimally processed, but high in sugar, fat, saturated fat and salt"),
            # High processing, no reds — class-3 wording refinement
            (3, NO_REDS, "Processed, but nutritionally decent"),
            (4, NO_REDS, "Ultra-processed, but nutritionally decent"),
            # High processing, has reds — class 3 is NOT ultra-processed
            (3, {"sugars": "low", "fat": "low", "saturated_fat": "low", "salt": "high"},
             "Processed and high in salt"),
            (4, {"sugars": "low", "fat": "low", "saturated_fat": "low", "salt": "high"},
             "Ultra-processed and high in salt"),
        ],
    )
    def test_quadrant_table(self, nova, lights, expected):
        assert headline(nova, lights) == expected

    def test_red_noun_joining(self):
        two = {"sugars": "high", "fat": "high", "saturated_fat": "low", "salt": "low"}
        assert headline(4, two) == "Ultra-processed and high in sugar and fat"

    def test_all_null_falls_back_to_processing_only(self):
        # ~17% of training rows publish no nutrients at all — the headline
        # must say so, never imply the product is fine
        for nova, phrase in [
            (1, "Minimally processed"),
            (3, "Processed"),
            (4, "Ultra-processed"),
        ]:
            assert headline(nova, ALL_NULL) == f"{phrase} \u2014 nutrition data unavailable"

    def test_partial_nulls_do_not_trigger_the_fallback(self):
        partial = dict(NO_REDS, sugars=None)
        assert headline(4, partial) == "Ultra-processed, but nutritionally decent"
        # a null band is never counted as a red either
        reds = dict(NO_REDS, sugars=None, salt="high")
        assert headline(4, reds) == "Ultra-processed and high in salt"

    def test_headline_is_never_a_verdict(self):
        # binding design principle (PRD section 1): descriptor, not verdict
        for nova in (1, 2, 3, 4):
            for lights in (NO_REDS, ALL_REDS, ALL_NULL):
                text = headline(nova, lights).lower()
                assert not any(w in text for w in ("healthy", "unhealthy", "good", "bad"))


class TestTrafficBands:
    @pytest.mark.parametrize(
        "nutrient,value,expected",
        [
            # solids, inclusive edges exactly as the PRD table reads
            ("sugars", 5.0, "low"), ("sugars", 5.01, "medium"),
            ("sugars", 22.5, "medium"), ("sugars", 22.51, "high"),
            ("fat", 3.0, "low"), ("fat", 17.5, "medium"), ("fat", 17.51, "high"),
            ("saturated_fat", 1.5, "low"), ("saturated_fat", 5.0, "medium"),
            ("saturated_fat", 5.01, "high"),
            ("salt", 0.3, "low"), ("salt", 1.5, "medium"), ("salt", 1.51, "high"),
        ],
    )
    def test_solid_edges(self, nutrient, value, expected):
        assert traffic_band(nutrient, value, beverage=False) == expected

    @pytest.mark.parametrize(
        "nutrient,value,expected",
        [
            # drink table = exactly half the solid values
            ("sugars", 2.5, "low"), ("sugars", 2.51, "medium"),
            ("sugars", 11.25, "medium"), ("sugars", 11.26, "high"),
            ("fat", 1.5, "low"), ("fat", 8.75, "medium"), ("fat", 8.76, "high"),
            ("saturated_fat", 0.75, "low"), ("saturated_fat", 2.5, "medium"),
            ("salt", 0.15, "low"), ("salt", 0.75, "medium"), ("salt", 0.76, "high"),
        ],
    )
    def test_drink_edges(self, nutrient, value, expected):
        assert traffic_band(nutrient, value, beverage=True) == expected


class TestBeverageDetection:
    def test_exact_tag_and_chain(self):
        assert is_beverage("en:beverages")
        assert is_beverage("en:breakfasts,en:beverages,en:teas")

    def test_plant_food_grouping_is_not_a_drink(self):
        # 'en:plant-based-foods-and-beverages' contains '-beverages',
        # not 'en:beverages' — lentils must not get drink thresholds
        assert not is_beverage(
            "en:plant-based-foods-and-beverages,en:legumes,en:lentils"
        )

    def test_empty_defaults_to_solids(self):
        assert not is_beverage("")
        assert not is_beverage(None)


class TestRowHelpers:
    def _row(self, **overrides):
        row = {
            "categories_tags": "en:breakfasts,en:spreads",
            "sugars_100g": 56.3,
            "fat_100g": 30.9,
            "saturated_fat_100g": 10.6,
            "salt_100g": 0.107,
            "fiber_100g": 0.0,
            "proteins_100g": 6.3,
            "energy_100g": 539.0,
            "carbohydrates_100g": 57.5,
        }
        row.update(overrides)
        return row

    def test_missing_nutrient_is_none_not_a_band(self):
        lights = traffic_lights(self._row(salt_100g=None))
        assert lights["salt"] is None
        assert lights["sugars"] == "high"

    def test_traffic_lights_all_four_keys_always_present(self):
        assert set(traffic_lights({})) == {
            "sugars", "fat", "saturated_fat", "salt"
        }

    def test_positives_custom_bands(self):
        assert positives(self._row(fiber_100g=6.0))["fiber"] == "good"
        assert positives(self._row(fiber_100g=3.0))["fiber"] == "moderate"
        assert positives(self._row(fiber_100g=2.9))["fiber"] == "low"
        assert positives(self._row(proteins_100g=10.0))["proteins"] == "good"
        assert positives(self._row(proteins_100g=5.0))["proteins"] == "moderate"
        assert positives(self._row(proteins_100g=4.9))["proteins"] == "low"
        assert positives(self._row(proteins_100g=None))["proteins"] is None

    def test_nutrition_facts_nulls_and_key_set(self):
        facts = nutrition_100g(self._row(fiber_100g=float("nan")))
        assert set(facts) == set(NUTRITION_KEYS)  # the standard 8-row panel
        assert facts["fiber"] is None
        assert facts["energy"] == 539.0

    def test_build_scan_facts_returns_all_four_fields(self):
        facts = build_scan_facts(self._row(), 4)
        assert set(facts) == {
            "nutrition_100g", "traffic_lights", "positives", "headline"
        }
        assert facts["headline"].startswith("Ultra-processed")
