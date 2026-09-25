"""Stage 6 — FastAPI serving-layer tests (PRD 6.1–6.5).

Network policy: the default suite never calls the live Open Food Facts API
— a captured real product payload (tests/fixtures/, complete with OFF's
`nova_group` + Nutri-Score fields) is served through a monkeypatched
`fetch_product`, so CI (Stage 9) stays deterministic. The real end-to-end
path is available behind `CHEWSY_LIVE_OFF=1`.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import off_client
from app.main import STATE, app, build_feature_frame, score_row
from src.batch_predict import run as batch_run
from src.clean import FORBIDDEN_LEAKAGE_FIELDS, normalize_live_row

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "off_product_3017620422003.json"
KNOWN_BARCODE = "3017620422003"
FORBIDDEN = set(FORBIDDEN_LEAKAGE_FIELDS) | {"nova_group"}


@pytest.fixture(scope="module")
def client():
    # context manager runs the lifespan: loads model.joblib + TreeExplainer
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def raw_product():
    """Captured REAL OFF payload — deliberately keeps nova_group=4 and the
    Nutri-Score family so the stripping tests prove something."""
    return json.loads(FIXTURE_PATH.read_text())["product"]


@pytest.fixture()
def clean_product(raw_product):
    return off_client.strip_forbidden(raw_product)


@pytest.fixture()
def fake_fetch(clean_product, monkeypatch):
    """Serve the fixture instead of hitting the network."""

    def _fetch(barcode: str) -> dict:
        assert barcode == KNOWN_BARCODE
        return dict(clean_product)

    monkeypatch.setattr(off_client, "fetch_product", _fetch)
    return _fetch


def _response_for(client) -> dict:
    """POST a monkeypatched-fetch /predict and return the JSON body."""
    resp = client.post("/predict", json={"barcode": KNOWN_BARCODE})
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestLeakageGuards:
    """Hard rule 11 — OFF's stored label must never reach the model/response."""

    def test_strip_forbidden_removes_off_labels(self, raw_product):
        assert "nova_group" in raw_product
        assert FORBIDDEN & set(raw_product), "fixture must contain leak fields"
        stripped = off_client.strip_forbidden(raw_product)
        assert not FORBIDDEN & set(stripped)
        # the useful payload survives
        assert stripped["nutriments"]["energy-kcal_100g"] == 539

    def test_normalize_live_row_rejects_nova_group(self, clean_product):
        row = off_client.extract_feature_input(KNOWN_BARCODE, clean_product)
        row["nova_group"] = 4
        with pytest.raises(AssertionError, match="nova_group"):
            normalize_live_row(row)

    def test_normalize_live_row_rejects_nutriscore(self, clean_product):
        row = off_client.extract_feature_input(KNOWN_BARCODE, clean_product)
        row["nutrition_grades"] = "d"
        with pytest.raises(AssertionError, match="leakage"):
            normalize_live_row(row)

    def test_extract_never_contains_forbidden_fields(self, clean_product):
        row = off_client.extract_feature_input(KNOWN_BARCODE, clean_product)
        assert not FORBIDDEN & set(row)

    def test_response_never_echoes_off_labels(self, client, fake_fetch):
        body = _response_for(client)
        text = json.dumps(body)
        for field in FORBIDDEN:
            assert field not in text, f"{field} leaked into the response"

    def test_raw_unstripped_payload_is_still_clean(
        self, client, raw_product, monkeypatch
    ):
        """Belt-and-braces: even if fetch_product handed back the payload
        WITH nova_group (stripping bypassed), extract_feature_input's
        whitelist keeps it out of the frame and the response."""
        assert "nova_group" in raw_product
        monkeypatch.setattr(
            off_client, "fetch_product", lambda barcode: dict(raw_product)
        )
        body = _response_for(client)
        assert "nova_group" not in json.dumps(body)
        assert body["predicted_nova"] in (1, 2, 3, 4)


class TestNormalization:
    """PRD Stage 6 handoff — live rows get the Stage 2 normalizations."""

    @pytest.fixture()
    def live_row(self, clean_product):
        row = off_client.extract_feature_input(KNOWN_BARCODE, clean_product)
        return normalize_live_row(row)

    def test_energy_mapped_as_kcal_not_kj(self, clean_product, live_row):
        # flatten_hit uses energy-kcal_100g (539); the kJ value (2252) would
        # blow past clean.py's 900 kcal cap and be NaN'd out
        assert live_row["energy_100g"] == 539

    def test_stage2_normalizations_applied(self, live_row):
        # multi-brand "Nutella, Ferrero" -> first segment, matching how
        # clean() parses the training data's list-repr ['Nutella', ...]
        # ('nutella' is in the FrequencyEncoder vocab; the full comma
        # string is not — train/serve parity, see extract_feature_input)
        assert live_row["brands"] == "nutella"
        assert live_row["categories_tags"].startswith("en:breakfasts")
        assert live_row["additives_n"] == 2
        assert live_row["unknown_ingredients_n"] == 0
        assert "sugar palm oil" in live_row["ingredients_pseudo_text"]

    def test_indicators_built_like_stage2(self, live_row):
        # clean() records missingness from the ORIGINAL nulls, same as the
        # training CSV; FeatureCreator preserves them at transform time
        assert live_row["fiber_100g_was_missing"] == 0
        assert live_row["sodium_100g_was_missing"] == 0
        assert live_row["text_was_missing"] == 0
        assert live_row["nutrients_all_missing"] == 0


class TestFetchProductClassification:
    """Hard rule 5: the client's 404/503 mapping itself — tested WITHOUT
    the live gate by faking ``requests.get`` (the /predict 404/503 tests
    monkeypatch fetch_product and would never catch a regression here)."""

    class _FakeResp:
        def __init__(self, status_code=200, payload=None, json_raises=False):
            self.status_code = status_code
            self._payload = payload
            self._json_raises = json_raises

        def json(self):
            if self._json_raises:
                raise ValueError("not json")
            return self._payload

    @staticmethod
    def _patch(monkeypatch, response):
        if isinstance(response, Exception):
            def _raise(*args, **kwargs):
                raise response
            monkeypatch.setattr(off_client.requests, "get", _raise)
        else:
            monkeypatch.setattr(off_client.requests, "get",
                                lambda *a, **k: response)

    def test_found_200_returns_stripped_product(self, monkeypatch, raw_product):
        self._patch(monkeypatch, self._FakeResp(200, {"status": 1, "product": raw_product}))
        product = off_client.fetch_product("3017620422003")
        assert "nova_group" not in product
        assert product["product_name"] == "Nutella"

    def test_status_zero_200_is_not_found(self, monkeypatch):
        self._patch(monkeypatch, self._FakeResp(200, {"status": 0}))
        with pytest.raises(off_client.ProductNotFound):
            off_client.fetch_product("0000000000000")

    def test_http_404_is_not_found(self, monkeypatch):
        self._patch(monkeypatch, self._FakeResp(404, {}))
        with pytest.raises(off_client.ProductNotFound):
            off_client.fetch_product("0000000000000")

    def test_http_429_with_status_zero_body_is_unavailable_not_404(
        self, monkeypatch
    ):
        # rate-limit bodies often carry {"status": 0} — must NOT read as
        # "barcode not found" during a rate-limited demo
        self._patch(monkeypatch, self._FakeResp(429, {"status": 0}))
        with pytest.raises(off_client.OffAPIUnavailable):
            off_client.fetch_product("3017620422003")

    def test_http_500_is_unavailable(self, monkeypatch):
        self._patch(monkeypatch, self._FakeResp(500, {}))
        with pytest.raises(off_client.OffAPIUnavailable):
            off_client.fetch_product("3017620422003")

    def test_timeout_is_unavailable(self, monkeypatch):
        import requests as requests_lib

        self._patch(monkeypatch, requests_lib.Timeout("timed out"))
        with pytest.raises(off_client.OffAPIUnavailable):
            off_client.fetch_product("3017620422003")

    def test_non_json_body_is_unavailable(self, monkeypatch):
        self._patch(monkeypatch, self._FakeResp(200, json_raises=True))
        with pytest.raises(off_client.OffAPIUnavailable):
            off_client.fetch_product("3017620422003")

    def test_product_wrong_type_is_unavailable(self, monkeypatch):
        # product as a list must not escape as an unhandled 500
        self._patch(monkeypatch, self._FakeResp(200, {"status": 1, "product": []}))
        with pytest.raises(off_client.OffAPIUnavailable):
            off_client.fetch_product("3017620422003")


class TestHealthAndMetrics:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True

    def test_metrics_counts_requests(self, client, fake_fetch):
        before = client.get("/metrics").json()
        _response_for(client)
        after = client.get("/metrics").json()
        assert after["requests"] == before["requests"] + 1
        assert after["avg_latency_ms"] >= 0


class TestPredict:
    def test_well_formed_response(self, client, fake_fetch):
        body = _response_for(client)
        assert body["barcode"] == KNOWN_BARCODE
        assert body["product_name"] == "Nutella"
        assert body["image_url"].startswith("http")
        assert body["predicted_nova"] in (1, 2, 3, 4)
        assert body["nova_label"]
        assert 0 < body["confidence"] <= 1
        assert set(body["class_probabilities"]) == {"1", "2", "3", "4"}
        assert abs(sum(body["class_probabilities"].values()) - 1.0) < 1e-3
        assert 1 <= len(body["shap_top_features"]) <= 5
        for f in body["shap_top_features"]:
            assert f["feature"]
            assert isinstance(f["shap_value"], float)

    def test_prediction_equals_direct_model_output(self, client, fake_fetch, clean_product):
        """The route must be a thin wrapper over the model — no passthrough,
        no OFF conditioning (the fixture's nova_group was stripped)."""
        body = _response_for(client)
        df = build_feature_frame(KNOWN_BARCODE, clean_product)
        pred, proba = score_row(STATE["pipeline"], df)
        assert body["predicted_nova"] == pred
        assert body["confidence"] == pytest.approx(float(proba[pred - 1]), abs=1e-4)

    def test_shap_values_match_explainer_for_predicted_class(
        self, client, fake_fetch, clean_product
    ):
        """Pin the class-index math (NOVA p -> adapter column p-1): a
        regression using `pred` instead of `pred-1` must fail here."""
        import numpy as np

        body = _response_for(client)
        pred = body["predicted_nova"]
        df = build_feature_frame(KNOWN_BARCODE, clean_product)
        transformed = STATE["pipeline"].named_steps["feature_pipeline"].transform(df)
        values = STATE["explainer"](transformed).values
        if isinstance(values, list):
            values = np.stack(values, axis=-1)
        contrib = np.asarray(values)[0, :, pred - 1]
        names = STATE["feature_names"]
        for f in body["shap_top_features"]:
            expected = round(float(contrib[names.index(f["feature"])]), 4)
            assert f["shap_value"] == pytest.approx(expected, abs=1e-4)

    def test_unknown_barcode_404(self, client, monkeypatch):
        def _raise(barcode):
            raise off_client.ProductNotFound(f"barcode {barcode} not found")

        monkeypatch.setattr(off_client, "fetch_product", _raise)
        resp = client.post("/predict", json={"barcode": "9999999999999"})
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_api_unreachable_503(self, client, monkeypatch):
        def _raise(barcode):
            raise off_client.OffAPIUnavailable("timeout")

        monkeypatch.setattr(off_client, "fetch_product", _raise)
        resp = client.post("/predict", json={"barcode": KNOWN_BARCODE})
        assert resp.status_code == 503

    def test_invalid_barcode_422(self, client):
        resp = client.post("/predict", json={"barcode": "abc"})
        assert resp.status_code == 422


class TestStage67TwoAxisResponse:
    """PRD 6.7 — nutrition facts, NHS traffic lights, combined headline.

    All four fields are computed locally from the normalized scoring row
    (Hard rule 12: threshold lookups on nutrition_100g, never fetched
    OFF grades) — the fixture's nova_group/Nutri-Score stripping tests
    above still cover the whole body.
    """

    def test_nutella_bands_facts_and_headline(self, client, fake_fetch):
        body = _response_for(client)

        # traffic lights from the SOLID table (fixture tags have no
        # en:beverages) — sugars 56.3 > 22.5 high, salt 0.107 <= 0.3 low
        assert body["traffic_lights"] == {
            "sugars": "high",
            "fat": "high",          # 30.9 > 17.5
            "saturated_fat": "high",  # 10.6 > 5
            "salt": "low",
        }
        # the 8 per-100g facts, Stage 2-normalized (PRD examples: energy
        # 539, salt 0.107) — null for anything unpublished, never invented
        assert body["nutrition_100g"] == {
            "energy": 539,
            "fat": 30.9,
            "saturated_fat": 10.6,
            "carbohydrates": 57.5,
            "sugars": 56.3,
            "fiber": 0.0,
            "proteins": 6.3,
            "salt": 0.107,
        }
        # custom positives (fibre good >=6 / moderate >=3; protein >=10/5)
        assert body["positives"] == {"fiber": "low", "proteins": "moderate"}
        assert body["additives_n"] == 2
        assert body["ingredients_n"] == 9
        assert body["ingredients_text"].startswith("Sucre, huile de palme")
        # two-axis headline: model computes NOVA 4, nutrients have reds
        assert body["predicted_nova"] == 4
        assert body["headline"] == (
            "Ultra-processed and high in sugar, fat and saturated fat"
        )

    def test_missing_nutrient_is_null_never_a_guessed_band(
        self, client, clean_product, monkeypatch
    ):
        product = dict(clean_product)
        product["nutriments"] = {
            k: v for k, v in clean_product["nutriments"].items() if k != "salt_100g"
        }
        monkeypatch.setattr(off_client, "fetch_product", lambda barcode: dict(product))
        body = _response_for(client)
        assert body["traffic_lights"]["salt"] is None
        assert body["nutrition_100g"]["salt"] is None
        # published nutrients still get their bands
        assert body["traffic_lights"]["sugars"] == "high"
        assert body["nutrition_100g"]["sugars"] == 56.3

    @pytest.mark.parametrize(
        "sugars,expected",
        [(5.0, "low"), (22.5, "medium")],  # inclusive edges of the PRD table
    )
    def test_band_boundaries_lock_inclusive_edges(
        self, client, clean_product, monkeypatch, sugars, expected
    ):
        product = dict(clean_product)
        nutriments = dict(clean_product["nutriments"])
        nutriments["sugars_100g"] = sugars
        product["nutriments"] = nutriments
        monkeypatch.setattr(off_client, "fetch_product", lambda barcode: dict(product))
        body = _response_for(client)
        assert body["traffic_lights"]["sugars"] == expected

    def test_beverage_flips_to_drink_thresholds(
        self, client, clean_product, monkeypatch
    ):
        def _product(categories):
            p = dict(clean_product)
            nutriments = dict(clean_product["nutriments"])
            nutriments["sugars_100g"] = 4.8  # low for solids, medium for drinks
            p["nutriments"] = nutriments
            p["categories_tags"] = categories
            return p

        # drink table: 4.8 > 2.5 -> medium
        monkeypatch.setattr(
            off_client, "fetch_product",
            lambda barcode: _product(["en:breakfasts", "en:beverages", "en:teas"]),
        )
        assert _response_for(client)["traffic_lights"]["sugars"] == "medium"

        # same product, empty categories -> solids table: 4.8 <= 5 -> low
        monkeypatch.setattr(
            off_client, "fetch_product", lambda barcode: _product([])
        )
        assert _response_for(client)["traffic_lights"]["sugars"] == "low"


class TestBatchPredict:
    def test_scores_csv_and_isolates_failures(
        self, client, fake_fetch, clean_product, tmp_path, monkeypatch
    ):
        calls = []

        def _fetch(barcode: str) -> dict:
            calls.append(barcode)
            if barcode == "9999999999999":
                raise off_client.ProductNotFound("not found")
            return dict(clean_product)

        monkeypatch.setattr(off_client, "fetch_product", _fetch)

        inp = tmp_path / "barcodes.csv"
        out = tmp_path / "preds.csv"
        pd.DataFrame(
            {"barcode": [KNOWN_BARCODE, "9999999999999", "not-a-barcode"]}
        ).to_csv(inp, index=False)
        result = batch_run(inp, out)

        assert out.exists()
        assert len(result) == 3
        good = result[result["error"] == ""].iloc[0]
        bad = result[result["error"].str.contains("not found")].iloc[0]
        junk = result[result["barcode"] == "not-a-barcode"].iloc[0]
        assert int(good["predicted_nova"]) in (1, 2, 3, 4)
        assert 0 < float(good["confidence"]) <= 1
        assert "not found" in bad["error"]
        assert pd.isna(bad["predicted_nova"])
        assert "invalid barcode" in junk["error"]  # validated, not sent to OFF


@pytest.mark.skipif(
    os.getenv("CHEWSY_LIVE_OFF") != "1",
    reason="live Open Food Facts call — opt in with CHEWSY_LIVE_OFF=1",
)
class TestLiveEndToEnd:
    def test_real_barcode_through_real_api(self, client):
        resp = client.post("/predict", json={"barcode": KNOWN_BARCODE})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["predicted_nova"] in (1, 2, 3, 4)
        assert body["product_name"]
        assert "nova_group" not in resp.text
