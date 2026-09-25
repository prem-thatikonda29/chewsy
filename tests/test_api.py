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
        assert live_row["brands"] == "nutella, ferrero"  # lowercased
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
        pd.DataFrame({"barcode": [KNOWN_BARCODE, "9999999999999"]}).to_csv(
            inp, index=False
        )
        result = batch_run(inp, out)

        assert out.exists()
        assert len(result) == 2
        good = result[result["error"] == ""].iloc[0]
        bad = result[result["error"] != ""].iloc[0]
        assert int(good["predicted_nova"]) in (1, 2, 3, 4)
        assert 0 < float(good["confidence"]) <= 1
        assert "not found" in bad["error"]
        assert pd.isna(bad["predicted_nova"])


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
