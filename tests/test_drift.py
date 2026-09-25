"""Stage 12.2 — KS drift-check tests (PRD 12.2). No network: the fetch
paths take monkeypatched products/responses, the math is pure."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from app import off_client
from src.drift_check import (
    KS_THRESHOLD,
    collect_live,
    discover_barcodes,
    ks_table,
    load_reference,
    plot_ks,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "off_product_3017620422003.json"
KNOWN_BARCODE = "3017620422003"
REF_PATH = REPO_ROOT / "models" / "training_reference.csv"


def make_frames(shift: float = 0.0, ref_null: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two small frames sharing the reference's column shapes."""
    ref = pd.DataFrame(
        {
            "energy_100g": [100.0, 200.0, 300.0, 400.0, 500.0],
            "sugars_100g": [1.0, 2.0, 3.0, 4.0, 5.0],
            "product_name": ["a", "b", None, "d", "e"],
            "text_was_missing": [0, 0, 1, 0, 0],
        }
    )
    live = ref.copy()
    live["energy_100g"] = live["energy_100g"] + shift
    if ref_null:
        live.loc[: ref_null - 1, "sugars_100g"] = None
    return ref, live


class TestKsTable:
    def test_identical_frames_have_zero_ks(self):
        ref, live = make_frames()
        table = ks_table(ref, live).set_index("column")
        assert table.loc["energy_100g", "ks_stat"] == 0.0
        assert table.loc["energy_100g", "null_delta"] == 0.0

    def test_disjoint_distributions_have_ks_one(self):
        ref, live = make_frames()
        live["energy_100g"] = [9000.0, 9100.0, 9200.0, 9300.0, 9400.0]
        table = ks_table(ref, live).set_index("column")
        assert table.loc["energy_100g", "ks_stat"] == pytest.approx(1.0)

    def test_nulls_excluded_from_ks_but_counted_in_null_delta(self):
        ref, live = make_frames(ref_null=2)  # live sugars 2/5 = 40% null
        table = ks_table(ref, live).set_index("column")
        sugars = table.loc["sugars_100g"]
        # KS compared only the 3 non-null live values vs 5 ref values
        assert sugars["ks_stat"] < 1.0
        assert sugars["ref_null_frac"] == 0.0
        assert sugars["live_null_frac"] == pytest.approx(0.4)
        assert sugars["null_delta"] == pytest.approx(0.4)

    def test_text_and_indicator_columns_get_null_delta_not_ks(self):
        ref, live = make_frames()
        live.loc[0, "product_name"] = None
        table = ks_table(ref, live).set_index("column")
        # text col: no KS, but the null-rate shift is reported
        assert pd.isna(table.loc["product_name", "ks_stat"])
        assert table.loc["product_name", "null_delta"] == pytest.approx(0.2)
        # indicator col: documentation signal, not a value distribution
        assert pd.isna(table.loc["text_was_missing", "ks_stat"])

    def test_empty_strings_count_as_missing_for_text(self):
        """Live-run finding (25 Sep): serving rows carry "" where the
        reference has NaN — without this the text null-delta was a
        spurious -0.202 and stole the 'biggest shift' headline."""
        ref, live = make_frames()
        ref["product_name"] = ["a", "", None, "d", "e"]  # 2/5 missing
        table = ks_table(ref, live).set_index("column")
        assert table.loc["product_name", "ref_null_frac"] == pytest.approx(0.4)
        # live (same frame) also has 1 None → 0.2; delta = 0.2 - 0.4
        assert table.loc["product_name", "null_delta"] == pytest.approx(-0.2)

    def test_worst_first_ordering(self):
        ref = pd.DataFrame(
            {"clean_100g": [1.0, 2.0, 3.0], "drifted_100g": [1.0, 2.0, 3.0]}
        )
        live = pd.DataFrame(
            {"clean_100g": [1.0, 2.0, 3.0], "drifted_100g": [80.0, 90.0, 100.0]}
        )
        table = ks_table(ref, live)
        assert table.iloc[0]["column"] == "drifted_100g"
        assert table.iloc[0]["ks_stat"] == pytest.approx(1.0)
        assert table.iloc[1]["ks_stat"] == 0.0

    def test_threshold_constant_exported(self):
        assert KS_THRESHOLD == pytest.approx(0.15)


class TestLoadReference:
    def test_shipped_reference_covers_its_numeric_columns(self):
        ref = load_reference(REF_PATH)
        assert {"energy_100g", "sugars_100g"} <= set(ref.columns)
        assert len(ref) > 1000
        # Hard rule: never a leakage/target column in the drift baseline
        for col in ("nova_group", "nutrition_grades", "nutriscore_grade"):
            assert col not in ref.columns


class TestCollectLive:
    def test_collect_live_uses_the_serving_path(self, monkeypatch):
        product = json.loads(FIXTURE_PATH.read_text())["product"]
        monkeypatch.setattr(
            off_client, "fetch_product", lambda barcode: dict(product)
        )
        live = collect_live([KNOWN_BARCODE])
        assert len(live) == 1
        ref = load_reference(REF_PATH)
        # the live frame lands in the reference's input space
        assert set(ref.columns) <= set(live.columns)
        assert not {"nova_group", "nutrition_grades"} & set(live.columns)

    def test_collect_live_skips_failed_barcodes(self, monkeypatch):
        def _fetch(barcode):
            if barcode == "9999999999999":
                raise off_client.ProductNotFound("nope")
            raise off_client.OffAPIUnavailable("down")

        monkeypatch.setattr(off_client, "fetch_product", _fetch)
        live = collect_live(["9999999999999", "1111111111111"])
        assert len(live) == 0

    def test_collect_live_retries_after_429_then_succeeds(self, monkeypatch):
        """Live-run finding (25 Sep): the product API 429s when swept —
        a single paced retry must rescue the row, not drop it."""
        monkeypatch.setattr("src.drift_check.time.sleep", lambda _s: None)
        product = json.loads(FIXTURE_PATH.read_text())["product"]
        calls = []

        def _fetch(barcode):
            calls.append(barcode)
            if len(calls) == 1:
                raise off_client.OffAPIUnavailable("Open Food Facts returned HTTP 429")
            return dict(product)

        monkeypatch.setattr(off_client, "fetch_product", _fetch)
        live = collect_live([KNOWN_BARCODE])
        assert len(live) == 1
        assert len(calls) == 2

    def test_collect_live_paces_requests(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("src.drift_check.time.sleep", sleeps.append)
        product = json.loads(FIXTURE_PATH.read_text())["product"]
        monkeypatch.setattr(off_client, "fetch_product", lambda b: dict(product))
        collect_live(["1111111111111", "2222222222222"])
        assert sleeps, "no pacing sleep between live fetches"
        assert all(s > 0 for s in sleeps)


class TestDiscoverBarcodes:
    def test_parses_codes_and_requests_only_code_field(self, monkeypatch):
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"hits": [{"code": "3017620422003"}, {"code": "5449000000996"}]}

        def _get(url, params=None, timeout=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["timeout"] = timeout
            captured["headers"] = headers
            return _Resp()

        monkeypatch.setattr("src.drift_check.requests.get", _get)
        codes = discover_barcodes(2)
        assert codes == ["3017620422003", "5449000000996"]
        assert captured["timeout"] <= 5  # Hard rule 5: timeouts everywhere
        assert captured["params"]["fields"] == "code"
        # the endpoint 400s without a q param (live-run finding, 25 Sep)
        assert captured["params"]["q"] == ""
        assert captured["headers"]["User-Agent"].startswith("Chewsy")
        assert "nutrition_grades" not in str(captured["params"])


class TestPlot:
    def test_plot_writes_png(self, tmp_path):
        ref, live = make_frames(shift=250.0)
        table = ks_table(ref, live)
        out = tmp_path / "drift.png"
        plot_ks(table, out)
        assert out.exists()
        assert out.stat().st_size > 0
