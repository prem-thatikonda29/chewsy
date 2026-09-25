"""Stage 12.2 — drift check: live-fetched rows vs models/training_reference.csv.

PRD 12.2: a small script comparing live nutrient distributions against the
packaged reference with ``scipy.stats.ks_2samp`` — one manual run + plot is
enough to demonstrate the concept, no always-on drift job.

Two signals per shared column (the Stage 12 handoff in the PRD):
  * **KS statistic** on non-null values (value-distribution drift), for
    numeric nutrient/count columns;
  * **null-fraction delta** for *every* shared column (documentation drift
    — OFF pages that stop recording fiber show up as a rising ``fiber_100g``
    NaN share before any value distribution moves).

Indicator columns (``*_was_missing``, ``nutrients_all_missing``) are the
missingness itself, so they get null deltas but no KS test.

Live rows travel the exact serving path (``off_client.fetch_product`` →
``build_feature_frame``), so the comparison happens in the reference's own
input space — train/serve parity for free.

Usage:
    python src/drift_check.py --n 100                  # random live sample
    python src/drift_check.py --barcodes barcodes.csv  # deterministic re-run
    python src/drift_check.py --ref models/training_reference.csv \\
                               --out docs/drift_report.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless: save PNG, never open a window

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from scipy.stats import ks_2samp  # noqa: E402

from app import off_client  # noqa: E402
from app.main import build_feature_frame  # noqa: E402
from src.fetch_training_set import USER_AGENT  # noqa: E402

# KS stat above this on a ~100-row sample is worth talking about in slides
# (0.15 ≈ clearly separated ECDFs at this sample size, not noise)
KS_THRESHOLD = 0.15

SEARCH_URL = "https://search.openfoodfacts.org/search"
PAGE_SIZE = 200  # confirmed working ceiling for this API (Stage 1)
DEFAULT_REF = REPO_ROOT / "models" / "training_reference.csv"
DEFAULT_OUT = REPO_ROOT / "docs" / "drift_report.png"
BARCODE_COL = "barcode"


def load_reference(path: Path | str = DEFAULT_REF) -> pd.DataFrame:
    """NaN-preserving reference frame (the packaged Stage 5 input space)."""
    return pd.read_csv(path)


def _is_indicator(col: str) -> bool:
    return col.endswith("_was_missing") or col == "nutrients_all_missing"


def _null_frac(series: pd.Series) -> float:
    """Missingness share of a column.

    Text columns count empty/whitespace strings as missing too: the
    serving path builds "" where the training CSV has NaN (both mean
    "no text"), and counting only NaN produced a spurious -0.202
    ingredients_pseudo_text shift in the first live run.
    """
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return float(series.fillna("").astype(str).str.strip().eq("").mean())
    return float(series.isna().mean())


def ks_table(ref: pd.DataFrame, live: pd.DataFrame) -> pd.DataFrame:
    """Per-shared-column drift table, worst first.

    Columns: ``column, ks_stat, p_value, ref_null_frac, live_null_frac,
    null_delta``. ``ks_stat``/``p_value`` are NaN for text and indicator
    columns (null-rate story only). Sort: numeric rows by KS desc, then
    non-numeric rows by |null_delta| desc.
    """
    rows = []
    for col in ref.columns:
        if col not in live.columns:
            continue
        r, lv = ref[col], live[col]
        ref_null = _null_frac(r)
        live_null = _null_frac(lv)
        ks_stat = p_value = float("nan")
        numeric = pd.api.types.is_numeric_dtype(r) and pd.api.types.is_numeric_dtype(lv)
        if numeric and not _is_indicator(col):
            rv, lvv = r.dropna(), lv.dropna()
            if len(rv) and len(lvv):
                ks_stat, p_value = map(float, ks_2samp(rv, lvv))
        rows.append(
            {
                "column": col,
                "ks_stat": ks_stat,
                "p_value": p_value,
                "ref_null_frac": ref_null,
                "live_null_frac": live_null,
                "null_delta": live_null - ref_null,
            }
        )
    table = pd.DataFrame(rows, columns=[
        "column", "ks_stat", "p_value",
        "ref_null_frac", "live_null_frac", "null_delta",
    ])
    table["abs_delta"] = table["null_delta"].abs()
    table = pd.concat(
        [
            table[table["ks_stat"].notna()].sort_values("ks_stat", ascending=False),
            table[table["ks_stat"].isna()].sort_values("abs_delta", ascending=False),
        ],
        ignore_index=True,
    ).drop(columns="abs_delta")
    return table


def discover_barcodes(n: int) -> list[str]:
    """n barcodes from a broad search-API sweep (code field only)."""
    codes: list[str] = []
    page = 1
    while len(codes) < n:
        resp = requests.get(
            SEARCH_URL,
            params={
                "q": "",  # required — the endpoint 400s without a q param;
                # empty string = broad sweep across all products
                "page": page,
                "page_size": min(n - len(codes), PAGE_SIZE),
                "fields": "code",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=5,  # Hard rule 5: never hang the drift run
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        if not hits:
            break
        codes.extend(str(h["code"]) for h in hits if h.get("code"))
        page += 1
    return codes[:n]


def collect_live(barcodes: list[str]) -> pd.DataFrame:
    """Fetch each barcode through the serving path; skip failures.

    Paced (0.3s between products) and 429-aware: the live run on 25 Sep
    dropped 80/100 rows to rate-limits before this — one backoff retry
    rescues the row instead. A dead row still must not kill the sweep.
    """
    frames = []
    for barcode in barcodes:
        for attempt in range(3):
            try:
                product = off_client.fetch_product(barcode)
                frame, _ = build_feature_frame(barcode, product)
                frames.append(frame)
                break
            except off_client.OffAPIUnavailable as exc:
                if "429" in str(exc) and attempt < 2:
                    time.sleep(2.0 * (attempt + 1))  # back off, then retry
                    continue
                print(f"  skip {barcode}: {exc}", file=sys.stderr)
                break
            except Exception as exc:  # noqa: BLE001 — dead row ≠ dead sweep
                print(f"  skip {barcode}: {exc}", file=sys.stderr)
                break
        time.sleep(0.3)  # polite pacing (same spirit as Stage 1 fetch)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_ks(table: pd.DataFrame, out_path: Path | str = DEFAULT_OUT) -> None:
    """Horizontal KS bars with the reference threshold line (slides asset)."""
    numeric = table[table["ks_stat"].notna()]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.4 * len(numeric) + 1.5)))
    if len(numeric):
        ax.barh(
            numeric["column"][::-1],
            numeric["ks_stat"][::-1],
            color="#6b7280",
        )
        ax.axvline(KS_THRESHOLD, color="#dc2626", linestyle="--", linewidth=1)
        ax.set_xlabel("KS statistic (live vs training reference)")
        ax.set_xlim(0, 1)
    else:
        ax.text(0.5, 0.5, "no shared numeric columns", ha="center", va="center")
        ax.set_xticks([])
    ax.set_title("Drift check — live rows vs training reference (per-column KS)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KS drift check vs training reference")
    parser.add_argument("--n", type=int, default=100, help="live barcodes to sample")
    parser.add_argument(
        "--barcodes", type=Path, default=None,
        help="CSV of barcodes (barcode column) — deterministic re-run, skips discovery",
    )
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    ref = load_reference(args.ref)
    if args.barcodes is not None:
        raw = pd.read_csv(args.barcodes)
        col = BARCODE_COL if BARCODE_COL in raw.columns else raw.columns[0]
        codes = (
            raw[col].astype(str).str.replace(r"\.0$", "", regex=True).tolist()
        )
    else:
        print(f"discovering {args.n} barcodes…")
        codes = discover_barcodes(args.n)
    if not codes:
        print("no barcodes to sample", file=sys.stderr)
        return 1

    print(f"fetching {len(codes)} live products…")
    live = collect_live(codes)
    if live.empty:
        print("no live rows fetched", file=sys.stderr)
        return 1

    table = ks_table(ref, live)
    print(f"\nlive rows: {len(live)} / requested {len(codes)}\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    numeric = table[table["ks_stat"].notna()]
    flagged = numeric[numeric["ks_stat"] >= KS_THRESHOLD]
    worst = numeric.iloc[0] if len(numeric) else None
    all_deltas = table.reindex(table["null_delta"].abs().sort_values(ascending=False).index)
    biggest_null = all_deltas.iloc[0] if len(table) else None

    print()
    if worst is not None:
        verdict = "DRIFT" if worst["ks_stat"] >= KS_THRESHOLD else "ok"
        print(
            f"worst KS: {worst['column']} = {worst['ks_stat']:.3f} "
            f"(p={worst['p_value']:.2g}) [{verdict}, threshold {KS_THRESHOLD}]"
        )
        print(f"columns over threshold: {len(flagged)} / {len(numeric)}")
    if biggest_null is not None:
        print(
            f"biggest null shift: {biggest_null['column']} = "
            f"{biggest_null['null_delta']:+.3f} "
            f"({biggest_null['ref_null_frac']:.0%} → {biggest_null['live_null_frac']:.0%})"
        )
    plot_ks(table, args.out)
    print(f"plot → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
