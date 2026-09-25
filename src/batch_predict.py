"""Stage 6.5 -- offline batch scoring of a CSV of barcodes (PRD 6.5).

Batch vs. real-time inference: ``app/main.py`` answers one scan at a time
for the live demo; this script scores a whole BACKLOG of unlabeled Open Food
Facts products offline -- literally how you'd backfill NOVA labels for the
products the community hasn't gotten to yet. Same model, same scoring path
(``app.main.score_row``), no SHAP (explanations are a per-scan UI concern).

Usage:
    python src/batch_predict.py --input barcodes.csv --output predictions.csv

Input CSV needs a barcode column (default name: ``barcode``). Rows whose
barcode cannot be fetched/scored are kept in the output with an ``error``
message instead of aborting the batch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from app import off_client  # noqa: E402
from app.main import build_feature_frame, load_pipeline, score_row  # noqa: E402

BARCODE_COL = "barcode"
OUTPUT_COLS = [
    "barcode", "product_name", "predicted_nova", "nova_label",
    "confidence", "error",
]


def score_one(pipeline, barcode: str) -> dict:
    """Score a single barcode; failures become an ``error`` string."""
    from app.schemas import NOVA_LABELS

    base = {
        "barcode": barcode,
        "product_name": "",
        "predicted_nova": pd.NA,
        "nova_label": "",
        "confidence": pd.NA,
        "error": "",
    }
    try:
        product = off_client.fetch_product(barcode)
        df = build_feature_frame(barcode, product)
        pred, proba = score_row(pipeline, df)
        base.update(
            product_name=(product.get("product_name") or "").strip(),
            predicted_nova=pred,
            nova_label=NOVA_LABELS[pred],
            confidence=round(float(proba[pred - 1]), 4),
        )
        return base
    except (
        off_client.ProductNotFound,
        off_client.OffAPIUnavailable,
        AssertionError,
    ) as exc:
        base["error"] = str(exc)
        return base


def run(input_csv: Path, output_csv: Path, barcode_col: str = BARCODE_COL) -> pd.DataFrame:
    df = pd.read_csv(input_csv, dtype=str)
    assert barcode_col in df.columns, (
        f"input CSV has no '{barcode_col}' column; columns: {list(df.columns)}"
    )
    barcodes = [b for b in df[barcode_col].dropna().tolist() if str(b).strip()]
    print(f"{len(barcodes)} barcodes to score -> {output_csv}")

    pipeline = load_pipeline()  # once, not per row
    results = []
    t0 = time.perf_counter()
    for i, barcode in enumerate(barcodes, start=1):
        results.append(score_one(pipeline, str(barcode).strip()))
        if i % 25 == 0 or i == len(barcodes):
            print(f"  {i}/{len(barcodes)} ({time.perf_counter() - t0:.1f}s)")

    out = pd.DataFrame(results, columns=OUTPUT_COLS)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    n_err = int((out["error"] != "").sum())
    print(
        f"done: {len(out) - n_err} scored, {n_err} failed -> {output_csv}"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with a barcode column")
    parser.add_argument("--output", type=Path, required=True,
                        help="CSV to write predictions to")
    parser.add_argument("--barcode-col", default=BARCODE_COL)
    args = parser.parse_args(argv)
    run(args.input, args.output, args.barcode_col)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
