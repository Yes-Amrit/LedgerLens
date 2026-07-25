"""
Task 0 — Data-driven threshold derivation for the rule-based structuring detector.

Run BEFORE setting the default threshold in tools/rule_based_detector.py.

Usage:
    python data/threshold_analysis.py

Prints:
    - Amount distribution for ALL structuring typology rows in SAML-D
    - Chosen threshold and justification (fed into rule_based_detector.py default)
"""

import json
import sys
from pathlib import Path

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / "SAML-D.csv"
MAPPING_PATH = Path(__file__).parent / "typology_mapping.json"

# Read in chunks to avoid loading 1 GB into RAM all at once
CHUNK_SIZE = 200_000
READ_ROWS = 1_500_000  # cap at 1.5 M — enough to see the full typology distribution


def main() -> None:
    # ── Load typology → bucket mapping ────────────────────────────────────────
    with open(MAPPING_PATH) as f:
        mapping: dict = json.load(f)

    structuring_typologies = {k for k, v in mapping.items() if v == "structuring"}
    print(f"Structuring typologies in mapping: {structuring_typologies}\n")

    # ── Stream CSV, collect structuring rows ──────────────────────────────────
    chunks = []
    rows_read = 0

    # Peek at headers first
    header_df = pd.read_csv(DATA_PATH, nrows=1)
    print(f"CSV columns detected: {header_df.columns.tolist()}\n")

    # Detect column names (case-sensitive — SAML-D uses title case)
    cols = header_df.columns.tolist()
    amount_col = "Amount" if "Amount" in cols else "amount"
    typology_col = "Laundering_type" if "Laundering_type" in cols else "laundering_type"
    label_col = "Is_laundering" if "Is_laundering" in cols else "is_laundering"

    for chunk in pd.read_csv(
        DATA_PATH,
        usecols=[amount_col, typology_col, label_col],
        chunksize=CHUNK_SIZE,
    ):
        struct_mask = chunk[typology_col].isin(structuring_typologies)
        chunks.append(chunk[struct_mask])
        rows_read += len(chunk)
        if rows_read >= READ_ROWS:
            break

    if not chunks:
        print("[ERROR] No structuring rows found — check typology_mapping.json column names.")
        sys.exit(1)

    df_struct = pd.concat(chunks, ignore_index=True)
    amounts = pd.to_numeric(df_struct[amount_col], errors="coerce").dropna()

    # ── Distribution analysis ─────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"Structuring rows found : {len(df_struct):,}")
    print(f"Rows read from CSV     : {rows_read:,}")
    print(f"{'='*60}\n")

    print("Amount distribution for STRUCTURING typologies:")
    print(amounts.describe().to_string())
    print()

    quantiles = amounts.quantile([0.50, 0.75, 0.90, 0.95])
    print("Quantiles:")
    for q, v in quantiles.items():
        print(f"  {int(q*100):>3}th percentile : ${v:>10,.2f}")

    # ── Threshold decision ────────────────────────────────────────────────────
    p90 = quantiles[0.90]
    p95 = quantiles[0.95]

    # We want a threshold T such that structuring transactions are BELOW T.
    # Using 90th percentile captures 90% of real structuring cases as the
    # "zone ceiling" while keeping the window tight enough for precision.
    # The rule then flags transactions between (T * 0.9) and T — the last
    # 10% of the zone — where structuring pressure to stay just under is highest.
    chosen_threshold = p90

    print(f"\n{'='*60}")
    print(f"CHOSEN THRESHOLD : ${chosen_threshold:,.2f}")
    print(f"{'='*60}")
    print(
        f"Justification: 90th percentile of actual structuring transaction amounts "
        f"in SAML-D ({len(df_struct):,} rows). This means 90% of real structuring "
        f"cases fall below this value, and the rule-based detector's zone "
        f"[threshold * 0.9, threshold) captures the dense clustering region. "
        f"Using the 95th percentile (${p95:,.2f}) would raise false-positive risk "
        f"by pulling in more large legitimate transactions."
    )
    print(
        f"\nUpdate tools/rule_based_detector.py: "
        f"detect_structuring(df, threshold={chosen_threshold:.2f}, ...)"
    )


if __name__ == "__main__":
    main()
