"""
Task 1 — Dataset loader for LedgerLens.

Handles SAML-D schema: Date + Time → timestamp, drops nulls in critical columns.
"""

from pathlib import Path
import pandas as pd

# Critical columns that must be non-null for any detector to work
_SAML_D_CRITICAL = ["Sender_account", "Amount", "Is_laundering", "Laundering_type"]


def load_dataset(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load a LedgerLens dataset CSV and return a clean DataFrame.

    - Combines SAML-D's separate Date + Time columns into a single `timestamp` (UTC-naive).
    - Generates a stable string `transaction_id` from the row index (SAML-D has no ID column).
    - Drops rows with nulls in any critical column (Amount, Sender_account, etc.).

    Args:
        path:   Path to the CSV file.
        nrows:  Optional row cap (useful for tests / threshold analysis).

    Returns:
        Clean pd.DataFrame with a `transaction_id` column guaranteed.
    """
    df = pd.read_csv(path, nrows=nrows)

    # ── SAML-D: combine Date + Time into a single timestamp ──────────────────
    if "Date" in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",          # coerce malformed rows to NaT, not a crash
            utc=False,                # SAML-D timestamps are tz-naive; keep consistent
        )
        df = df.drop(columns=["Date", "Time"])

    # ── Stable transaction_id (row index as string) ───────────────────────────
    if "transaction_id" not in df.columns:
        df.insert(0, "transaction_id", df.index.astype(str))

    # ── Drop rows with nulls in critical columns ──────────────────────────────
    critical_present = [c for c in _SAML_D_CRITICAL if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_present).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        import logging
        logging.info(f"[loader] Dropped {dropped:,} rows with nulls in {critical_present}")

    return df
