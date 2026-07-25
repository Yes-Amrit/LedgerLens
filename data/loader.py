import os
import logging
import pandas as pd

# Critical columns that must be non-null for any detector to work
_CRITICAL_COLUMNS = ["Amount", "Sender_account", "Receiver_account", "Is_laundering", "Laundering_type"]


def load_dataset(path: str = "data/sample_saml_d.csv", nrows: int | None = None) -> pd.DataFrame:
    """
    Load a LedgerLens dataset CSV and return a clean DataFrame.

    - Combines SAML-D's separate Date + Time columns into a single `Timestamp` column.
    - Generates a stable string `transaction_id` from the row index (SAML-D has no ID column).
    - Drops rows with nulls in any critical column.

    Args:
        path:  Path to the CSV file.
        nrows: Optional row cap (useful for tests / threshold analysis).

    Returns:
        Clean pd.DataFrame with a `Timestamp` and `transaction_id` column guaranteed.
    """
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, nrows=nrows)

    # Combine Date + Time into a single Timestamp column
    if "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce",
        )

    # Stable transaction_id (row index as string)
    if "transaction_id" not in df.columns:
        df.insert(0, "transaction_id", df.index.astype(str))

    # Drop rows with nulls in critical columns
    critical_present = [c for c in _CRITICAL_COLUMNS if c in df.columns]
    before = len(df)
    if critical_present:
        df = df.dropna(subset=critical_present).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logging.info(f"[loader] Dropped {dropped:,} rows with nulls in {critical_present}")

    return df