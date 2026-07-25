"""
Task 2 — Feature engineering for the anomaly detectors.

Computes per-account features that all 3 detectors consume:
  1. rolling_7d_sum    — rolling 7-day transaction sum per account
  2. velocity_24h      — transaction count in the last 24 hours per account
  3. amount_deviation  — each transaction's amount minus the account's historical mean

Works on both SAML-D (Sender_account / Amount / timestamp) and
PaySim  (nameOrig    / amount    / step).
"""

import pandas as pd
import numpy as np


# ── Column-name aliases ───────────────────────────────────────────────────────
def _resolve(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column name that exists in df, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-account rolling features to df and return the augmented DataFrame.

    New columns added:
        rolling_7d_sum   float  — rolling sum of Amount over the last 7 days (per account)
        velocity_24h     int    — count of transactions in the last 24 hours (per account)
        amount_deviation float  — (this_amount - account_mean_amount)

    The original columns are preserved unchanged.

    Args:
        df: DataFrame with at least an amount column and (optionally) a timestamp
            or step column for time-based rolling features.

    Returns:
        Augmented DataFrame. A copy is made so the caller's df is not mutated.
    """
    df = df.copy()

    # ── Resolve schema ────────────────────────────────────────────────────────
    acct_col   = _resolve(df, ["Sender_account", "nameOrig", "account_id"])
    amount_col = _resolve(df, ["Amount", "amount"])
    time_col   = _resolve(df, ["Timestamp", "timestamp", "Time", "step"])

    if amount_col is None:
        # Nothing useful to compute; return df unchanged
        return df

    amt = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)

    # ── Feature 3: amount deviation from account mean ─────────────────────────
    if acct_col is not None:
        acct_mean = df.groupby(acct_col)[amount_col].transform(
            lambda x: pd.to_numeric(x, errors="coerce").mean()
        )
        df["amount_deviation"] = amt - acct_mean.fillna(amt.mean())
    else:
        df["amount_deviation"] = amt - amt.mean()

    # ── Time-based features require a usable timestamp ────────────────────────
    ts = None
    if time_col == "Timestamp" and "Timestamp" in df.columns:
        ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    elif time_col == "timestamp" and "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    elif time_col == "step" and "step" in df.columns:
        # PaySim 'step' is in hours — convert to a fake datetime for rolling
        ts = pd.to_datetime(df["step"], unit="h", origin="unix", errors="coerce")

    if ts is None or acct_col is None:
        # Can't compute rolling features without time + account; fill with NaN
        df["rolling_7d_sum"] = np.nan
        df["velocity_24h"]   = np.nan
        df["unique_counterparties_7d"] = np.nan
        return df

    dest_col = _resolve(df, ["Receiver_account", "nameDest", "receiver_id"])

    df = df.copy()
    df["_ts_sort"] = ts
    df = df.sort_values([acct_col, "_ts_sort"])

    rolling_7d  = []
    velocity_24 = []
    unique_counterparties = []

    SEVEN_DAYS_NS = np.timedelta64(7, 'D').astype('int64')
    ONE_DAY_NS    = np.timedelta64(1, 'D').astype('int64')

    for _, group in df.groupby(acct_col, sort=False):
        g_ts  = group["_ts_sort"].values.astype("datetime64[ns]").astype("int64")
        g_amt = pd.to_numeric(group[amount_col], errors="coerce").fillna(0.0).values
        g_dest = group[dest_col].values if dest_col else None
        n = len(group)

        r7  = np.zeros(n)
        v24 = np.zeros(n, dtype=int)
        uc7 = np.zeros(n, dtype=int)

        for i in range(n):
            t_i = g_ts[i]
            if t_i == pd.NaT.value:  # NaT has a specific int64 sentinel
                r7[i]  = g_amt[i]
                v24[i] = 1
                uc7[i] = 1 if g_dest is not None else 0
                continue
            # Vectorized: compare int64 timestamps directly (nanoseconds)
            diffs = t_i - g_ts  # positive means g_ts[j] is before t_i
            mask_7d  = (diffs >= 0) & (diffs <= SEVEN_DAYS_NS)
            mask_24h = (diffs >= 0) & (diffs <= ONE_DAY_NS)
            r7[i]  = g_amt[mask_7d].sum()
            v24[i] = int(mask_24h.sum())
            if g_dest is not None:
                # Calculate unique counterparties in 7d window
                # Convert the masked slice to a set for unique count
                uc7[i] = len(set(g_dest[mask_7d]))
            else:
                uc7[i] = 0

        rolling_7d.extend(r7.tolist())
        velocity_24.extend(v24.tolist())
        unique_counterparties.extend(uc7.tolist())

    df["rolling_7d_sum"] = rolling_7d
    df["velocity_24h"]   = velocity_24
    df["unique_counterparties_7d"] = unique_counterparties
    df = df.drop(columns=["_ts_sort"]).sort_index()

    return df
