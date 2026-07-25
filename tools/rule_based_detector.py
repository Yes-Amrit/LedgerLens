import pandas as pd
from typing import List, Dict

def detect_structuring(df: pd.DataFrame, threshold: float = 7258.49, window_days: int = 1) -> dict:
    """
    Rule-based detector for structuring (smurfing).

    v3 — data-driven redesign based on ground-truth analysis of SAML-D.
    """
    if df.empty or 'transaction_id' not in df.columns:
        return {"flagged_transactions": [], "anomaly_scores": {}, "method_used": "rule_based"}

    flagged: List[dict] = []
    scores: Dict[str, float] = {}

    try:
        # ── Schema detection ──────────────────────────────────────────────────
        amount_col = 'Amount' if 'Amount' in df.columns else 'amount'
        time_col = 'Timestamp' if 'Timestamp' in df.columns else 'timestamp'
        
        # FIX: Added 'account_id' to the list of accepted column names
        acct_col = None
        for col in ['Sender_account', 'nameOrig', 'account_id']:
            if col in df.columns:
                acct_col = col
                break

        if not acct_col or amount_col not in df.columns or time_col not in df.columns:
            return {"flagged_transactions": [], "anomaly_scores": {}, "method_used": "rule_based"}

        df_work = df.copy()
        df_work['_amt'] = pd.to_numeric(df_work[amount_col], errors='coerce').fillna(0)
        df_work['_t'] = pd.to_datetime(df_work[time_col], errors='coerce')
        df_work = df_work.dropna(subset=['_t'])

        # ── Parameters ─────────────────
        struct_low      = 1_000     
        struct_high     = 9_500     
        reporting_target = 10_000
        sum_low         = reporting_target * 0.70
        sum_high        = reporting_target * 1.50 
        spec_low        = threshold * 0.90
        spec_high       = threshold
        time_window     = pd.Timedelta(days=window_days)

        # ── Per-account velocity ──────────────────────────────────────────────
        acct_tx_count = df_work.groupby(acct_col)['transaction_id'].transform('count')

        # ── Score every transaction ───────────────────────────────────────────
        raw_scores: Dict[str, float] = {}

        for idx, row in df_work.iterrows():
            amt   = row['_amt']
            tx_id = str(row['transaction_id'])
            score = 0.0

            # Signal 1: Amount in structuring zone (0.4 weight)
            if struct_low <= amt < struct_high:
                score += 0.4 * ((amt - struct_low) / (struct_high - struct_low))

            # Signal 2: High velocity for this account (0.3 weight)
            n_acct = acct_tx_count.loc[idx]
            if n_acct >= 3:
                score += 0.3
            elif n_acct == 2:
                score += 0.15

            raw_scores[tx_id] = score

        # ── Rolling window boost & Spec Rule ──────────────────────────────────
        df_sorted = df_work.sort_values(by=[acct_col, '_t'])
        for account, group in df_sorted.groupby(acct_col):
            if len(group) < 2:
                continue
            records = group.reset_index(drop=True)
            amounts = records['_amt'].values
            times   = records['_t'].values
            tx_ids  = [str(x) for x in records['transaction_id'].values]

            for i in range(len(records)):
                ws, window_ids = amounts[i], [tx_ids[i]]
                
                spec_count = 1 if (spec_low <= amounts[i] <= spec_high) else 0
                spec_window_ids = [tx_ids[i]] if spec_count == 1 else []

                for j in range(i + 1, len(records)):
                    if times[j] - times[i] > time_window:
                        break
                    
                    ws += amounts[j]
                    window_ids.append(tx_ids[j])
                    
                    if spec_low <= amounts[j] <= spec_high:
                        spec_count += 1
                        spec_window_ids.append(tx_ids[j])

                    if sum_low <= ws <= sum_high and len(window_ids) >= 2:
                        for tid in window_ids:
                            raw_scores[tid] = min(1.0, raw_scores.get(tid, 0.0) + 0.3)
                            
                    if spec_count >= 3:
                        for tid in spec_window_ids:
                            raw_scores[tid] = 1.0

        # ── Build flagged list ──
        FLAG_THRESHOLD = 0.40
        for idx, row in df_work.iterrows():
            tx_id = str(row['transaction_id'])
            sc    = raw_scores.get(tx_id, 0.0)
            scores[tx_id] = sc

            if sc >= FLAG_THRESHOLD:
                amt = row['_amt']
                reason_parts = []
                
                if struct_low <= amt < struct_high:
                    reason_parts.append(f"amount {amt:.2f} in structuring zone [{struct_low},{struct_high:.2f})")
                if acct_tx_count.loc[idx] >= 2:
                    reason_parts.append(f"account has {acct_tx_count.loc[idx]} transactions")
                if sc == 1.0 and spec_low <= amt <= spec_high:
                    reason_parts.append(f"spec rule: 3+ txns in [{spec_low:.2f},{spec_high:.2f}] in window")

                flagged.append({
                    "transaction_id": tx_id,
                    "account_id":     str(row.get(acct_col, 'unknown')),
                    "amount":         float(amt),
                    "timestamp":      row.get('_t', 0),
                    "reason_features": {"structuring_rule": "; ".join(reason_parts)},
                })

    except Exception as exc:
        import logging
        logging.error(f"[rule_based_detector] failed: {exc}", exc_info=True)

    return {
        "flagged_transactions": flagged,
        "anomaly_scores":       scores,
        "method_used":          "rule_based",
    }