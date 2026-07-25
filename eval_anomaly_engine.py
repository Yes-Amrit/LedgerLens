import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score,
)

from agent.nodes.anomaly_node import run_anomaly_detection
from data.loader import load_dataset

# ---------------------------------------------------------------------
# Ground-truth columns per dataset.
# ---------------------------------------------------------------------
LABEL_COLUMN = {
    "samld": "Is_laundering",
    "paysim": "isFraud",
}
ID_COLUMN = {
    "samld": "transaction_id",   
    "paysim": "transaction_id",
}
ACCT_COLUMN = {
    "samld": "Sender_account",
    "paysim": "nameOrig",
}


@dataclass
class EvalResult:
    target_pattern: str
    dataset: str
    split: str
    n_accounts_sampled: int
    n_total_rows: int
    n_true_positive_rows: int
    n_flagged: int
    precision: float
    recall: float
    f1: float
    pr_auc: Optional[float]
    tp: int
    fp: int
    fn: int
    tn: int
    method_used: str


def prepare_datasets(df: pd.DataFrame, label_col: str, acct_col: str,
                     n_positive: int, n_negative: int,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Samples by ACCOUNT rather than by row to preserve per-account features like
    rolling_7d_sum and velocity_24h. Then splits the sampled accounts 70/30 into
    tuning and held-out sets.
    """
    pos_accts = df[df[label_col] == 1][acct_col].unique()
    all_accts = df[acct_col].unique()
    neg_accts = np.setdiff1d(all_accts, pos_accts)

    if len(pos_accts) < n_positive:
        print(f"[WARN] requested {n_positive} positive accounts but only "
              f"{len(pos_accts)} exist in this slice — using all of them. "
              f"This typology/dataset may be too sparse.")
        n_positive = len(pos_accts)

    rng = np.random.RandomState(random_state)
    pos_sample = rng.choice(pos_accts, n_positive, replace=False)
    neg_sample = rng.choice(neg_accts, min(n_negative, len(neg_accts)), replace=False)

    # 70/30 split
    n_pos_tune = int(0.7 * len(pos_sample))
    n_neg_tune = int(0.7 * len(neg_sample))

    pos_tune = pos_sample[:n_pos_tune]
    pos_held = pos_sample[n_pos_tune:]

    neg_tune = neg_sample[:n_neg_tune]
    neg_held = neg_sample[n_neg_tune:]

    tune_accts = np.concatenate([pos_tune, neg_tune])
    held_accts = np.concatenate([pos_held, neg_held])

    df_tune = df[df[acct_col].isin(tune_accts)].copy().reset_index(drop=True)
    df_held = df[df[acct_col].isin(held_accts)].copy().reset_index(drop=True)

    print(f"--- Dataset Split Info ---")
    print(f"Total Accounts Sampled: {len(tune_accts) + len(held_accts)} "
          f"(Pos: {len(pos_sample)}, Neg: {len(neg_sample)})")
    print(f"Tuning Accounts: {len(tune_accts)} (Rows: {len(df_tune)})")
    print(f"Held-Out Accounts: {len(held_accts)} (Rows: {len(df_held)})")
    print(f"--------------------------\n")

    return df_tune, df_held


def filter_to_typology(df: pd.DataFrame, target_pattern: str,
                        typology_col: str = "Laundering_type") -> pd.DataFrame:
    """
    For structuring/layering evals on SAML-D, restrict the positive
    class to rows matching the target typology specifically.
    """
    with open("data/typology_mapping.json") as f:
        mapping = json.load(f)  
    matching_typologies = [k for k, v in mapping.items() if v == target_pattern]
    if not matching_typologies:
        raise ValueError(
            f"No typologies in typology_mapping.json map to "
            f"'{target_pattern}' — mapping may be incomplete."
        )
    is_target_typology = df[typology_col].isin(matching_typologies)
    return df[is_target_typology | (df[LABEL_COLUMN["samld"]] == 0)]


def evaluate(df: pd.DataFrame, target_pattern: str, dataset: str,
             id_col: str, label_col: str, split_name: str, n_accounts_sampled: int) -> EvalResult:
    result_dict = run_anomaly_detection(df, target_pattern)
    scores = result_dict["anomaly_scores"]          
    flagged_ids = {str(t["transaction_id"]) for t in result_dict["flagged_transactions"]}
    method_used = result_dict["method_used"]

    df = df.set_index(id_col)
    y_true = df[label_col].astype(int)

    # y_pred from the binary flagged set
    y_pred = y_true.index.astype(str).to_series().apply(lambda i: int(i in flagged_ids))

    # y_score from anomaly_scores, aligned to df order, for PR-AUC
    y_score = y_true.index.astype(str).to_series().apply(lambda i: scores.get(i, 0.0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = None
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        pr_auc = average_precision_score(y_true, y_score)

    return EvalResult(
        target_pattern=target_pattern,
        dataset=dataset,
        split=split_name,
        n_accounts_sampled=n_accounts_sampled,
        n_total_rows=len(df),
        n_true_positive_rows=int(y_true.sum()),
        n_flagged=len(flagged_ids),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        pr_auc=pr_auc,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        method_used=method_used,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True,
                         choices=["structuring", "layering", "cash_out", "none"])
    parser.add_argument("--dataset", required=True, choices=["samld", "paysim"])
    parser.add_argument("--split", required=True, choices=["tuning", "held-out"],
                         help="Which split of the data to evaluate on")
    parser.add_argument("--saml-path", default="data/SAML-D.csv")
    parser.add_argument("--paysim-path", default="data/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--n-positive", type=int, default=200,
                         help="Target count of true-positive ACCOUNTS in the eval sample")
    parser.add_argument("--n-negative", type=int, default=4800,
                         help="Count of true-negative ACCOUNTS in the eval sample")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    label_col = LABEL_COLUMN[args.dataset]
    id_col = ID_COLUMN[args.dataset]
    acct_col = ACCT_COLUMN[args.dataset]
    path = args.saml_path if args.dataset == "samld" else args.paysim_path

    # Read up to 1.5M rows to prevent OOM
    df = load_dataset(path, nrows=1500000)
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)

    if args.dataset == "samld" and args.pattern in {"structuring", "layering"}:
        df = filter_to_typology(df, args.pattern)

    df_tune, df_held = prepare_datasets(df, label_col, acct_col, args.n_positive, args.n_negative,
                                        args.random_state)

    df_eval = df_tune if args.split == "tuning" else df_held
    n_accts = len(df_eval[acct_col].unique())

    result = evaluate(df_eval, args.pattern, args.dataset, id_col, label_col, args.split, n_accts)

    print(json.dumps(asdict(result), indent=2))

    # Sanity flags
    if result.n_true_positive_rows < 20:
        print("\n[SANITY WARNING] Fewer than 20 true-positive rows in this "
              "split — precision/recall here are not statistically reliable.")
    if result.precision < 0.3 and result.n_flagged > 0:
        print("\n[SANITY WARNING] Low precision — detector is flagging "
              "mostly non-anomalies. Review threshold/rule parameters "
              "before treating this pattern as demo-ready.")


if __name__ == "__main__":
    main()
