import argparse
import json
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score,
)

from agent.nodes.anomaly_node import run_anomaly_detection


# ---------------------------------------------------------------------
# Ground-truth columns per dataset. If your loader renames these during
# schema normalization, update here rather than downstream.
# ---------------------------------------------------------------------
LABEL_COLUMN = {
    "samld": "Is_laundering",
    "paysim": "isFraud",
}
ID_COLUMN = {
    "samld": "transaction_id",   # adjust if your loader names it differently
    "paysim": "transaction_id",
}


@dataclass
class EvalResult:
    target_pattern: str
    dataset: str
    n_total: int
    n_true_positive_in_sample: int
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


def stratified_sample(df: pd.DataFrame, label_col: str,
                       n_positive: int, n_negative: int,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Pull ALL available positives up to n_positive (warn if fewer exist —
    this itself is diagnostic: it tells you if a typology is too sparse
    to evaluate meaningfully) plus a random n_negative sample of
    negatives. This guarantees the eval set actually contains enough
    ground-truth anomalies to compute precision/recall on, unlike a
    flat random sample at SAML-D's ~0.1% base rate.
    """
    positives = df[df[label_col] == 1]
    negatives = df[df[label_col] == 0]

    if len(positives) < n_positive:
        print(f"[WARN] requested {n_positive} positives but only "
              f"{len(positives)} exist in this slice — using all of them. "
              f"This typology/dataset may be too sparse to train or "
              f"evaluate a dedicated detector on.")
    pos_sample = positives.sample(min(n_positive, len(positives)),
                                   random_state=random_state)
    neg_sample = negatives.sample(min(n_negative, len(negatives)),
                                   random_state=random_state)

    combined = pd.concat([pos_sample, neg_sample]).sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)
    return combined


def filter_to_typology(df: pd.DataFrame, target_pattern: str,
                        typology_col: str = "Laundering_type") -> pd.DataFrame:
    """
    For structuring/layering evals on SAML-D, restrict the positive
    class to rows matching the target typology specifically — otherwise
    "true positives" includes unrelated typologies your structuring
    detector was never meant to catch, which deflates recall
    artificially. Requires typology_mapping.json (from eda.py) to map
    target_pattern -> raw typology name(s) in the data.
    """
    with open("data/typology_mapping.json") as f:
        mapping = json.load(f)  # raw_typology_name -> bucket
    matching_typologies = [k for k, v in mapping.items() if v == target_pattern]
    if not matching_typologies:
        raise ValueError(
            f"No typologies in typology_mapping.json map to "
            f"'{target_pattern}' — mapping may be incomplete."
        )
    is_target_typology = df[typology_col].isin(matching_typologies)
    # keep: rows matching this typology (positives) + all clean rows (negatives)
    return df[is_target_typology | (df[LABEL_COLUMN["samld"]] == 0)]


def evaluate(df: pd.DataFrame, target_pattern: str, dataset: str,
             id_col: str, label_col: str) -> EvalResult:
    result_dict = run_anomaly_detection(df, target_pattern)
    scores = result_dict["anomaly_scores"]          # {transaction_id: float}
    flagged_ids = {t["transaction_id"] for t in result_dict["flagged_transactions"]}
    method_used = result_dict["method_used"]

    df = df.set_index(id_col)
    y_true = df[label_col].astype(int)

    # y_pred from the binary flagged set (what risk_node/escalation_node act on)
    y_pred = y_true.index.to_series().apply(lambda i: int(i in flagged_ids))

    # y_score from anomaly_scores, aligned to df order, for PR-AUC
    y_score = y_true.index.to_series().apply(lambda i: scores.get(i, 0.0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = None
    if y_true.sum() > 0 and y_true.sum() < len(y_true):
        pr_auc = average_precision_score(y_true, y_score)

    return EvalResult(
        target_pattern=target_pattern,
        dataset=dataset,
        n_total=len(df),
        n_true_positive_in_sample=int(y_true.sum()),
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
    parser.add_argument("--saml-path", default="data/SAML-D.csv")
    parser.add_argument("--paysim-path", default="data/PS_20174392719_1491204439457_log.csv")
    parser.add_argument("--n-positive", type=int, default=200,
                         help="Target count of true-positive rows in the eval sample")
    parser.add_argument("--n-negative", type=int, default=4800,
                         help="Count of true-negative rows in the eval sample")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    label_col = LABEL_COLUMN[args.dataset]
    id_col = ID_COLUMN[args.dataset]
    path = args.saml_path if args.dataset == "samld" else args.paysim_path

    # Read up to 1.5M rows to prevent OOM while guaranteeing enough positives (SAML-D 0.1% rate = ~1500 positives)
    df = pd.read_csv(path, nrows=1500000)
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)  # fallback if no explicit txn id exists

    if args.dataset == "samld" and args.pattern in {"structuring", "layering"}:
        df = filter_to_typology(df, args.pattern)

    sample = stratified_sample(df, label_col, args.n_positive, args.n_negative,
                                args.random_state)

    result = evaluate(sample, args.pattern, args.dataset, id_col, label_col)

    print(json.dumps(asdict(result), indent=2))

    # Sanity flags — these are the checks the last report skipped
    if result.n_true_positive_in_sample < 20:
        print("\n[SANITY WARNING] Fewer than 20 true positives in this "
              "sample — precision/recall here are not statistically "
              "reliable. Increase --n-positive or reconsider whether "
              "this typology has enough data for a dedicated detector.")
    if result.precision < 0.3 and result.n_flagged > 0:
        print("\n[SANITY WARNING] Low precision — detector is flagging "
              "mostly non-anomalies. Review threshold/rule parameters "
              "before treating this pattern as demo-ready.")


if __name__ == "__main__":
    main()
