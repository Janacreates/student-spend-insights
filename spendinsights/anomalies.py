"""Flag unusual transactions: transparent rules vs. an Isolation Forest, scored against injected anomalies.

Framed like an internal control test: which transactions should a human review? Detectors only
see cleaned transactions and *predicted* categories; the injected-anomaly labels are used
purely to measure precision and recall.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from .config import FIXED_CATEGORIES, Paths
from .viz import COLORS, plt, save, style

IF_FEATURES = ["log_spend", "robust_z", "log_ratio_cat", "log_merchant_global", "log_merchant_student", "is_dup"]


def _features(txns: pd.DataFrame) -> pd.DataFrame:
    d = txns[(txns.direction == "debit") & ~txns.category.isin(FIXED_CATEGORIES)].copy()
    d["merchant"] = d.description_norm

    med = d.groupby(["student_id", "category"]).spend.transform("median")
    mad = (d.spend - med).abs().groupby([d.student_id, d.category]).transform("median")
    d["robust_z"] = (d.spend - med) / (1.4826 * mad.clip(lower=1.0))
    d["ratio_cat"] = d.spend / d.groupby("category").spend.transform("median").clip(lower=1.0)
    d["merchant_global"] = d.groupby("merchant").txn_id.transform("count")
    d["merchant_student"] = d.groupby(["student_id", "merchant"]).txn_id.transform("count")

    d = d.sort_values(["student_id", "merchant", "spend", "date", "txn_id"])
    same = ((d.student_id == d.student_id.shift()) & (d.merchant == d.merchant.shift())
            & (d.spend == d.spend.shift()) & ((d.date - d.date.shift()).dt.days.abs() <= 1))
    d["is_dup"] = same.astype(int)  # flags the later charge of a same-merchant, same-amount pair

    d["log_spend"] = np.log1p(d.spend)
    d["log_ratio_cat"] = np.log(d.ratio_cat.clip(lower=1e-3))
    d["log_merchant_global"] = np.log(d.merchant_global)
    d["log_merchant_student"] = np.log(d.merchant_student)
    return d.sort_values("txn_id").reset_index(drop=True)


def _score(flag: np.ndarray, truth: np.ndarray, types: np.ndarray) -> dict:
    tp = int((flag & truth).sum())
    fp = int((flag & ~truth).sum())
    fn = int((~flag & truth).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    by_type = {t: float(flag[(types == t)].mean()) for t in ("spike", "duplicate", "unknown_merchant")
               if (types == t).any()}
    return {"flagged": int(flag.sum()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "recall_by_type": by_type}


def run_anomalies(paths: Paths, txns: pd.DataFrame, truth: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    style()
    d = _features(txns).merge(truth[["txn_id", "is_anomaly", "anomaly_type"]], on="txn_id", how="left")
    y = d.is_anomaly.fillna(False).to_numpy(dtype=bool)
    types = d.anomaly_type.fillna("").to_numpy()

    # rule-based detector: three explicit, explainable rules
    r_spike = (d.robust_z > 6).to_numpy()
    r_dup = d.is_dup.to_numpy(dtype=bool)
    r_rare = ((d.merchant_student == 1) & (d.merchant_global < 60) & (d.spend > 40)).to_numpy()
    rules = r_spike | r_dup | r_rare

    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=seed)
    iso.fit(d[IF_FEATURES])
    if_score = -iso.score_samples(d[IF_FEATURES])  # higher = more anomalous
    if_flag = iso.predict(d[IF_FEATURES]) == -1

    union = rules | if_flag
    per_rule = {}
    for name, mask, target in (("robust_z_spike", r_spike, "spike"), ("duplicate_pair", r_dup, "duplicate"),
                               ("rare_merchant", r_rare, "unknown_merchant")):
        per_rule[name] = {
            "flagged": int(mask.sum()),
            "precision": float((mask & y).sum() / max(mask.sum(), 1)),
            "recall_of_target_type": float((mask & (types == target)).sum() / max((types == target).sum(), 1)),
        }
    k = int(y.sum())
    top_k = np.zeros(len(d), dtype=bool)
    top_k[np.argsort(-if_score)[:k]] = True

    metrics = {
        "n_transactions_scored": int(len(d)),
        "n_injected": k,
        "injected_by_type": {t: int((types == t).sum()) for t in ("spike", "duplicate", "unknown_merchant")},
        "rules": _score(rules, y, types),
        "isolation_forest": _score(if_flag, y, types),
        "union": _score(union, y, types),
        "isolation_forest_top_k": _score(top_k, y, types),
        "isolation_forest_average_precision": float(average_precision_score(y, if_score)),
        "base_rate": float(y.mean()),
        "rules_by_rule": per_rule,
    }
    metrics["rules_precision_lift_vs_base_rate"] = float(metrics["rules"]["precision"] / metrics["base_rate"])

    reason = np.where(r_spike, "unusually large for this student/category",
             np.where(r_dup, "possible duplicate charge",
             np.where(r_rare, "rare merchant, first time for this student",
             np.where(if_flag, "statistical outlier (isolation forest)", ""))))
    flags = d.loc[union, ["txn_id", "student_id", "date", "description_norm", "category", "spend"]].copy()
    flags["reason"] = reason[union]
    flags["if_score"] = if_score[union]

    # figure: recall/precision per detector
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["Rules", "Isolation\nForest", "Rules +\nIsolation Forest"]
    keys = ["rules", "isolation_forest", "union"]
    x = np.arange(len(names))
    ax.bar(x - 0.18, [100 * metrics[k_]["precision"] for k_ in keys], 0.36, label="precision", color=COLORS["primary"])
    ax.bar(x + 0.18, [100 * metrics[k_]["recall"] for k_ in keys], 0.36, label="recall", color=COLORS["accent"])
    ax.set_xticks(x, names)
    ax.set_ylabel("%")
    ax.set_title("Anomaly detection vs. injected anomalies")
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, paths.figures / "fig_anomaly_detection.png")
    return flags.reset_index(drop=True), metrics
