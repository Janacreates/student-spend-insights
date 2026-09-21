"""Transaction categorisation: hand-written rules vs. a text classifier trained on a small labelled sample.

Setup mirrors a realistic workflow: only ~10% of transactions carry a trusted label; a model
labels the rest and *abstains* when it is not confident. Ground-truth labels for the remaining
transactions exist only because the data is synthetic, so we can measure how good the labels are.

Two evaluations are reported:
  * random split  - most test merchants were seen in training
  * unseen merchants - test merchants never appear in training (the harder, more honest case)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# Ordered: earlier rules win. Written the way an analyst would from the merchants they know,
# including generic keywords (CAFE, PIZZA, PHARMACY...) so the baseline is not a strawman.
# CAVEAT: I wrote these while looking at the simulator's merchant list, so on the 'unseen
# merchants' test they are optimistic - a real analyst would not know the unseen merchants.
RULES: List[Tuple[str, str]] = [
    ("FROM PARENT", "income"), ("PAYROLL", "income"), ("BURSARY", "income"),
    ("SCHOLARSHIP", "income"), ("AID DEPOSIT", "income"),
    ("TUITION", "tuition"), ("RENT", "rent"),
    ("UBER *EATS", "dining"), ("DOORDASH", "dining"), ("SKIPTHEDISHES", "dining"),
    ("TIM HORTONS", "dining"), ("STARBUCKS", "dining"), ("MCDONALDS", "dining"),
    ("SUBWAY", "dining"), ("A&W", "dining"), ("PIZZA", "dining"), ("SUSHI", "dining"),
    ("CAFE", "dining"), ("BISTRO", "dining"), ("RESTAURANT", "dining"), ("BURGER", "dining"),
    ("UBER", "transport"), ("LYFT", "transport"), ("PETRO", "transport"), ("SHELL", "transport"),
    ("ESSO", "transport"), ("GAS BAR", "transport"), ("TRANSIT", "transport"),
    ("PARKING", "transport"), ("VIA RAIL", "transport"),
    ("NETFLIX", "subscriptions"), ("SPOTIFY", "subscriptions"), ("APPLE.COM/BILL", "subscriptions"),
    ("YOUTUBE PREMIUM", "subscriptions"), ("PRIME", "subscriptions"), ("DISNEY", "subscriptions"),
    ("ADOBE", "subscriptions"), ("FITNESS", "subscriptions"), ("NORDVPN", "subscriptions"),
    ("OPENAI", "subscriptions"), ("CRUNCHYROLL", "subscriptions"),
    ("SAFEWAY", "groceries"), ("SUPERSTORE", "groceries"), ("WALMART", "groceries"),
    ("SOBEYS", "groceries"), ("COSTCO", "groceries"), ("FRESHCO", "groceries"),
    ("NO FRILLS", "groceries"), ("SUPERMARKET", "groceries"), ("FOODS", "groceries"),
    ("SHOPPERS", "health"), ("LONDON DRUGS", "health"), ("PHARMACY", "health"),
    ("CLINIC", "health"), ("DENTAL", "health"),
    ("BOOKSTORE", "education"), ("CHEGG", "education"), ("PEARSON", "education"),
    ("COURSERA", "education"), ("STAPLES", "education"), ("PRINT", "education"),
    ("AMAZON", "shopping"), ("ZARA", "shopping"), ("H&M", "shopping"), ("BEST BUY", "shopping"),
    ("WINNERS", "shopping"), ("IKEA", "shopping"), ("DOLLARAMA", "shopping"),
    ("CINEPLEX", "entertainment"), ("STEAM", "entertainment"), ("PLAYSTATION", "entertainment"),
    ("TICKETMASTER", "entertainment"), ("BOWL", "entertainment"), ("LIVE NATION", "entertainment"),
]

CONF_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def rule_based(desc: str) -> str:
    for keyword, cat in RULES:
        if keyword in desc:
            return cat
    return "uncategorized"


def make_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, C=20)),
    ])


def _predict(model: Pipeline, texts: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    proba = model.predict_proba(texts)
    return model.classes_[proba.argmax(axis=1)], proba.max(axis=1)


def _group_key(desc: pd.Series) -> pd.Series:
    """Proxy for merchant identity using only the text: spaces removed, first 5 characters.

    Keeps spelling variants of one merchant ('TIM HORTONS', 'TIMHORTONS', truncated forms) in the
    same group without needing the true merchant id, which a real pipeline would not have.
    """
    return desc.str.replace(" ", "", regex=False).str[:5]


def _pick_threshold(labeled: pd.DataFrame, seed: int, target_acc: float = 0.95) -> float:
    """Smallest confidence threshold whose confident predictions reach `target_acc`.

    Validation holds out whole merchant groups (25%), not random rows, so the threshold reflects
    behaviour on merchants the model has not seen. A random validation split would leak merchants
    and give an over-optimistic (too low) threshold.
    """
    rng = np.random.default_rng(seed)
    key = _group_key(labeled.description_norm)
    groups = key.unique()
    held = set(rng.choice(groups, size=max(1, int(0.25 * len(groups))), replace=False))
    va_mask = key.isin(held)
    tr, va = labeled[~va_mask], labeled[va_mask]
    model = make_model().fit(tr.description_norm, tr.true_category)
    pred, conf = _predict(model, va.description_norm)
    for thr in CONF_GRID:
        sel = conf >= thr
        if sel.sum() >= 30 and (pred[sel] == va.true_category.to_numpy()[sel]).mean() >= target_acc:
            return thr
    return CONF_GRID[-1]


def _report(y_true, y_rule, y_model, conf, thr) -> Dict[str, float]:
    covered = conf >= thr
    rule_hit = y_rule != "uncategorized"
    # hybrid: precise keyword rules first, then the model if confident, else abstain
    y_hyb = np.where(rule_hit, y_rule, np.where(covered, y_model, "uncategorized"))
    hyb_hit = y_hyb != "uncategorized"
    return {
        "n_test": int(len(y_true)),
        "rules_coverage": float(rule_hit.mean()),
        "rules_accuracy_on_covered": float(accuracy_score(y_true[rule_hit], y_rule[rule_hit])) if rule_hit.any() else 0.0,
        "model_accuracy_all": float(accuracy_score(y_true, y_model)),
        "model_macro_f1": float(f1_score(y_true, y_model, average="macro")),
        "model_threshold": float(thr),
        "model_coverage": float(covered.mean()),
        "model_accuracy_on_covered": float(accuracy_score(y_true[covered], y_model[covered])) if covered.any() else 0.0,
        "hybrid_coverage": float(hyb_hit.mean()),
        "hybrid_accuracy_on_covered": float(accuracy_score(y_true[hyb_hit], y_hyb[hyb_hit])) if hyb_hit.any() else 0.0,
        "hybrid_correct_share_of_all": float((y_hyb == y_true).mean()),
    }


def _coverage_curve(y_true, y_model, conf) -> List[Dict[str, float]]:
    out = []
    for thr in CONF_GRID:
        sel = conf >= thr
        out.append({
            "threshold": thr,
            "coverage": float(sel.mean()),
            "accuracy": float(accuracy_score(y_true[sel], y_model[sel])) if sel.any() else float("nan"),
        })
    return out


def run_categorization(txns: pd.DataFrame, truth: pd.DataFrame, seed: int = 42,
                       label_frac: float = 0.10) -> Tuple[pd.DataFrame, dict]:
    """Label every transaction; return (predictions, metrics).

    predictions columns: txn_id, category, confidence, category_source
    """
    df = txns.merge(truth[["txn_id", "true_category", "merchant"]], on="txn_id", how="left")
    known = df.description_norm != "UNKNOWN"
    rng = np.random.default_rng(seed)

    # ---- main workflow: ~10% labelled sample -> model labels the rest ----
    labeled_mask = known & (rng.random(len(df)) < label_frac)
    labeled = df[labeled_mask]
    thr = _pick_threshold(labeled, seed)
    model = make_model().fit(labeled.description_norm, labeled.true_category)
    pred, conf = _predict(model, df.description_norm)

    rule_pred = np.array([rule_based(d) for d in df.description_norm], dtype=object)
    confident = conf >= thr
    rule_hit = rule_pred != "uncategorized"
    category = np.where(rule_hit, rule_pred, np.where(confident, pred, "uncategorized")).astype(object)
    source = np.where(rule_hit, "rule", np.where(confident, "model", "abstain")).astype(object)
    category[labeled_mask.to_numpy()] = df.true_category[labeled_mask].to_numpy()
    source[labeled_mask.to_numpy()] = "labeled"
    category[~known.to_numpy()] = "uncategorized"
    source[~known.to_numpy()] = "unknown_description"
    preds = pd.DataFrame({"txn_id": df.txn_id, "category": category,
                          "confidence": np.where(labeled_mask, 1.0, conf), "category_source": source})

    # ---- evaluation 1: rows the model had to label itself ----
    test = df[known & ~labeled_mask]
    y = test.true_category.to_numpy()
    y_rule = np.array([rule_based(d) for d in test.description_norm], dtype=object)
    y_mod, c_mod = _predict(model, test.description_norm)
    random_split = _report(y, y_rule, y_mod, c_mod, thr)
    random_split["coverage_curve"] = _coverage_curve(y, y_mod, c_mod)

    # ---- evaluation 2: merchants the model has never seen ----
    merchants = df[known].groupby("true_category").merchant.unique()
    unseen = set()
    for cat, ms in merchants.items():
        ms = list(ms)
        k = max(1, int(round(0.25 * len(ms))))
        unseen.update(rng.choice(ms, size=k, replace=False))
    is_unseen = df.merchant.isin(unseen)
    pool = df[known & ~is_unseen]
    lab2 = pool[rng.random(len(pool)) < label_frac]
    thr2 = _pick_threshold(lab2, seed)
    model2 = make_model().fit(lab2.description_norm, lab2.true_category)
    test2 = df[known & is_unseen]
    y2 = test2.true_category.to_numpy()
    y2_rule = np.array([rule_based(d) for d in test2.description_norm], dtype=object)
    y2_mod, c2_mod = _predict(model2, test2.description_norm)
    unseen_eval = _report(y2, y2_rule, y2_mod, c2_mod, thr2)
    unseen_eval["n_unseen_merchants"] = len(unseen)
    unseen_eval["coverage_curve"] = _coverage_curve(y2, y2_mod, c2_mod)

    # ---- ambiguous merchants: same brand, different category ----
    amb = {}
    for name in ["UBER *EATS", "UBER *TRIP", "AMAZON.CA", "AMAZON PRIME*MEMBERSHIP",
                 "APPLE STORE", "APPLE.COM/BILL"]:
        sub = test[test.merchant == name]
        if len(sub) == 0:
            continue
        pm, _ = _predict(model, sub.description_norm)
        pr = np.array([rule_based(d) for d in sub.description_norm], dtype=object)
        amb[name] = {"n": int(len(sub)),
                     "rules_accuracy": float((pr == sub.true_category.to_numpy()).mean()),
                     "model_accuracy": float((pm == sub.true_category.to_numpy()).mean())}

    metrics = {
        "labeled_rows": int(labeled_mask.sum()),
        "labeled_fraction": float(labeled_mask.sum() / len(df)),
        "random_split": random_split,
        "unseen_merchants": unseen_eval,
        "ambiguous_merchants": amb,
        "abstain_rate_overall": float((preds.category_source == "abstain").mean()),
    }
    return preds, metrics
