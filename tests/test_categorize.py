import pandas as pd

from spendinsights.categorize import rule_based, run_categorization
from spendinsights.clean import clean
from spendinsights.generate import generate


def test_rules_handle_ambiguous_brands():
    assert rule_based("UBER *EATS") == "dining"
    assert rule_based("UBER *TRIP") == "transport"
    assert rule_based("E-TRANSFER FROM PARENT") == "income"   # contains 'RENT' but must not be rent
    assert rule_based("SOME UNKNOWN PLACE") == "uncategorized"


def test_run_categorization_labels_every_transaction():
    raw, truth, _ = generate(n_students=30, n_months=8, seed=5)
    txns, _ = clean(raw)
    preds, metrics = run_categorization(txns, truth, seed=5)
    assert set(preds.txn_id) == set(txns.txn_id)
    assert preds.category.notna().all()
    assert set(preds.category_source) <= {"labeled", "rule", "model", "abstain", "unknown_description"}
    assert 0.05 < metrics["labeled_fraction"] < 0.15
    # unknown descriptions must never be auto-categorised
    unknown = txns.description_norm == "UNKNOWN"
    assert (preds.set_index("txn_id").loc[txns[unknown].txn_id, "category"] == "uncategorized").all()
    assert metrics["unseen_merchants"]["hybrid_coverage"] <= 1.0
