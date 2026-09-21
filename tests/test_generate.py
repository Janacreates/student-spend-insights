import pandas as pd

from spendinsights.generate import generate


def test_generation_is_deterministic():
    a_raw, a_truth, _ = generate(n_students=8, n_months=6, seed=7)
    b_raw, b_truth, _ = generate(n_students=8, n_months=6, seed=7)
    pd.testing.assert_frame_equal(a_raw, b_raw)
    pd.testing.assert_frame_equal(a_truth, b_truth)


def test_different_seeds_differ():
    a, _, _ = generate(n_students=8, n_months=6, seed=1)
    b, _, _ = generate(n_students=8, n_months=6, seed=2)
    assert len(a) != len(b) or not a.equals(b)


def test_raw_export_is_messy_and_truth_is_separate():
    raw, truth, students = generate(n_students=20, n_months=8, seed=3)
    assert list(raw.columns) == ["txn_id", "student_id", "posted_date", "description", "amount", "txn_type"]
    assert "true_category" not in raw.columns          # labels never leak into the raw export
    assert set(truth.txn_id) == set(raw.txn_id)
    assert raw.txn_id.duplicated().any()                # exact duplicate export rows exist
    assert raw.posted_date.str.contains("/").any() and raw.posted_date.str.contains("-").any()
    assert raw.amount.str.contains(r"\$").any()
    assert truth.is_anomaly.any()
    assert students.student_id.nunique() == 20
