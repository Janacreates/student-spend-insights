import numpy as np
import pandas as pd

from spendinsights.survival import crossing_table, warned_before_crossing

SEP = pd.Timestamp("2025-09-01")  # 30 days


def _txn(student, day, spend, category="dining", direction="debit"):
    return {"student_id": student, "date": SEP + pd.Timedelta(days=day - 1), "month": SEP,
            "category": category, "spend": spend, "direction": direction}


def _table(txns, budgets):
    forecasts = pd.DataFrame([{"student_id": s, "category": "dining", "month": SEP, "p50": b, "actual": a}
                              for s, b, a in budgets])
    students = pd.DataFrame({"student_id": [s for s, _, _ in budgets], "housing": "shared"})
    return crossing_table(pd.DataFrame(txns), forecasts, students).set_index("student_id")


def test_crossing_day_is_the_first_day_cumulative_spend_exceeds_the_budget():
    t = _table([_txn("A", 3, 40.0), _txn("A", 8, 70.0), _txn("A", 20, 5.0)], [("A", 100.0, 115.0)])
    assert t.loc["A", "cross_day"] == 8
    assert t.loc["A", "event"] == 1 and t.loc["A", "duration"] == 8


def test_months_that_never_cross_are_censored_at_month_end():
    # B spends exactly its budget (not over it); credits and other categories don't count
    t = _table([_txn("B", 5, 60.0), _txn("B", 9, 40.0), _txn("B", 10, 500.0, direction="credit"),
                _txn("B", 11, 500.0, category="rent")], [("B", 100.0, 100.0)])
    assert np.isnan(t.loc["B", "cross_day"])
    assert t.loc["B", "event"] == 0 and t.loc["B", "duration"] == 30


def test_near_zero_budgets_are_dropped_like_in_the_nudge_rule():
    t = _table([_txn("C", 2, 10.0)], [("C", 4.0, 10.0), ("D", 50.0, 0.0)])
    assert list(t.index) == ["D"]


def test_an_alert_only_counts_as_in_time_if_the_student_is_still_under_budget():
    flagged = pd.Series([True, True, True, False])
    overspend = pd.Series([True, True, False, True])
    cross_day = pd.Series([12.0, 8.0, np.nan, 20.0])
    # alert on day 10: row 0 crosses later (in time), row 1 already crossed, row 2 no overspend, row 3 not flagged
    assert warned_before_crossing(flagged, overspend, cross_day, 10).tolist() == [True, False, False, False]
