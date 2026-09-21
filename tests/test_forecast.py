import numpy as np
import pandas as pd

from spendinsights.forecast import build_features, build_panel, walk_forward


def _monthly(n_students=12, n_months=18, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-09-01", periods=n_months, freq="MS")
    rows = []
    for s in range(n_students):
        for cat in ("dining", "groceries"):
            base = rng.uniform(50, 300)
            for m in months:
                rows.append((f"S{s}", m.strftime("%Y-%m-%d"), cat, base * rng.uniform(0.7, 1.3), 5))
    return pd.DataFrame(rows, columns=["student_id", "month", "category", "spend", "n_txns"])


def test_features_use_only_past_months():
    panel = build_panel(_monthly(), ["dining", "groceries"])
    base = build_features(panel)

    t = 10
    altered = panel.copy()
    altered.loc[altered.month_idx >= t, "spend"] += 1000.0   # change the present and the future
    new = build_features(altered)

    cols = ["lag1", "lag2", "lag3", "roll3", "roll6", "expanding_mean"]
    a = base[base.month_idx == t][cols].to_numpy()
    b = new[new.month_idx == t][cols].to_numpy()
    np.testing.assert_allclose(a, b)  # features at month t are unaffected by spend at t or later


def test_walk_forward_predictions_do_not_depend_on_the_target_month():
    panel = build_panel(_monthly(), ["dining", "groceries"])
    n_months = int(panel.month_idx.max()) + 1
    res1 = walk_forward(build_features(panel), n_months, n_test=2, seed=1)

    altered = panel.copy()
    altered.loc[altered.month_idx == n_months - 1, "spend"] *= 50  # corrupt only the last month's actuals
    res2 = walk_forward(build_features(altered), n_months, n_test=2, seed=1)

    last = n_months - 1
    p1 = res1[res1.month_idx == last].sort_values(["student_id", "category"]).p50.to_numpy()
    p2 = res2[res2.month_idx == last].sort_values(["student_id", "category"]).p50.to_numpy()
    np.testing.assert_allclose(p1, p2)  # prediction for month t never saw month t's spend


def test_intervals_are_ordered_and_nonnegative():
    panel = build_panel(_monthly(), ["dining", "groceries"])
    n_months = int(panel.month_idx.max()) + 1
    res = walk_forward(build_features(panel), n_months, n_test=2, seed=1)
    assert (res.p10 <= res.p90).all()
    assert (res.p10 >= 0).all() and (res.p50 >= 0).all()


def test_walk_forward_works_when_lag12_is_not_yet_available():
    # 14 months with 2 test months: the first training window has < 12 months of history, so the
    # lag12 column is entirely missing. Newer scikit-learn versions raise on all-NaN features.
    panel = build_panel(_monthly(n_months=14), ["dining", "groceries"])
    n_months = int(panel.month_idx.max()) + 1
    res = walk_forward(build_features(panel), n_months, n_test=2, seed=1)
    assert len(res) > 0
    assert res.p50.notna().all()
