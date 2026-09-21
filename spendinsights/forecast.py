"""Next-month spend forecasts per student and category, with calibrated prediction intervals.

Protocol (no leakage): for each test month t, models are trained only on target months <= t-2,
the interval is calibrated on month t-1, and month t is predicted. Baselines use the same rows.

Intervals: gradient-boosted 10%/90% quantile models, then conformalised quantile regression
(CQR) using the calibration month, so the nominal 80% interval is checked against reality.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import VARIABLE_CATEGORIES, Paths
from .viz import COLORS, plt, save, style

FEATURES = ["lag1", "lag2", "lag3", "lag12", "roll3", "roll6", "expanding_mean",
            "moy_sin", "moy_cos", "cat_code"]
MIN_HISTORY = 3  # first target month that has lag1..lag3


def build_panel(monthly: pd.DataFrame, categories: List[str] = None) -> pd.DataFrame:
    """Full student x category x month grid of spend (zeros where nothing was spent)."""
    categories = categories or VARIABLE_CATEGORIES
    monthly = monthly.copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    months = sorted(monthly.month.unique())
    students = sorted(monthly.student_id.unique())
    idx = pd.MultiIndex.from_product([students, categories, months], names=["student_id", "category", "month"])
    sub = monthly[monthly.category.isin(categories)].set_index(["student_id", "category", "month"]).spend
    panel = sub.reindex(idx, fill_value=0.0).reset_index()
    panel["month_idx"] = panel.month.map({m: i for i, m in enumerate(months)})
    return panel


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Lag/rolling features. Every feature for month t uses only months < t (shifted first)."""
    p = panel.sort_values(["student_id", "category", "month_idx"]).copy()
    g = p.groupby(["student_id", "category"])["spend"]
    for k in (1, 2, 3, 12):
        p[f"lag{k}"] = g.shift(k)
    p["roll3"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    p["roll6"] = g.transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    p["expanding_mean"] = g.transform(lambda s: s.shift(1).expanding().mean())
    moy = p.month.dt.month
    p["moy_sin"] = np.sin(2 * np.pi * moy / 12)
    p["moy_cos"] = np.cos(2 * np.pi * moy / 12)
    p["cat_code"] = p.category.astype("category").cat.codes
    return p


def _fit(X: pd.DataFrame, y: pd.Series, seed: int, **kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=30,
        random_state=seed, **kw).fit(X, y)


def walk_forward(feat: pd.DataFrame, n_months: int, n_test: int, seed: int = 42) -> pd.DataFrame:
    frames = []
    for t in range(n_months - n_test, n_months):
        train = feat[(feat.month_idx >= MIN_HISTORY) & (feat.month_idx <= t - 2)]
        cal = feat[feat.month_idx == t - 1]
        test = feat[feat.month_idx == t].copy()
        # a feature that is entirely missing in this training window (e.g. lag12 with < 12 months of
        # history) carries no information, and newer scikit-learn versions reject such columns
        feats = [c for c in FEATURES if train[c].notna().any()]
        m50 = _fit(train[feats], train.spend, seed, loss="absolute_error")
        m10 = _fit(train[feats], train.spend, seed, loss="quantile", quantile=0.1)
        m90 = _fit(train[feats], train.spend, seed, loss="quantile", quantile=0.9)

        # CQR: widen (or shrink) the raw interval by the (1-alpha) quantile of calibration scores
        y_cal = cal.spend.to_numpy()
        scores = np.maximum(m10.predict(cal[feats]) - y_cal, y_cal - m90.predict(cal[feats]))
        n = len(scores)
        qhat = np.quantile(scores, min(1.0, np.ceil((n + 1) * 0.8) / n), method="higher")

        lo_raw = np.maximum(m10.predict(test[feats]), 0)
        hi_raw = np.maximum(m90.predict(test[feats]), lo_raw)
        test["p50"] = np.maximum(m50.predict(test[feats]), 0)
        test["p10_raw"], test["p90_raw"] = lo_raw, hi_raw
        test["p10"], test["p90"] = np.maximum(lo_raw - qhat, 0), hi_raw + qhat
        test["qhat"] = qhat
        test["base_last_month"] = test.lag1
        test["base_roll3"] = test.roll3
        test["base_seasonal_naive"] = test.lag12.fillna(test.roll3)
        frames.append(test)
    return pd.concat(frames, ignore_index=True)


def _mae(y, p) -> float:
    return float(np.mean(np.abs(y - p)))


def _wape(y, p) -> float:
    return float(np.abs(y - p).sum() / y.sum())


def evaluate(res: pd.DataFrame) -> dict:
    y = res.spend.to_numpy()
    methods = {"last_month": "base_last_month", "rolling_3m_mean": "base_roll3",
               "seasonal_naive": "base_seasonal_naive", "gradient_boosting": "p50"}
    out: dict = {"n_predictions": int(len(res)),
                 "test_months": [str(m.date()) for m in sorted(res.month.unique())]}
    out["mae"] = {k: _mae(y, res[c].to_numpy()) for k, c in methods.items()}
    out["wape"] = {k: _wape(y, res[c].to_numpy()) for k, c in methods.items()}
    best_base = min(v for k, v in out["mae"].items() if k != "gradient_boosting")
    out["mae_improvement_vs_best_baseline_pct"] = float(100 * (best_base - out["mae"]["gradient_boosting"]) / best_base)
    out["mae_by_category"] = {
        cat: {k: _mae(g.spend.to_numpy(), g[c].to_numpy()) for k, c in methods.items()}
        for cat, g in res.groupby("category")}
    cover = lambda lo, hi: float(((res.spend >= res[lo]) & (res.spend <= res[hi])).mean())  # noqa: E731
    out["interval_nominal"] = 0.80
    out["interval_coverage_raw"] = cover("p10_raw", "p90_raw")
    out["interval_coverage_conformal"] = cover("p10", "p90")
    out["interval_mean_width_raw"] = float((res.p90_raw - res.p10_raw).mean())
    out["interval_mean_width_conformal"] = float((res.p90 - res.p10).mean())
    return out


def plot_forecasts(paths: Paths, res: pd.DataFrame, metrics: dict, feat: pd.DataFrame) -> None:
    style()
    # accuracy by method
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    names = list(metrics["mae"])
    vals = [metrics["mae"][n] for n in names]
    colors = [COLORS["grey"]] * (len(names) - 1) + [COLORS["primary"]]
    ax.bar([n.replace("_", "\n") for n in names], vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"${v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("MAE ($ per student-category-month)")
    ax.set_title("Next-month forecast error (walk-forward, lower is better)")
    ax.grid(axis="x", visible=False)
    save(fig, paths.figures / "fig_forecast_accuracy.png")

    # interval coverage
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    labels = ["Raw quantile\nmodels", "After conformal\ncalibration"]
    cov = [metrics["interval_coverage_raw"], metrics["interval_coverage_conformal"]]
    ax.bar(labels, [100 * c for c in cov], color=[COLORS["grey"], COLORS["primary"]])
    ax.axhline(80, color=COLORS["accent"], linestyle="--")
    ax.text(1.45, 81, "target 80%", color=COLORS["accent"], ha="right", fontsize=9)
    for i, c in enumerate(cov):
        ax.text(i, 100 * c + 1, f"{100 * c:.1f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Actual spend inside interval (%)")
    ax.set_title("Do the 80% intervals deserve the label?")
    ax.grid(axis="x", visible=False)
    save(fig, paths.figures / "fig_interval_coverage.png")

    # one example series: the median-spending student's dining
    dining = res[res.category == "dining"]
    by_student = dining.groupby("student_id").spend.mean().sort_values()
    sid = by_student.index[len(by_student) // 2]
    hist = feat[(feat.student_id == sid) & (feat.category == "dining")].sort_values("month")
    fut = dining[dining.student_id == sid].sort_values("month")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.month, hist.spend, color=COLORS["grey"], marker="o", markersize=3, label="actual")
    ax.plot(fut.month, fut.p50, color=COLORS["primary"], marker="o", label="forecast (median)")
    ax.fill_between(fut.month, fut.p10, fut.p90, color=COLORS["primary"], alpha=0.2, label="80% interval")
    ax.set_ylabel("Dining spend ($ / month)")
    ax.set_title(f"Example: dining forecast for student {sid}")
    ax.legend(frameon=False)
    save(fig, paths.figures / "fig_forecast_example.png")


def run_forecast(paths: Paths, monthly: pd.DataFrame, n_test: int = 6, seed: int = 42) -> Tuple[pd.DataFrame, dict]:
    panel = build_panel(monthly)
    n_months = int(panel.month_idx.max()) + 1
    feat = build_features(panel)
    res = walk_forward(feat, n_months, n_test, seed)
    metrics = evaluate(res)
    plot_forecasts(paths, res, metrics, feat)
    keep = ["student_id", "category", "month", "spend", "p50", "p10", "p90", "p10_raw", "p90_raw"]
    return res[keep].rename(columns={"spend": "actual"}).assign(month=lambda d: pd.to_datetime(d.month)), metrics
