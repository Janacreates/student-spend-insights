"""Exploratory analysis: where the money goes, who varies most, and one tested hypothesis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import db
from .config import FIXED_CATEGORIES, VARIABLE_CATEGORIES, Paths
from .viz import COLORS, plt, save, style

EXAM_MONTHS = (4, 12)
OTHER_TERM_MONTHS = (9, 10, 11, 1, 2, 3)


def _bootstrap_ci(diff: np.ndarray, n: int = 2000, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run_eda(paths: Paths) -> dict:
    style()
    totals = db.query(paths, "category_totals")
    monthly = db.query(paths, "monthly_category_spend")
    monthly["month"] = pd.to_datetime(monthly["month"])
    metrics: dict = {}

    # ---- 1. where the money goes ----
    share = totals.set_index("category")["pct_of_spend"]
    metrics["pct_of_spend_by_category"] = {k: float(v) for k, v in share.items()}
    fixed_pct = float(share[share.index.isin(FIXED_CATEGORIES)].sum())
    var_totals = totals[totals.category.isin(VARIABLE_CATEGORIES)].set_index("category").total_spend
    var_share = (100 * var_totals / var_totals.sum()).sort_values(ascending=False)
    metrics["fixed_pct_of_all_spend"] = fixed_pct
    metrics["variable_share_pct"] = {k: float(v) for k, v in var_share.items()}
    metrics["top2_variable_categories"] = list(var_share.index[:2])
    metrics["top2_variable_share_pct"] = float(var_share.iloc[:2].sum())

    fig, ax = plt.subplots(figsize=(7, 4.2))
    order = totals.sort_values("pct_of_spend")
    colors = [COLORS["grey"] if c in FIXED_CATEGORIES or c in ("income", "uncategorized") else COLORS["primary"]
              for c in order.category]
    ax.barh(order.category, order.pct_of_spend, color=colors)
    for y, v in enumerate(order.pct_of_spend):
        ax.text(v + 0.3, y, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% of total debit spend")
    ax.set_title("Where the money goes (grey = fixed / non-controllable)")
    ax.grid(axis="y", visible=False)
    save(fig, paths.figures / "fig_spend_by_category.png")

    # ---- 2. monthly variable spend by category ----
    var = monthly[monthly.category.isin(VARIABLE_CATEGORIES)]
    pivot = var.pivot_table(index="month", columns="category", values="spend", aggfunc="sum").fillna(0)
    per_student = pivot / monthly.student_id.nunique()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    palette = [COLORS["primary"], COLORS["accent"], COLORS["green"], COLORS["red"], COLORS["grey"]]
    for col, color in zip(VARIABLE_CATEGORIES, palette):
        ax.plot(per_student.index, per_student[col], label=col, color=color, linewidth=1.8)
    ax.set_ylabel("Average $ per student per month")
    ax.set_title("Monthly variable spending by category")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    save(fig, paths.figures / "fig_monthly_variable_spend.png")

    # ---- 3. spread across students ----
    per_stu = var.groupby(["student_id", "month"]).spend.sum().groupby("student_id").mean()
    q = per_stu.quantile([0.1, 0.5, 0.9])
    metrics["student_monthly_variable_spend"] = {"p10": float(q[0.1]), "median": float(q[0.5]), "p90": float(q[0.9])}
    top20 = per_stu.sort_values(ascending=False)
    metrics["top20pct_students_share_of_variable_spend_pct"] = float(
        100 * top20.iloc[: int(0.2 * len(top20))].sum() / top20.sum())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(per_stu, bins=25, color=COLORS["primary"], alpha=0.85)
    for lbl, val, c in (("median", q[0.5], COLORS["accent"]), ("p90", q[0.9], COLORS["red"])):
        ax.axvline(val, color=c, linestyle="--")
        ax.text(val, ax.get_ylim()[1] * 0.92, f" {lbl} ${val:,.0f}", color=c, fontsize=9)
    ax.set_xlabel("Average monthly variable spend per student ($)")
    ax.set_ylabel("Students")
    ax.set_title("Variable spending varies widely between students")
    save(fig, paths.figures / "fig_student_distribution.png")

    # ---- 4. tested hypothesis: dining spend is higher in exam months ----
    dining = monthly[monthly.category == "dining"].copy()
    dining["moy"] = dining.month.dt.month
    grid = dining.pivot_table(index="student_id", columns="moy", values="spend", aggfunc="mean")
    exam = grid[[m for m in EXAM_MONTHS if m in grid.columns]].mean(axis=1)
    other = grid[[m for m in OTHER_TERM_MONTHS if m in grid.columns]].mean(axis=1)
    ok = exam.notna() & other.notna()
    exam, other = exam[ok], other[ok]
    diff = (exam - other).to_numpy()
    w = stats.wilcoxon(diff)
    lo, hi = _bootstrap_ci(diff)
    metrics["dining_exam_effect"] = {
        "n_students": int(ok.sum()),
        "mean_exam": float(exam.mean()),
        "mean_other_term": float(other.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "relative_lift_pct": float(100 * diff.mean() / other.mean()),
        "wilcoxon_p": float(w.pvalue),
        "test": "paired Wilcoxon signed-rank on per-student means (Apr/Dec vs Sep-Nov, Jan-Mar)",
    }
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bp = ax.boxplot([other, exam], tick_labels=["Other term months", "Exam months (Apr, Dec)"],
                    patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], (COLORS["light"], COLORS["accent"])):
        patch.set_facecolor(c)
    ax.set_ylabel("Avg monthly dining spend per student ($)")
    ax.set_title(f"Dining +${diff.mean():.0f}/month in exam months (95% CI {lo:.0f} to {hi:.0f})")
    ax.grid(axis="x", visible=False)
    save(fig, paths.figures / "fig_exam_dining.png")

    metrics["top_merchants"] = db.query(paths, "top_merchants").head(5).to_dict(orient="records")

    # month-over-month swings (SQL window function LAG) for the controllable categories
    mom = db.query(paths, "month_over_month").dropna(subset=["pct_change_vs_prev_month"])
    mom = mom[mom.category.isin(VARIABLE_CATEGORIES)]
    big = mom.loc[mom.pct_change_vs_prev_month.idxmax()]
    metrics["largest_month_over_month_increase"] = {
        "category": str(big.category), "month": str(big.month), "pct": float(big.pct_change_vs_prev_month)}
    return metrics
