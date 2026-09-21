"""Budget nudges: when should a student be warned, and how would we measure whether warnings work?

Part 1 (real data, retrospective): budget = forecast median. Halfway through the month (day 15)
project month-end spend from the pace so far; nudge if the projection exceeds tau x budget.
tau is tuned on the first half of the test months and scored on the second half.

Part 2 (SIMULATION): a randomised experiment on the flagged cases. The treatment effect is an
ASSUMPTION I set (nudges cut remaining-month spend by ~10%); it is not evidence that nudges work.
What the simulation demonstrates is the *measurement method*: randomisation, a confidence
interval, variance reduction with CUPED, and how many students the test would need.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import VARIABLE_CATEGORIES, Paths
from .viz import COLORS, plt, save, style

TAU_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]
OVERSPEND_FACTOR = 1.15  # "material overspend" = actual month-end spend > 115% of budget
ASSUMED_EFFECT = 0.10    # ASSUMPTION for the simulation


def _mid_month_table(txns: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    d = txns[(txns.direction == "debit") & txns.category.isin(VARIABLE_CATEGORIES)]
    mtd = (d[d.date.dt.day <= 15].groupby(["student_id", "category", "month"]).spend.sum()
           .rename("mtd").reset_index())
    f = forecasts.merge(mtd, on=["student_id", "category", "month"], how="left")
    f["mtd"] = f["mtd"].fillna(0.0)
    f["budget"] = f["p50"]
    f = f[f.budget >= 5].copy()  # ignore near-zero budgets (ratio is meaningless)
    f["projected"] = f.mtd * f.month.dt.days_in_month / 15.0
    f["ratio"] = f.projected / f.budget
    f["overspend"] = f.actual > OVERSPEND_FACTOR * f.budget
    f["remaining"] = (f.actual - f.mtd).clip(lower=0)
    return f


def _pr(f: pd.DataFrame, tau: float) -> dict:
    flag = f.ratio > tau
    tp = int((flag & f.overspend).sum())
    fp = int((flag & ~f.overspend).sum())
    fn = int((~flag & f.overspend).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tau": tau, "flagged": int(flag.sum()), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def _cuped(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float]:
    """Variance reduction using a pre-treatment covariate x (here: the forecast budget)."""
    theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), float(theta)


def _welch_p(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def _simulate(f: pd.DataFrame, rng: np.random.Generator, n_per_arm: int = None) -> dict:
    """One simulated randomised experiment on rows of f (each row = one flagged student-category-month)."""
    g = f.sample(n=2 * n_per_arm, random_state=int(rng.integers(1 << 31))) if n_per_arm else f
    treated = rng.permutation(np.arange(len(g)) % 2).astype(bool)
    delta = np.clip(rng.normal(ASSUMED_EFFECT, 0.05, len(g)), 0, 0.3)
    y = g.mtd.to_numpy() + g.remaining.to_numpy() * (1 - delta * treated)
    x = g.budget.to_numpy()
    y_adj, theta = _cuped(y, x)
    return {
        "y": y, "y_adj": y_adj, "treated": treated, "theta": theta,
        "p_raw": _welch_p(y[treated], y[~treated]),
        "p_cuped": _welch_p(y_adj[treated], y_adj[~treated]),
    }


def run_nudges(paths: Paths, txns: pd.DataFrame, forecasts: pd.DataFrame, seed: int = 42) -> dict:
    style()
    rng = np.random.default_rng(seed)
    f = _mid_month_table(txns, forecasts)
    months = sorted(f.month.unique())
    split = months[len(months) // 2]
    tune, evalset = f[f.month < split], f[f.month >= split]

    grid_tune = [_pr(tune, t) for t in TAU_GRID]
    best = max(grid_tune, key=lambda r: r["f1"])
    tau = best["tau"]
    grid_eval = [_pr(evalset, t) for t in TAU_GRID]
    at_tau = next(r for r in grid_eval if r["tau"] == tau)
    metrics = {
        "rows": int(len(f)),
        "overspend_definition": f"actual > {OVERSPEND_FACTOR:.2f} x forecast-median budget",
        "base_rate_overspend_eval": float(evalset.overspend.mean()),
        "tau_tuned_on_first_half": tau,
        "tuning_f1": best["f1"],
        "eval_at_tau": at_tau,
        "eval_grid": grid_eval,
    }

    # ---- simulated A/B test on all cases flagged at tau ----
    flagged = f[f.ratio > tau].reset_index(drop=True)
    sim = _simulate(flagged, rng)
    y, y_adj, t = sim["y"], sim["y_adj"], sim["treated"]
    diff = float(y[~t].mean() - y[t].mean())
    control, treat = y[~t], y[t]
    boots = []
    for _ in range(1000):  # bootstrap the difference in means
        c = control[rng.integers(0, len(control), len(control))]
        tr = treat[rng.integers(0, len(treat), len(treat))]
        boots.append(c.mean() - tr.mean())
    var_reduction = 1 - float(np.var(y_adj, ddof=1) / np.var(y, ddof=1))
    sd = float(y.std(ddof=1))
    delta_abs = ASSUMED_EFFECT * float(flagged.remaining.mean())
    n_needed_raw = int(np.ceil(2 * ((1.96 + 0.84) * sd / delta_abs) ** 2))
    n_needed_cuped = int(np.ceil(2 * ((1.96 + 0.84) * float(y_adj.std(ddof=1)) / delta_abs) ** 2))

    # empirical power by sample size
    sizes = [n for n in (25, 50, 100, 200, 400, 800) if 2 * n <= len(flagged)]
    power = {"n_per_arm": sizes, "raw": [], "cuped": []}
    for n in sizes:
        sims = [_simulate(flagged, rng, n) for _ in range(300)]
        power["raw"].append(float(np.mean([s["p_raw"] < 0.05 for s in sims])))
        power["cuped"].append(float(np.mean([s["p_cuped"] < 0.05 for s in sims])))

    metrics["simulated_experiment"] = {
        "IMPORTANT": "effect size is an assumption, not a finding",
        "assumed_relative_effect_on_remaining_spend": ASSUMED_EFFECT,
        "n_flagged_cases": int(len(flagged)),
        "mean_outcome_control": float(y[~t].mean()),
        "mean_outcome_treated": float(y[t].mean()),
        "diff_in_means": diff,
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "p_value_raw": sim["p_raw"],
        "p_value_cuped": sim["p_cuped"],
        "cuped_variance_reduction": var_reduction,
        "n_per_arm_needed_80pct_power_raw": n_needed_raw,
        "n_per_arm_needed_80pct_power_cuped": n_needed_cuped,
        "power_by_sample_size": power,
    }

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["recall"] * 100 for r in grid_eval], [r["precision"] * 100 for r in grid_eval],
            marker="o", color=COLORS["primary"])
    for r in grid_eval:
        ax.annotate(f"{r['tau']:.1f}", (r["recall"] * 100, r["precision"] * 100), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    ax.axhline(100 * metrics["base_rate_overspend_eval"], color=COLORS["grey"], linestyle="--")
    ax.text(ax.get_xlim()[0] + 1, 100 * metrics["base_rate_overspend_eval"] + 1,
            "precision if we nudged everyone", color=COLORS["grey"], fontsize=8)
    ax.set_xlabel("Recall of material overspend (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Nudge trigger: pace-based projection vs. budget (labels = tau)")
    save(fig, paths.figures / "fig_nudge_precision_recall.png")

    if sizes:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(sizes, [100 * p for p in power["raw"]], marker="o", label="raw difference in means", color=COLORS["grey"])
        ax.plot(sizes, [100 * p for p in power["cuped"]], marker="o", label="CUPED-adjusted", color=COLORS["primary"])
        ax.axhline(80, color=COLORS["accent"], linestyle="--")
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.minorticks_off()
        ax.set_xlabel("Cases per arm")
        ax.set_ylabel("Power (%) for assumed 10% effect")
        ax.set_title("Simulated A/B test: how many cases are enough?")
        ax.legend(frameon=False)
        save(fig, paths.figures / "fig_ab_power.png")
    return metrics
