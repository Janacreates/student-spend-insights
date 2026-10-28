"""Time-to-event (survival) analysis: on which day does a student's spending first cross its budget?

Each student x category x test month is one unit, with the forecast median as the budget (as in
nudges.py). The event is the first day cumulative spend exceeds the budget. Months that never cross
are right-censored at month end, which is why this needs survival methods: averaging crossing days
over only the months that crossed would ignore the half that never did.

Kaplan-Meier curves show how quickly budgets are used up, a Cox model tests what makes a crossing
come sooner, and the crossing days answer a timing question for the budget alert: a warning only
helps if it arrives while the student is still under budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index

from .config import VARIABLE_CATEGORIES, Paths
from .nudges import OVERSPEND_FACTOR, TAU_GRID, _pr
from .viz import COLORS, plt, save, style

KEYS = ["student_id", "category", "month"]
ALERT_DAYS = [5, 7, 10, 12, 15, 18, 20]
KM_DAYS = [10, 15, 20]


def _debits(txns: pd.DataFrame) -> pd.DataFrame:
    d = txns[(txns.direction == "debit") & txns.category.isin(VARIABLE_CATEGORIES)]
    return d.assign(day=d.date.dt.day)


def crossing_table(txns: pd.DataFrame, forecasts: pd.DataFrame, students: pd.DataFrame) -> pd.DataFrame:
    """One row per student x category x month: budget, first crossing day (NaN if never) and censoring."""
    f = forecasts.assign(budget=forecasts.p50)
    f = f[f.budget >= 5].copy()  # same rule as nudges.py: near-zero budgets make the ratio meaningless
    f["overspend"] = f.actual > OVERSPEND_FACTOR * f.budget
    daily = _debits(txns).groupby(KEYS + ["day"]).spend.sum().reset_index().sort_values("day")
    daily["cum"] = daily.groupby(KEYS).spend.cumsum()
    daily = daily.merge(f[KEYS + ["budget"]], on=KEYS)
    cross = daily[daily.cum > daily.budget].groupby(KEYS).day.min().rename("cross_day").reset_index()
    f = f.merge(cross, on=KEYS, how="left").merge(students[["student_id", "housing"]], on="student_id", how="left")
    f["event"] = f.cross_day.notna().astype(int)
    f["duration"] = f.cross_day.fillna(f.month.dt.days_in_month).astype(float)  # censored at month end
    return f.reset_index(drop=True)


def warned_before_crossing(flagged: pd.Series, overspend: pd.Series, cross_day: pd.Series, day: int) -> pd.Series:
    """True where an alert on `day` caught a material overspend while the student was still under budget."""
    return flagged & overspend & ~(cross_day <= day)


def _km(t: pd.DataFrame) -> dict:
    km = KaplanMeierFitter().fit(t.duration, t.event)
    out = {"n": int(len(t)), "crossed_by_month_end": float(t.event.mean())}
    for d in KM_DAYS:
        out[f"crossed_by_day_{d}"] = float(1 - km.survival_function_at_times(d).iloc[0])
    return out


def _design(t: pd.DataFrame, strata: bool = False) -> pd.DataFrame:
    cols = ["housing"] if strata else ["category", "housing"]
    X = pd.get_dummies(t[cols], drop_first=True).astype(float)  # references: dining, family
    X["log_budget"] = np.log(t.budget)
    if strata:
        X["category"] = t.category
    return X.assign(duration=t.duration, event=t.event)


def _hazard_ratios(cph: CoxPHFitter) -> dict:
    s = cph.summary
    return {c: {"hr": float(s.loc[c, "exp(coef)"]),
                "ci95": [float(s.loc[c, "exp(coef) lower 95%"]), float(s.loc[c, "exp(coef) upper 95%"])],
                "p": float(s.loc[c, "p"])} for c in s.index}


def cox_model(t: pd.DataFrame) -> dict:
    """Cox model on every covariate, its proportional-hazards check, and a model stratified by category."""
    months = sorted(t.month.unique())
    split = months[len(months) // 2]
    X = _design(t)
    cph = CoxPHFitter().fit(X, "duration", "event")
    ph = proportional_hazard_test(cph, X, time_transform="rank").summary
    train, test = X[t.month < split], X[t.month >= split]
    held = CoxPHFitter().fit(train, "duration", "event")
    # higher partial hazard = crossing sooner, so negate it for concordance with durations
    c_held = concordance_index(test.duration, -held.predict_partial_hazard(test), test.event)
    strat = CoxPHFitter().fit(_design(t, strata=True), "duration", "event", strata=["category"])
    return {
        "hazard_ratios": _hazard_ratios(cph),
        "concordance_train": float(cph.concordance_index_),
        "concordance_heldout": float(c_held),
        "proportional_hazards_test_p": {c: float(ph.loc[c, "p"]) for c in ph.index},
        "stratified_by_category": {"hazard_ratios": _hazard_ratios(strat),
                                   "concordance": float(strat.concordance_index_)},
    }


def alert_timing(txns: pd.DataFrame, t: pd.DataFrame, days=ALERT_DAYS) -> list:
    """Re-run the pace-based alert on each day: tau tuned on the first half of test months, scored on the second."""
    months = sorted(t.month.unique())
    split = months[len(months) // 2]
    d = _debits(txns)
    rows = []
    for day in days:
        mtd = d[d.day <= day].groupby(KEYS).spend.sum().rename("mtd").reset_index()
        g = t.merge(mtd, on=KEYS, how="left").fillna({"mtd": 0.0})
        g["ratio"] = g.mtd * g.month.dt.days_in_month / day / g.budget
        tune, ev = g[g.month < split], g[g.month >= split]
        tau = max((_pr(tune, x) for x in TAU_GRID), key=lambda r: r["f1"])["tau"]
        r = _pr(ev, tau)
        in_time = warned_before_crossing(ev.ratio > tau, ev.overspend, ev.cross_day, day)
        r["day"] = day
        r["warned_before_crossing"] = float(in_time.sum() / ev.overspend.sum())
        rows.append(r)
    return rows


def run_survival(paths: Paths, txns: pd.DataFrame, forecasts: pd.DataFrame, students: pd.DataFrame) -> dict:
    style()
    t = crossing_table(txns, forecasts, students)
    over = t[t.overspend]
    timing = alert_timing(txns, t)
    metrics = {
        "units": int(len(t)),
        "event_definition": "first day cumulative spend > forecast-median budget; censored at month end",
        "kaplan_meier": {"all": _km(t),
                         "by_category": {c: _km(g) for c, g in t.groupby("category")},
                         "by_housing": {h: _km(g) for h, g in t.groupby("housing")}},
        "material_overspends": {
            "n": int(len(over)),
            "median_crossing_day": float(over.cross_day.median()),
            **{f"already_over_budget_by_day_{d}": float((over.cross_day <= d).mean()) for d in (10, 15)},
        },
        "cox": cox_model(t),
        "alert_timing_heldout": timing,
    }

    # ---- figure: how fast budgets are crossed, and what that means for the alert day ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    palette = [COLORS[k] for k in ("primary", "accent", "green", "red", "grey")]
    for color, (cat, g) in zip(palette, t.groupby("category")):
        km = KaplanMeierFitter().fit(g.duration, g.event)
        sf = km.survival_function_.iloc[:, 0]
        a1.step(sf.index, 100 * (1 - sf.values), where="post", color=color, label=cat)
    a1.axvline(15, color=COLORS["grey"], linestyle="--")
    a1.text(15.4, 2, "day-15 alert", color=COLORS["grey"], fontsize=8)
    a1.set_xlim(1, 31)
    a1.set_xlabel("Day of month")
    a1.set_ylabel("Months already over budget (%)")
    a1.set_title("When budgets get crossed (Kaplan-Meier)")
    a1.legend(frameon=False, fontsize=8)
    xs = [r["day"] for r in timing]
    a2.plot(xs, [100 * r["precision"] for r in timing], marker="o", color=COLORS["primary"], label="precision")
    a2.plot(xs, [100 * r["recall"] for r in timing], marker="o", color=COLORS["grey"], label="recall")
    a2.plot(xs, [100 * r["warned_before_crossing"] for r in timing], marker="o", color=COLORS["accent"],
            label="overspends warned while still under budget")
    a2.set_xlabel("Alert day")
    a2.set_ylabel("%")
    a2.set_title("Alert day trade-off (held-out months)")
    a2.legend(frameon=False, fontsize=8)
    save(fig, paths.figures / "fig_budget_crossing.png")
    return metrics
