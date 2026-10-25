"""Star schema for BI tools: small dimension tables joined to narrow fact tables on integer keys.

Power BI (or any BI tool) models data best as a star: one fact table per business event
(a transaction, a monthly forecast, an anomaly flag) and descriptive dimensions around it
(date, student, merchant, category). The tables live next to the analysis tables in
data/spend.db, and can be exported as CSVs for Power BI.

    python -m spendinsights.warehouse            # rebuild the star schema in data/spend.db
    python -m spendinsights.warehouse --export   # ... and write data/powerbi/<table>.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from . import db
from .config import FIXED_CATEGORIES, VARIABLE_CATEGORIES, Paths
from .eda import EXAM_MONTHS
from .nudges import OVERSPEND_FACTOR, _mid_month_table

DIMENSIONS = ["dim_date", "dim_student", "dim_merchant", "dim_category"]
FACTS = ["fact_transactions", "fact_forecast", "fact_flags"]
TABLES = DIMENSIONS + FACTS

# Which dimension each fact column points at, as (fact column, dimension, dimension key).
FOREIGN_KEYS = {
    "fact_transactions": [("date_key", "dim_date", "date_key"), ("student_key", "dim_student", "student_key"),
                          ("merchant_key", "dim_merchant", "merchant_key"),
                          ("category_key", "dim_category", "category_key")],
    "fact_forecast": [("month_key", "dim_date", "date_key"), ("student_key", "dim_student", "student_key"),
                      ("category_key", "dim_category", "category_key")],
    "fact_flags": [("date_key", "dim_date", "date_key"), ("student_key", "dim_student", "student_key"),
                   ("merchant_key", "dim_merchant", "merchant_key"),
                   ("category_key", "dim_category", "category_key")],
}


def _date_key(s: pd.Series) -> pd.Series:
    """Integer YYYYMMDD key, the usual date-dimension key in BI models."""
    d = pd.to_datetime(s)
    return (d.dt.year * 10000 + d.dt.month * 100 + d.dt.day).astype(int)


def _category_group(category: str) -> str:
    if category in FIXED_CATEGORIES:
        return "fixed"
    if category in VARIABLE_CATEGORIES:
        return "controllable"
    if category == "income":
        return "income"
    return "other"


def _dim_date(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One row per calendar day, so BI time intelligence (month-over-month etc.) works."""
    d = pd.Series(pd.date_range(start.normalize(), end.normalize(), freq="D"))
    return pd.DataFrame({
        "date_key": _date_key(d),
        "date": d.dt.strftime("%Y-%m-%d"),
        "year": d.dt.year,
        "quarter": d.dt.quarter,
        "month": d.dt.month,
        "month_name": d.dt.strftime("%b"),
        "month_start": d.dt.to_period("M").dt.start_time.dt.strftime("%Y-%m-%d"),
        # academic year runs September to August, e.g. "2023-24"
        "academic_year": [f"{y}-{str(y + 1)[2:]}" if m >= 9 else f"{y - 1}-{str(y)[2:]}"
                          for y, m in zip(d.dt.year, d.dt.month)],
        "day_of_month": d.dt.day,
        "day_of_week": d.dt.strftime("%a"),
        "is_weekend": (d.dt.dayofweek >= 5).astype(int),
        "is_exam_month": d.dt.month.isin(EXAM_MONTHS).astype(int),
    })


def build_tables(txns: pd.DataFrame, students: pd.DataFrame, forecasts: pd.DataFrame,
                 flags: pd.DataFrame, tau: float) -> Dict[str, pd.DataFrame]:
    """Build every dimension and fact table from the cleaned, categorised data.

    `txns` needs datetime `date` and `month` columns and `forecasts` a datetime `month`;
    `tau` is the tuned day-15 alert threshold from the nudges step.
    """
    dim_date = _dim_date(txns.date.min(), txns.date.max())

    dim_student = students.sort_values("student_id").reset_index(drop=True)
    dim_student.insert(0, "student_key", range(1, len(dim_student) + 1))

    merchants = sorted(txns.description_norm.unique())
    dim_merchant = pd.DataFrame({"merchant_key": range(1, len(merchants) + 1), "merchant_name": merchants})

    categories = sorted(set(txns.category) | set(forecasts.category))
    dim_category = pd.DataFrame({"category_key": range(1, len(categories) + 1), "category": categories})
    dim_category["category_group"] = dim_category.category.map(_category_group)
    dim_category["is_controllable"] = (dim_category.category_group == "controllable").astype(int)

    s_key = dict(zip(dim_student.student_id, dim_student.student_key))
    m_key = dict(zip(dim_merchant.merchant_name, dim_merchant.merchant_key))
    c_key = dict(zip(dim_category.category, dim_category.category_key))

    fact_transactions = pd.DataFrame({
        "txn_id": txns.txn_id,
        "date_key": _date_key(txns.date),
        "student_key": txns.student_id.map(s_key),
        "merchant_key": txns.description_norm.map(m_key),
        "category_key": txns.category.map(c_key),
        "direction": txns.direction,
        "amount": txns.amount,
        "spend": txns.spend,
        "confidence": txns.confidence,
        "category_source": txns.category_source,
    })

    # Forecast facts carry the budget-alert outcome too: budget = forecast median, alert when the
    # day-15 projection exceeds tau x budget, "material overspend" when actual > 1.15 x budget.
    alerts = _mid_month_table(txns, forecasts)[["student_id", "category", "month", "mtd", "projected", "ratio"]]
    f = forecasts.merge(alerts, on=["student_id", "category", "month"], how="left")
    eligible = f.ratio.notna()  # the nudges step ignores near-zero budgets
    fact_forecast = pd.DataFrame({
        "student_key": f.student_id.map(s_key),
        "category_key": f.category.map(c_key),
        "month_key": _date_key(f.month),
        "actual": f.actual,
        "forecast": f.p50,
        "lower_80": f.p10,
        "upper_80": f.p90,
        "abs_error": (f.actual - f.p50).abs(),
        "in_interval": ((f.actual >= f.p10) & (f.actual <= f.p90)).astype(int),
        "budget_eligible": eligible.astype(int),
        "is_material_overspend": (f.actual > OVERSPEND_FACTOR * f.p50).astype("Int64").where(eligible, pd.NA),
        "spend_by_day15": f.mtd,
        "projected_month_end": f.projected,
        "alert_day15": (f.ratio > tau).astype("Int64").where(eligible, pd.NA),
    })

    fact_flags = pd.DataFrame({
        "flag_id": range(1, len(flags) + 1),
        "txn_id": flags.txn_id,
        "date_key": _date_key(flags.date),
        "student_key": flags.student_id.map(s_key),
        "merchant_key": flags.description_norm.map(m_key),
        "category_key": flags.category.map(c_key),
        "spend": flags.spend,
        "reason": flags.reason,
        "isolation_forest_score": flags.if_score,
    })

    return {"dim_date": dim_date, "dim_student": dim_student, "dim_merchant": dim_merchant,
            "dim_category": dim_category, "fact_transactions": fact_transactions,
            "fact_forecast": fact_forecast, "fact_flags": fact_flags}


def _tau_from_metrics(paths: Paths) -> float:
    metrics = json.loads((paths.reports / "metrics.json").read_text(encoding="utf-8"))
    return float(metrics["nudges"]["tau_tuned_on_first_half"])


def build_warehouse(paths: Paths, tau: float | None = None) -> Dict[str, int]:
    """(Re)build the star schema from the analysis tables already in data/spend.db."""
    with db.connect(paths) as con:
        txns = pd.read_sql_query("SELECT * FROM transactions", con, parse_dates=["date", "month"])
        students = pd.read_sql_query("SELECT * FROM students", con)
        forecasts = pd.read_sql_query("SELECT * FROM forecasts", con, parse_dates=["month"])
        flags = pd.read_sql_query("SELECT * FROM flags", con)
    tables = build_tables(txns, students, forecasts, flags, tau if tau is not None else _tau_from_metrics(paths))

    with db.connect(paths) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False, if_exists="replace")
        for fact, keys in FOREIGN_KEYS.items():
            for col, _, _ in keys:
                con.execute(f"CREATE INDEX IF NOT EXISTS idx_{fact}_{col} ON {fact}({col})")
    return {name: int(len(df)) for name, df in tables.items()}


def export_csv(paths: Paths) -> List[Path]:
    """Write every star-schema table to data/powerbi/<table>.csv for Power BI's Get Data > Text/CSV."""
    out_dir = paths.data / "powerbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with db.connect(paths) as con:
        for name in TABLES:
            out = out_dir / f"{name}.csv"
            pd.read_sql_query(f"SELECT * FROM {name}", con).to_csv(out, index=False)
            written.append(out)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="also write data/powerbi/*.csv")
    args = ap.parse_args()
    paths = Paths()
    counts = build_warehouse(paths)
    for name, n in counts.items():
        print(f"{name:<18} {n:>8,} rows")
    if args.export:
        for p in export_csv(paths):
            print(f"wrote {p.relative_to(paths.root)}")


if __name__ == "__main__":
    main()
