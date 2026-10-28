"""End-to-end run: generate -> clean -> categorise -> SQL/EDA -> forecast -> anomalies -> nudges -> survival -> star schema -> deck.

    python -m spendinsights.pipeline
    python -m spendinsights.pipeline --students 40 --months 16 --test-months 3   # quick run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import db
from .anomalies import run_anomalies
from .categorize import run_categorization
from .clean import clean, write_quality_report
from .config import SEED, Paths
from .deck import build_deck
from .eda import run_eda
from .forecast import run_forecast
from .generate import write_dataset
from .nudges import run_nudges
from .survival import run_survival
from .warehouse import build_warehouse


def _step(name: str, verbose: bool):
    if verbose:
        print(f"[{time.strftime('%H:%M:%S')}] {name}", flush=True)


def run(paths: Paths = Paths(), n_students: int = 150, n_months: int = 24, seed: int = SEED,
        n_test_months: int = 6, make_deck: bool = True, verbose: bool = True) -> dict:
    _step("generate synthetic data", verbose)
    write_dataset(paths, n_students=n_students, n_months=n_months, seed=seed)

    _step("load + clean raw export", verbose)
    raw = pd.read_csv(paths.raw_csv, dtype=str, keep_default_na=False, na_values=[""])
    truth = pd.read_csv(paths.truth_csv)
    students = pd.read_csv(paths.students_csv)
    txns, profile = clean(raw)
    write_quality_report(profile, paths.reports / "data_quality_report.md")

    _step("categorise transactions", verbose)
    preds, cat_metrics = run_categorization(txns, truth, seed=seed)
    txns = txns.merge(preds, on="txn_id", how="left")

    _step("build SQLite database + EDA", verbose)
    db.build_db(paths, txns, students)
    eda_metrics = run_eda(paths)

    _step("forecast next-month spend", verbose)
    monthly = db.query(paths, "monthly_category_spend")
    forecasts, fc_metrics = run_forecast(paths, monthly, n_test=n_test_months, seed=seed)

    _step("detect anomalies", verbose)
    flags, an_metrics = run_anomalies(paths, txns, truth, seed=seed)

    _step("budget nudges + simulated A/B test", verbose)
    nudge_metrics = run_nudges(paths, txns, forecasts, seed=seed)

    _step("survival analysis: when budgets get crossed", verbose)
    survival_metrics = run_survival(paths, txns, forecasts, students)

    db.save_table(paths, "forecasts", forecasts)
    db.save_table(paths, "flags", flags)
    recurring = db.query(paths, "recurring_charges")
    db.save_table(paths, "recurring_charges", recurring)

    _step("build star schema for BI", verbose)
    build_warehouse(paths, tau=nudge_metrics["tau_tuned_on_first_half"])

    metrics = {
        "config": {"students": n_students, "months": n_months, "seed": seed, "test_months": n_test_months},
        "data": {"students": n_students, "months": n_months, "raw_rows": profile["rows_raw"],
                 "clean_rows": profile["rows_clean"]},
        "data_quality": profile,
        "categorization": cat_metrics,
        "eda": eda_metrics,
        "forecast": fc_metrics,
        "anomalies": an_metrics,
        "nudges": nudge_metrics,
        "survival": survival_metrics,
        "recurring_subscriptions_found": int(len(recurring)),
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    if make_deck:
        _step("build findings deck", verbose)
        build_deck(paths, metrics)
    _step("done", verbose)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--students", type=int, default=150)
    ap.add_argument("--months", type=int, default=24)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-deck", action="store_true")
    args = ap.parse_args()
    run(n_students=args.students, n_months=args.months, seed=args.seed,
        n_test_months=args.test_months, make_deck=not args.no_deck)


if __name__ == "__main__":
    main()
