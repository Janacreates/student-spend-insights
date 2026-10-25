import sqlite3

import pandas as pd
import pytest

from spendinsights import db
from spendinsights.config import Paths
from spendinsights.pipeline import run
from spendinsights.warehouse import DIMENSIONS, FOREIGN_KEYS, TABLES, export_csv

BI_QUERIES = ["spend_by_segment", "fixed_vs_controllable", "merchant_leaderboard",
              "forecast_accuracy_by_category", "exam_month_effect", "flags_by_reason"]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    paths = Paths(root=tmp_path_factory.mktemp("wh"))
    metrics = run(paths, n_students=40, n_months=16, seed=11, n_test_months=3, make_deck=False, verbose=False)
    con = sqlite3.connect(paths.db)
    yield paths, metrics, con
    con.close()


def _scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def test_dimension_keys_are_unique(built):
    _, _, con = built
    for dim in DIMENSIONS:
        key = dim.replace("dim_", "") + "_key"
        n, n_distinct = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT {key}) FROM {dim}").fetchone()
        assert n > 0 and n == n_distinct, dim


def test_every_fact_foreign_key_resolves(built):
    _, _, con = built
    for fact, keys in FOREIGN_KEYS.items():
        for col, dim, dim_key in keys:
            orphans = _scalar(con, f"SELECT COUNT(*) FROM {fact} AS f LEFT JOIN {dim} AS d "
                                   f"ON d.{dim_key} = f.{col} WHERE d.{dim_key} IS NULL")
            assert orphans == 0, f"{fact}.{col} has {orphans} keys missing from {dim}"


def test_transaction_fact_reconciles_with_source(built):
    _, _, con = built
    n_src, spend_src, amount_src = con.execute("SELECT COUNT(*), SUM(spend), SUM(amount) FROM transactions").fetchone()
    n_fact, n_ids, spend_fact, amount_fact = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT txn_id), SUM(spend), SUM(amount) FROM fact_transactions").fetchone()
    assert n_fact == n_ids == n_src
    assert spend_fact == pytest.approx(spend_src)
    assert amount_fact == pytest.approx(amount_src)


def test_date_dimension_is_contiguous_and_covers_every_transaction(built):
    _, _, con = built
    dates = pd.to_datetime(pd.read_sql_query("SELECT date FROM dim_date ORDER BY date_key", con).date)
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()
    lo, hi = con.execute("SELECT MIN(date), MAX(date) FROM transactions").fetchone()
    assert dates.min() == pd.Timestamp(lo) and dates.max() == pd.Timestamp(hi)


def test_forecast_fact_reproduces_pipeline_metrics(built):
    _, m, con = built
    mae, coverage = con.execute("SELECT AVG(abs_error), AVG(in_interval) FROM fact_forecast").fetchone()
    assert mae == pytest.approx(m["forecast"]["mae"]["gradient_boosting"])
    assert coverage == pytest.approx(m["forecast"]["interval_coverage_conformal"])

    # The nudges step tunes the alert on the first half of the test months and scores the second half.
    months = [r[0] for r in con.execute(
        "SELECT DISTINCT month_key FROM fact_forecast WHERE budget_eligible = 1 ORDER BY month_key")]
    split = months[len(months) // 2]
    tp, alerts, overspends = con.execute(
        "SELECT SUM(alert_day15 * is_material_overspend), SUM(alert_day15), SUM(is_material_overspend) "
        f"FROM fact_forecast WHERE budget_eligible = 1 AND month_key >= {split}").fetchone()
    expected = m["nudges"]["eval_at_tau"]
    assert (tp / alerts if alerts else 0.0) == pytest.approx(expected["precision"])
    assert (tp / overspends if overspends else 0.0) == pytest.approx(expected["recall"])


def test_bi_queries_run_and_account_for_all_spend(built):
    paths, _, con = built
    for name in BI_QUERIES:
        assert len(db.query(paths, f"bi/{name}")) > 0, name

    split = db.query(paths, "bi/fixed_vs_controllable")
    debit_total = _scalar(con, "SELECT SUM(spend) FROM transactions WHERE direction = 'debit'")
    assert split.spend.sum() == pytest.approx(debit_total, abs=1.0)  # per-row rounding to cents
    assert split.groupby("month").pct_of_month.sum().between(99.5, 100.5).all()


def test_export_writes_one_csv_per_table(built):
    paths, _, con = built
    written = export_csv(paths)
    assert sorted(p.stem for p in written) == sorted(TABLES)
    fact = pd.read_csv(paths.data / "powerbi" / "fact_transactions.csv")
    assert len(fact) == _scalar(con, "SELECT COUNT(*) FROM fact_transactions")
