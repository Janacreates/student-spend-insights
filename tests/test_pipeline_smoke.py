from spendinsights.config import Paths
from spendinsights.pipeline import run


def test_pipeline_runs_end_to_end_on_small_data(tmp_path):
    paths = Paths(root=tmp_path)
    m = run(paths, n_students=40, n_months=16, seed=11, n_test_months=3, make_deck=False, verbose=False)

    for key in ("data_quality", "categorization", "eda", "forecast", "anomalies", "nudges"):
        assert key in m
    assert paths.db.exists()
    assert (paths.reports / "metrics.json").exists()
    assert (paths.figures / "fig_forecast_accuracy.png").exists()
    assert m["forecast"]["n_predictions"] > 0
    assert 0.0 <= m["forecast"]["interval_coverage_conformal"] <= 1.0
    assert m["anomalies"]["n_injected"] > 0
