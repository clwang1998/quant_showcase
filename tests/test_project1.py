import numpy as np

from quant_showcase.project1.pipeline import run


def test_project1_pipeline_runs() -> None:
    report = run(seed=123)
    assert report.name == "project1_gat_multifactor"
    assert -1.0 <= report.metrics["ic"] <= 1.0
    assert report.metrics["long_count"] > 0
    assert report.metrics["short_count"] > 0
    assert np.isfinite(report.metrics["annual_volatility"])
