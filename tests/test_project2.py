from quant_showcase.project2.pipeline import run


def test_project2_variance_reduction_positive() -> None:
    report = run(seed=123)
    assert report.name == "project2_asian_options"
    assert report.metrics["control_variate_se"] < report.metrics["plain_se"]
    assert report.metrics["variance_reduction"] > 0
