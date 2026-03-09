from quant_showcase.project1.pipeline import run as run_p1
from quant_showcase.project2.pipeline import run as run_p2
from quant_showcase.project3.pipeline import run as run_p3


def test_reports_have_metrics() -> None:
    for report in (run_p1(7), run_p2(7), run_p3(7)):
        assert isinstance(report.metrics, dict)
        assert len(report.metrics) > 0
