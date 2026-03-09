from quant_showcase.project3.pipeline import run


def test_project3_pipeline_runs() -> None:
    report = run(seed=123)
    assert report.name == "project3_rl_execution"
    assert report.metrics["ppo_completion"] > 0.95
    assert report.metrics["dt_completion"] > 0.95
