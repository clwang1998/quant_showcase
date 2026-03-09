from __future__ import annotations

import argparse
import json

from quant_showcase.project1.pipeline import run as run_project1
from quant_showcase.project2.pipeline import run as run_project2
from quant_showcase.project3.pipeline import run as run_project3


def _report_to_json(report) -> str:
    payload = {
        "name": report.name,
        "metrics": {k: float(v) for k, v in report.metrics.items()},
    }
    return json.dumps(payload, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantitative engineering showcase")
    parser.add_argument("pipeline", choices=["project1", "project2", "project3", "all"])
    args = parser.parse_args()

    if args.pipeline == "project1":
        print(_report_to_json(run_project1()))
    elif args.pipeline == "project2":
        print(_report_to_json(run_project2()))
    elif args.pipeline == "project3":
        print(_report_to_json(run_project3()))
    else:
        for fn in (run_project1, run_project2, run_project3):
            print(_report_to_json(fn()))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
