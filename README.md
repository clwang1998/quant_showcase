# Quantitative Engineering Showcase 

This repository is a clean, from-scratch, production-style showcase of three quant research systems aligned with my full-time work at **Ubiquant Investment Co., Ltd (2022–present)**.

The code is intentionally organized like a real engineering project: modular package structure, deterministic pipelines, test coverage, explicit interfaces, and reproducible synthetic data.

## Why this repository exists

This project demonstrates:

- End-to-end system design across research domains
- Production-minded software engineering (clear boundaries, testability, reproducibility)
- Quantitative rigor (optimization, variance reduction, risk-aware evaluation)
- Clear mapping from resume claims to executable components

## Research Narrative Template

Each project follows the same structure to make technical storytelling explicit:

1. `Problem` (what breaks in conventional practice)
2. `Hypothesis` (what mechanism should improve outcomes)
3. `Method Stack` (ordered methods, from baseline to advanced)
4. `API Contract` (`Config -> run(config) -> PipelineReport`)
5. `Evaluation` (metrics linked to the claim)

## Repository Structure

```text
quant_projects/
├── src/quant_showcase/
│   ├── core/
│   │   ├── metrics.py          # IC/RankIC and risk metrics
│   │   ├── portfolio.py        # constrained long-short optimizer
│   │   ├── types.py            # shared dataclasses
│   │   └── utils.py            # seed management
│   ├── project1/
│   │   ├── data.py             # synthetic A-share cross-section generator
│   │   ├── graph.py            # heterogeneous graph + multi-head GAT embedding
│   │   ├── model.py            # deterministic ridge forecaster wrapper
│   │   └── pipeline.py         # full alpha pipeline
│   ├── project2/
│   │   ├── pricing.py          # Asian MC engine + variance reduction + Greeks + IV
│   │   └── pipeline.py         # pricing and risk report pipeline
│   ├── project3/
│   │   ├── env.py              # execution environment + market impact
│   │   ├── agents.py           # PPO-style / SAC-style / DT-style policies
│   │   └── pipeline.py         # training + evaluation pipeline
│   ├── cli.py                  # unified command-line interface
│   └── main.py                 # package entrypoint
├── tests/
│   ├── test_project1.py
│   ├── test_project2.py
│   ├── test_project3.py
│   └── test_cli.py
├── configs/
│   ├── project1.yaml
│   ├── project2.yaml
│   └── project3.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### 2. Run each pipeline

```bash
quant-showcase project1
quant-showcase project2
quant-showcase project3
```

Or run all:

```bash
quant-showcase all
```

Each command prints structured JSON metrics to stdout.

### Python API Shape (Unified)

```python
from quant_showcase.project1 import Project1Config, run

cfg = Project1Config(seed=42)
report = run(config=cfg)
print(report.name, report.metrics)
```

The same `Config -> run(config) -> report` shape is used in all three projects.

### 3. Run tests

```bash
pytest -q
```

## Project 1: Multi-Factor Alpha + GAT Stock Representation

### Problem
Classical cross-sectional alpha models often treat each stock as independent and ignore explicit inter-stock topology.

### Hypothesis
Graph-aware embeddings improve return signal quality when fused with factor features.

### Method Stack

- Heterogeneous graph construction
  - sector co-membership
  - supply-chain links
  - style-factor similarity
- Multi-head GAT-style embedding
- Feature fusion: `[graph_embedding || engineered_factors]`
- Forecasting model: deterministic ridge baseline (designed to be swappable with LGBM/XGBoost)
- Constrained portfolio optimizer with turnover penalty

### API Contract

- Config: `Project1Config`
- Entry: `quant_showcase.project1.run(config=...)`
- Output: `PipelineReport` with `ic`, `rank_ic`, exposure, and volatility metrics

### Optimization Linkage to Resume
The portfolio step uses a Lagrangian objective:

`L(w) = w^T Σ w - λ α^T w + ρ ||w - w_prev||_1`

subject to long/short constraints and exposure controls.

## Project 2: Asian Option Pricing + Derivatives Risk

### Problem
Path-dependent payoffs are noisy under naive Monte Carlo and difficult to use in desk workflows.

### Hypothesis
Variance reduction and robust risk outputs produce stable prices with lower estimator variance.

### Method Stack

- Plain Monte Carlo
- Antithetic variates
- Control variates (geometric Asian closed-form as control)
- Finite-difference Greeks (`Delta`, `Gamma`, `Vega`)
- Implied volatility inversion (root solving)

### API Contract

- Config: `Project2Config`
- Entry: `quant_showcase.project2.run(config=...)`
- Output: `PipelineReport` with pricing, standard errors, variance reduction, and Greeks

## Project 3: Deep RL for Intraday Futures Execution

### Problem
Intraday execution requires balancing completion, impact, and timing under uncertainty.

### Hypothesis
Combining online policy learning with offline schedule priors improves implementation shortfall.

### Method Stack

- Execution environment with Almgren-Chriss-style temporary/permanent impact
- PPO-style policy interface with lightweight policy-improvement loop
- SAC-style stochastic actor interface
- Decision Transformer-style offline scheduling policy abstraction
- Policy evaluation based on implementation shortfall (bps)

### API Contract

- Config: `Project3Config`
- Entry: `quant_showcase.project3.run(config=...)`
- Output: `PipelineReport` with policy-level IS and completion metrics

## Reproducibility and scope

This repository is built for **academic and interview demonstration**. It is not a direct production trading deployment.

- Data in this repo is synthetic by default for reproducibility
- Interfaces are designed to be replaced with real internal data sources
- Optional dependencies (`torch`, `torch-geometric`, `lightgbm`, `xgboost`) are included to support extension

## Mapping to resume claims

This codebase is designed to make the following claims inspectable through architecture and implementation patterns:

- topology-aware multi-factor alpha system with constrained portfolio optimization
- variance-reduced Monte Carlo pricing for Asian options with risk sensitivity outputs
- RL-based execution framework combining online and offline policy styles

## License

For graduate application and technical demonstration.
