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

  ## 融合策略

| 模块 | 现有infra 贡献 | 我的贡献 |
|------|---------------|-----------|
| **工程框架** | `src/` 包结构、CLI、config YAML、pytest | — |
| **Project 1 · 图** | `build_heterogeneous_graph` 框架 | 修正 supply-chain 有向性；新增 `build_learnable_graph`（Sparsemax） |
| **Project 1 · 模型** | `RidgeForecaster` | `GraphAlphaNet`（PyTorch GATConv + Pearson loss）；`TemporalStockMemory` EMA |
| **Project 1 · 组合** | PGD 框架 | Ledoit-Wolf 协方差收缩；完整次梯度注释 |
| **Project 2** | `AsianOptionEngine` 框架 | Heston 路径（Euler-Milstein）；pathwise delta；IV Brent 50 iter |
| **Project 3 · Env** | `ExecutionEnv` 5维状态 | 扩展到 13 维（LOB depth / momentum / fill history） |
| **Project 3 · Agents** | `LinearPolicy` 骨架 | PPO GAE-λ；SAC tanh squash + 自动温度；DT TWAP/VWAP-U |

## 快速开始

```bash
pip install -e .
quant-showcase project1   # 异构图 + RidgeForecaster（零依赖）
quant-showcase project2   # 亚式期权（Plain / Antithetic / CV / Heston）
quant-showcase project3   # RL 执行（PPO / SAC / DT-VWAP）
quant-showcase all        # 全部运行
```

## 深度模式（需要 PyTorch + PyG）

```python
from quant_showcase.project1.pipeline import run_deep
report = run_deep()   # GraphAlphaNet + Pearson loss
```

## 目录结构

```
src/quant_showcase/
├── core/
│   ├── types.py      # 数据类型（CrossSectionData, PortfolioResult …）
│   ├── metrics.py    # IC / Rank IC / max drawdown
│   ├── portfolio.py  # PGD + Ledoit-Wolf
│   └── utils.py
├── project1/
│   ├── graph.py      # 异构图 + Sparsemax 可学习图 + NumPy GAT
│   ├── model.py      # RidgeForecaster / GraphAlphaNet / TemporalStockMemory
│   ├── data.py       # 合成数据生成
│   └── pipeline.py   # run() / run_deep()
├── project2/
│   ├── pricing.py    # GBM + Heston + 方差缩减 + Greeks + IV
│   └── pipeline.py
└── project3/
    ├── env.py        # Almgren-Chriss 执行环境（5/13 维状态）
    ├── agents.py     # PPO-GAE / SAC-auto-α / DT-VWAP
    └── pipeline.py
```

## 关键设计决策

### Sparsemax vs Softmax（Project 1）
- Softmax：所有边权重 > 0，需手工 threshold
- **Sparsemax**：投影到概率单纯形，大多数边精确为 0，无需超参数

### supply-chain 边为有向边（Project 1）
- OpenClaw 原版：`symmetric=False`（已正确）但无注释
- 经济含义：上游→下游 ≠ 下游→上游，不对称传导

### Pearson Loss vs MSE（Project 1）
- Ubiquant 竞赛评估指标是 per-time_id Pearson IC
- 直接最小化 `-corr(pred, target)` 而非 MSE

### GAE-λ（Project 3）
- 纯 REINFORCE：高方差
- **GAE**：在偏差-方差之间折中，λ=0 → TD，λ=1 → MC

### tanh Jacobian 修正（Project 3 SAC）
- `log π(a|s) = log π_unc(a') - Σ log(1 - tanh²(a'_i))`
- 忽略此项会导致熵估计偏低，alpha 过热

## License

For graduate application and technical demonstration.
