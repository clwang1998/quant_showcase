from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_showcase.core.metrics import pearson_corr, rank_ic
from quant_showcase.core.portfolio import optimize_long_short
from quant_showcase.core.types import PipelineReport
from .data import make_synthetic_cross_section
from .graph import build_heterogeneous_graph, simple_multihead_gat_embedding
from .model import RidgeForecaster


RESEARCH_NARRATIVE = {
    "problem": "Classical cross-sectional alpha models ignore explicit inter-stock topology.",
    "hypothesis": "Graph-aware stock embeddings improve cross-sectional signal quality when combined with factor features.",
    "method_stack": [
        "Heterogeneous graph construction (sector/supply-chain/style)",
        "Multi-head GAT-style embedding",
        "Embedding + factor fusion for return forecasting",
        "Lagrangian long-short optimization with turnover penalty",
    ],
}


@dataclass(frozen=True)
class Project1Config:
    seed: int = 42
    n_stocks: int = 400
    n_factors: int = 220
    lookback_days: int = 120
    train_ratio: float = 0.7
    ridge_alpha: float = 1.2
    top_k: int = 30
    risk_aversion: float = 3.0
    tc_penalty: float = 0.0008


def run(seed: int = 42, config: Project1Config | None = None) -> PipelineReport:
    cfg = config or Project1Config(seed=seed)
    cs = make_synthetic_cross_section(
        n_stocks=cfg.n_stocks,
        n_factors=cfg.n_factors,
        lookback_days=cfg.lookback_days,
        seed=cfg.seed,
    )

    edge_index, edge_weight = build_heterogeneous_graph(
        cs.sector_matrix, cs.supplychain_matrix, cs.style_matrix
    )
    embedding = simple_multihead_gat_embedding(cs.features, edge_index, edge_weight, seed=cfg.seed)

    full_x = np.concatenate([embedding, cs.features], axis=1)
    train_n = int(cfg.train_ratio * full_x.shape[0])

    model = RidgeForecaster(alpha=cfg.ridge_alpha)
    model.fit(full_x[:train_n], cs.target[:train_n])
    pred = model.predict(full_x[train_n:])
    target = cs.target[train_n:]

    ic = pearson_corr(pred, target)
    ric = rank_ic(pred, target)

    rets = np.diff(cs.prices_window, axis=0) / (cs.prices_window[:-1] + 1e-12)
    cov = np.cov(rets[-60:].T)

    alpha_scores = model.predict(full_x)
    port = optimize_long_short(
        alpha_scores,
        cov,
        top_k=cfg.top_k,
        lam=cfg.risk_aversion,
        tc_penalty=cfg.tc_penalty,
    )

    metrics = {
        "ic": float(ic),
        "rank_ic": float(ric),
        "long_count": float(port.long_count),
        "short_count": float(port.short_count),
        "gross_exposure": port.gross_exposure,
        "annual_volatility": port.annual_volatility,
        "expected_alpha": port.expected_alpha,
    }
    return PipelineReport(name="project1_gat_multifactor", metrics=metrics)
