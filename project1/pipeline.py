"""
project1/pipeline.py
────────────────────
Two execution paths / 两条运行路径：

run()       — Production: heterogeneous graph + RidgeForecaster (zero-dependency)
              生产路径：异构图 + RidgeForecaster（零依赖，CI 友好）
run_deep()  — Deep: sparsemax learnable graph + GraphAlphaNet (requires PyTorch)
              深度路径：Sparsemax 可学习图 + GraphAlphaNet（需要 PyTorch）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_showcase.core.metrics import pearson_corr, rank_ic
from quant_showcase.core.portfolio import optimize_long_short
from quant_showcase.core.types import PipelineReport
from .data import make_synthetic_cross_section
from .graph import build_heterogeneous_graph, build_learnable_graph, numpy_gat_embedding
from .model import RidgeForecaster, TemporalStockMemory


RESEARCH_NARRATIVE = {
    "problem": (
        "Classical cross-sectional alpha models treat stocks as isolated entities, "
        "ignoring inter-stock topology (sector co-movement, supply-chain links, "
        "style-factor clusters)."
        # 传统截面 alpha 模型将股票视为孤立实体，
        # 忽略股票间的拓扑结构（行业共动、供应链联系、风格因子聚类）
    ),
    "hypothesis": (
        "Graph-aware embeddings improve cross-sectional IC when combined with "
        "200+ engineered alpha factors."
        # 图感知 embedding 与 200+ alpha 因子融合时，能提升截面 IC
    ),
    "method_stack": [
        # Heterogeneous graph: sector (symmetric) / supply-chain (directed) / style-factor
        # 异构图：sector（对称）/ supply-chain（有向）/ style-factor
        "Heterogeneous graph: sector (symmetric) / supply-chain (directed) / style-factor",
        # OR: learnable sparse graph via Sparsemax(ZZ^T/√d) — no domain labels
        # 或：Sparsemax(ZZ^T/√d) 可学习稀疏图——无需领域标签
        "OR learnable sparse graph via Sparsemax(ZZ^T/√d) — no domain labels needed",
        # Multi-head GAT embedding (NumPy fallback or PyTorch GATConv)
        # 多头 GAT embedding（NumPy fallback 或 PyTorch GATConv）
        "Multi-head GAT embedding (NumPy fallback or PyTorch GATConv)",
        # concat(GAT_embed, alpha_factors) → Ridge / GraphAlphaNet
        # concat(GAT_embed, alpha_因子) → Ridge / GraphAlphaNet
        "concat(GAT_embed || alpha_factors) → Ridge / GraphAlphaNet",
        # Lagrangian long-short optimization with Ledoit-Wolf covariance shrinkage
        # Lagrangian 长短组合优化，使用 Ledoit-Wolf 协方差收缩
        "Lagrangian long-short optimization with Ledoit-Wolf covariance shrinkage",
        # Turnover penalty: L(w) = w^T Σ w − λ α^T w + ρ‖w − w_prev‖₁
        # 换手惩罚：L(w) = w^T Σ w − λ α^T w + ρ‖w − w_prev‖₁
        "Turnover penalty: L(w) = w^T Σ w − λ α^T w + ρ‖w − w_prev‖₁",
    ],
}


@dataclass(frozen=True)
class Project1Config:
    seed: int             = 42
    n_stocks: int         = 400
    n_factors: int        = 220
    lookback_days: int    = 120
    train_ratio: float    = 0.7
    ridge_alpha: float    = 1.2
    top_k: int            = 30
    risk_aversion: float  = 3.0
    tc_penalty: float     = 0.0008
    use_learnable_graph: bool = False   # True = sparsemax path / True = sparsemax 路径


def _winsorize_zscore(X: np.ndarray, q: float = 0.01) -> np.ndarray:
    """
    Cross-sectional winsorize then z-score.
    截面 winsorize 后再 z-score。

    Winsorize at q and 1-q quantiles to clip extreme outliers before
    z-scoring — otherwise single outlier stocks dominate the mean/std.
    在 z-score 前先按 q 和 1-q 分位数截断极端异常值——
    否则少数极端股票会主导均值和标准差的计算。
    """
    lo = np.nanquantile(X, q, axis=0)
    hi = np.nanquantile(X, 1 - q, axis=0)
    X  = np.clip(X, lo, hi)
    return (X - X.mean(0)) / (X.std(0) + 1e-9)


def _ledoit_wolf_cov(returns: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    """
    Ledoit-Wolf linear shrinkage covariance.
    Ledoit-Wolf 线性收缩协方差。

    Σ̂ = (1-s)·Σ_sample + s·μ_diag·I
    Prevents singular covariance matrix when n_stocks >> n_days.
    防止股票数 >> 观测天数时协方差矩阵奇异。
    """
    Sigma = np.cov(returns.T)
    return (1 - shrink) * Sigma + shrink * np.eye(len(Sigma)) * np.diag(Sigma).mean()


def run(seed: int = 42, config: Project1Config | None = None) -> PipelineReport:
    """
    Production path: heterogeneous graph + NumPy GAT + RidgeForecaster.
    生产路径：异构图 + NumPy GAT + RidgeForecaster。

    Zero deep learning dependencies, fully deterministic, suitable for CI.
    零深度学习依赖，完全确定性，适合持续集成环境。
    """
    cfg = config or Project1Config(seed=seed)
    cs  = make_synthetic_cross_section(
        n_stocks=cfg.n_stocks, n_factors=cfg.n_factors,
        lookback_days=cfg.lookback_days, seed=cfg.seed,
    )

    # Preprocess: winsorize + z-score cross-sectionally
    # 预处理：截面 winsorize + z-score
    X = _winsorize_zscore(cs.features)

    # Graph construction: heterogeneous or learnable
    # 图构建：异构图或可学习图
    if cfg.use_learnable_graph:
        Zn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        edge_index, edge_weight = build_learnable_graph(Zn, top_k=15)
    else:
        edge_index, edge_weight = build_heterogeneous_graph(
            cs.sector_matrix, cs.supplychain_matrix, cs.style_matrix
        )

    # GAT embedding via NumPy (no PyTorch needed)
    # NumPy 实现的 GAT embedding（不需要 PyTorch）
    embedding = numpy_gat_embedding(
        X, edge_index, edge_weight, hidden_dim=32, heads=4, seed=cfg.seed,
    )

    # Feature fusion: [GAT_embed(128) || alpha_factors(220)]
    # 特征融合：[GAT_embed(128) || alpha_因子(220)]
    full_x = np.concatenate([embedding, X], axis=1)
    train_n = int(cfg.train_ratio * full_x.shape[0])

    model = RidgeForecaster(alpha=cfg.ridge_alpha)
    model.fit(full_x[:train_n], cs.target[:train_n])
    pred   = model.predict(full_x[train_n:])
    target = cs.target[train_n:]

    # IC and Rank IC evaluation
    # IC 和 Rank IC 评估
    ic  = pearson_corr(pred, target)
    ric = rank_ic(pred, target)

    # Ledoit-Wolf covariance + Lagrangian portfolio optimization
    # Ledoit-Wolf 协方差 + Lagrangian 组合优化
    rets = np.diff(cs.prices_window, axis=0) / (cs.prices_window[:-1] + 1e-12)
    cov  = _ledoit_wolf_cov(rets[-60:])

    alpha_scores = model.predict(full_x)
    port = optimize_long_short(
        alpha_scores, cov,
        top_k=cfg.top_k, lam=cfg.risk_aversion, tc_penalty=cfg.tc_penalty,
    )

    graph_type = "learnable_sparsemax" if cfg.use_learnable_graph else "heterogeneous"
    return PipelineReport(name=f"project1_gat_{graph_type}", metrics={
        "ic":                float(ic),
        "rank_ic":           float(ric),
        "long_count":        float(port.long_count),
        "short_count":       float(port.short_count),
        "gross_exposure":    port.gross_exposure,
        "annual_volatility": port.annual_volatility,
        "expected_alpha":    port.expected_alpha,
        "n_edges":           float(edge_index.shape[1]),
        # Graph density: fraction of all possible directed edges that are present
        # 图密度：实际边数占所有可能有向边的比例
        "graph_density":     float(edge_index.shape[1] / (cfg.n_stocks * (cfg.n_stocks - 1))),
    })


def run_deep(seed: int = 42) -> PipelineReport:
    """
    Deep path: GraphAlphaNet (PyTorch + PyG) with Pearson loss and full training loop.
    深度路径：GraphAlphaNet（PyTorch + PyG），Pearson loss，完整训练循环。

    Requires: pip install torch torch-geometric
    依赖：pip install torch torch-geometric
    """
    try:
        from .model import GraphAlphaNet
        import torch
    except ImportError:
        print("[run_deep] PyTorch not found, falling back to run()")
        return run(seed=seed)

    cfg = Project1Config(seed=seed, use_learnable_graph=True)
    cs  = make_synthetic_cross_section(
        n_stocks=cfg.n_stocks, n_factors=cfg.n_factors,
        lookback_days=cfg.lookback_days, seed=cfg.seed,
    )
    X  = _winsorize_zscore(cs.features)
    Zn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    ei, ew = build_learnable_graph(Zn, top_k=15)

    # Build snapshot list (same cross-section repeated to simulate multi-period)
    # 构建 snapshot 列表（同一截面重复，模拟多期训练）
    snapshots = [(X.astype(np.float32), ei, ew, cs.target)] * 5

    model = GraphAlphaNet(in_dim=cfg.n_factors, heads=4)
    model.fit(snapshots, lr=1e-3, epochs=30, device="cpu")

    model.eval()
    with torch.no_grad():
        Xt  = torch.tensor(X, dtype=torch.float32)
        eit = torch.tensor(ei, dtype=torch.long)
        ewt = torch.tensor(ew, dtype=torch.float32)
        pred_all = model(Xt, eit, ewt).squeeze(-1).numpy()

    n_val = int(0.3 * len(pred_all))
    ic    = pearson_corr(pred_all[-n_val:], cs.target[-n_val:])
    ric   = rank_ic(pred_all[-n_val:], cs.target[-n_val:])

    rets = np.diff(cs.prices_window, axis=0) / (cs.prices_window[:-1] + 1e-12)
    cov  = _ledoit_wolf_cov(rets[-60:])
    port = optimize_long_short(pred_all, cov, top_k=cfg.top_k,
                               lam=cfg.risk_aversion, tc_penalty=cfg.tc_penalty)

    return PipelineReport(name="project1_gat_deep", metrics={
        "ic":                float(ic),
        "rank_ic":           float(ric),
        "long_count":        float(port.long_count),
        "short_count":       float(port.short_count),
        "annual_volatility": port.annual_volatility,
        "expected_alpha":    port.expected_alpha,
    })
