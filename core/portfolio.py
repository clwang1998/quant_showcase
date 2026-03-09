"""
core/portfolio.py
─────────────────
Lagrangian long-short portfolio optimization via projected gradient descent.
Lagrangian 多约束长短组合优化，使用投影梯度下降（PGD）。

Objective / 目标函数：
  min  w^T Σ w  −  λ α^T w  +  ρ ‖w − w_prev‖₁
  s.t. w_long ≥ 0, w_short ≤ 0
       ‖w_long‖₁ = 1, ‖w_short‖₁ = 1

[OpenClaw] PGD framework and projection structure
[OpenClaw] PGD 框架和投影结构
[Ours]     Ledoit-Wolf covariance shrinkage + subgradient annotation
[我们]     Ledoit-Wolf 协方差收缩 + 次梯度注释
"""
from __future__ import annotations

import numpy as np

from .types import PortfolioResult


def ledoit_wolf_cov(returns: np.ndarray, shrink: float = 0.2) -> np.ndarray:
    """
    Linear shrinkage covariance estimator (Ledoit & Wolf, 2004).
    线性收缩协方差估计（Ledoit & Wolf, 2004）。

    Formula / 公式：
      Σ̂ = (1-s)·Σ_sample + s·μ_diag·I

    Why shrinkage?
    为什么需要收缩？
      Sample covariance is ill-conditioned when n_stocks >> n_days.
      In high dimensions, eigenvalues are severely biased (Marchenko-Pastur).
      Shrinking toward a scaled identity stabilizes inversion and reduces
      estimation error in PGD.
      当股票数 >> 观测天数时，样本协方差矩阵条件数极差。
      高维下特征值严重偏差（Marchenko-Pastur 定理）。
      向缩放单位矩阵收缩可以稳定矩阵求逆，降低 PGD 中的估计误差。

    shrink=0.2 is the empirical default; oracle optimal shrinkage requires
    Marchenko-Pastur analysis of the n/p ratio.
    shrink=0.2 是经验默认值；最优收缩系数需要根据 n/p 做 Marchenko-Pastur 分析。
    """
    Sigma = np.cov(returns.T)
    return (1 - shrink) * Sigma + shrink * np.eye(len(Sigma)) * np.diag(Sigma).mean()


def optimize_long_short(
    alpha: np.ndarray,
    cov: np.ndarray,
    top_k: int = 50,
    lam: float = 4.0,
    tc_penalty: float = 0.001,
    prev_weights: np.ndarray | None = None,
    max_iter: int = 300,
    lr: float = 0.01,
) -> PortfolioResult:
    """
    PGD (projected gradient descent) with transaction cost subgradient.
    带交易成本次梯度的 PGD（投影梯度下降）。

    Gradient components / 梯度各项：
      ∂/∂w [w^T Σ w]       = 2Σw          (risk term / 风险项)
      ∂/∂w [−λ α^T w]      = −λα          (alpha term / alpha 项)
      ∂/∂w [ρ‖w−w_prev‖₁] = ρ·sign(w−w_prev)

    Note on the L1 subgradient:
    关于 L1 次梯度的说明：
      ‖w − w_prev‖₁ is non-differentiable at w = w_prev.
      ‖w − w_prev‖₁ 在 w = w_prev 处不可微。
      np.sign() returns a valid subgradient (0 at the kink, ±1 elsewhere).
      np.sign() 返回一个有效次梯度（在不连续点处返回 0，其他处返回 ±1）。
      PGD convergence is guaranteed for convex objectives with subgradients.
      对于带次梯度的凸目标函数，PGD 收敛性有理论保证。

    PGD projection steps / PGD 投影步骤：
      1. Gradient step                  梯度下降一步
      2. Zero out weights outside top_k  把 top_k 外的权重清零（支撑集约束）
      3. Clip: long ≥ 0, short ≤ 0     截断：多头 ≥ 0，空头 ≤ 0
      4. Normalize: ‖w_long‖₁=1, ‖w_short‖₁=1  归一化使 L1 范数为 1
    """
    n = alpha.shape[0]
    if prev_weights is None:
        prev_weights = np.zeros(n)

    # Initialize with equal-weight long-top / short-bottom stocks
    # 初始化：等权重持有分数最高的多头和最低的空头
    order     = np.argsort(alpha)
    short_idx = order[:top_k]
    long_idx  = order[-top_k:]

    w            = np.zeros(n)
    w[long_idx]  =  1.0 / top_k
    w[short_idx] = -1.0 / top_k

    for _ in range(max_iter):
        # Compute full gradient (risk + alpha + transaction cost subgradient)
        # 计算完整梯度（风险项 + alpha 项 + 交易成本次梯度）
        g  = 2.0 * cov @ w - lam * alpha + tc_penalty * np.sign(w - prev_weights)
        w -= lr * g

        # Project: zero out stocks outside the trading universe
        # 投影：清零不在交易股票池内的权重
        mask = np.ones(n, dtype=bool)
        mask[long_idx]  = False
        mask[short_idx] = False
        w[mask] = 0.0

        # Project: enforce long ≥ 0, short ≤ 0
        # 投影：强制多头 ≥ 0，空头 ≤ 0
        w[long_idx]  = np.clip(w[long_idx],  0.0, None)
        w[short_idx] = np.clip(w[short_idx], None, 0.0)

        # Project: normalize to unit L1 norm in each leg
        # 投影：每条腿归一化到 L1 范数为 1
        ls = w[long_idx].sum()
        if ls > 1e-12:
            w[long_idx] /= ls
        ss = np.abs(w[short_idx]).sum()
        if ss > 1e-12:
            w[short_idx] /= ss

    return PortfolioResult(
        weights=w,
        expected_alpha=float(alpha @ w),
        # Annualized vol: sqrt(w^T Σ w * 252) assuming daily returns
        # 年化波动率：sqrt(w^T Σ w * 252)，假设日度收益率
        annual_volatility=float(np.sqrt(max(w @ cov @ w, 0.0) * 252.0)),
        gross_exposure=float(np.abs(w).sum()),
        long_count=int((w > 1e-8).sum()),
        short_count=int((w < -1e-8).sum()),
    )
