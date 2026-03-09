from __future__ import annotations

import numpy as np

from .types import PortfolioResult


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
    n = alpha.shape[0]
    if prev_weights is None:
        prev_weights = np.zeros(n)

    order = np.argsort(alpha)
    short_idx = order[:top_k]
    long_idx = order[-top_k:]

    w = np.zeros(n)
    w[long_idx] = 1.0 / top_k
    w[short_idx] = -1.0 / top_k

    def grad(x: np.ndarray) -> np.ndarray:
        g_risk = 2.0 * cov @ x
        g_alpha = -lam * alpha
        g_tc = tc_penalty * np.sign(x - prev_weights)
        return g_risk + g_alpha + g_tc

    for _ in range(max_iter):
        w -= lr * grad(w)

        # Keep only selected universe; prevent exposure leak.
        mask = np.ones(n, dtype=bool)
        mask[long_idx] = False
        mask[short_idx] = False
        w[mask] = 0.0

        w[long_idx] = np.clip(w[long_idx], 0.0, None)
        w[short_idx] = np.clip(w[short_idx], None, 0.0)

        lsum = w[long_idx].sum()
        if lsum > 1e-12:
            w[long_idx] /= lsum

        ssum = np.abs(w[short_idx]).sum()
        if ssum > 1e-12:
            w[short_idx] /= ssum

    exp_alpha = float(alpha @ w)
    ann_vol = float(np.sqrt(max(w @ cov @ w, 0.0) * 252.0))
    gross = float(np.abs(w).sum())
    return PortfolioResult(
        weights=w,
        expected_alpha=exp_alpha,
        annual_volatility=ann_vol,
        gross_exposure=gross,
        long_count=int(np.sum(w > 1e-8)),
        short_count=int(np.sum(w < -1e-8)),
    )
