from __future__ import annotations

import numpy as np


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = np.linalg.norm(x0) * np.linalg.norm(y0) + 1e-12
    return float(np.dot(x0, y0) / denom)


def rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    pred_rank = np.argsort(np.argsort(pred))
    tgt_rank = np.argsort(np.argsort(target))
    return pearson_corr(pred_rank.astype(float), tgt_rank.astype(float))


def annualized_volatility(portfolio_returns: np.ndarray) -> float:
    return float(np.std(portfolio_returns) * np.sqrt(252))


def max_drawdown(equity_curve: np.ndarray) -> float:
    running_peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - running_peak) / (running_peak + 1e-12)
    return float(np.min(dd))
