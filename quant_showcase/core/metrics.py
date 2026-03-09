"""
core/metrics.py
───────────────
Evaluation metrics for alpha signal quality and portfolio performance.
Alpha 信号质量和组合表现的评估指标。
"""
from __future__ import annotations

import numpy as np


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """
    Pearson IC (Information Coefficient).
    Pearson IC（信息系数）。

    Measures linear correlation between predicted and realized returns.
    衡量预测收益率与实际收益率之间的线性相关性。

    Returns value in [−1, 1]; higher is better for long-short alpha.
    返回值在 [−1, 1]，对于多空 alpha 策略越高越好。
    """
    x0    = x - np.mean(x)
    y0    = y - np.mean(y)
    denom = np.linalg.norm(x0) * np.linalg.norm(y0) + 1e-12
    return float(np.dot(x0, y0) / denom)


def rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Rank IC (Spearman correlation between predicted and actual return ranks).
    Rank IC（预测收益排名与实际收益排名的 Spearman 相关）。

    More robust than Pearson IC: insensitive to outliers and monotone
    transformations of predictions.
    比 Pearson IC 更稳健：对异常值和预测值的单调变换不敏感。
    """
    pred_rank = np.argsort(np.argsort(pred))
    tgt_rank  = np.argsort(np.argsort(target))
    return pearson_corr(pred_rank.astype(float), tgt_rank.astype(float))


def annualized_volatility(portfolio_returns: np.ndarray) -> float:
    """
    Annualized volatility assuming daily returns (252 trading days).
    年化波动率，假设日度收益率（252个交易日）。
    """
    return float(np.std(portfolio_returns) * np.sqrt(252))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum drawdown: largest peak-to-trough decline.
    最大回撤：净值曲线中最大的峰谷下跌幅度。

    Returns a negative number (loss fraction from peak).
    返回负数（从峰值下跌的比例）。
    """
    running_peak = np.maximum.accumulate(equity_curve)
    dd           = (equity_curve - running_peak) / (running_peak + 1e-12)
    return float(np.min(dd))
