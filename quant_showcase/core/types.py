"""
core/types.py
─────────────
Shared data types and result containers.
共享数据类型和结果容器。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CrossSectionData:
    """
    One cross-sectional snapshot for Project 1.
    Project 1 的一个截面快照。

    features         : (N, F) factor matrix / 因子矩阵
    target           : (N,) forward return labels / 未来收益率标签
    sector_matrix    : (N, N) same-sector indicator / 同行业矩阵
    supplychain_matrix: (N, N) supply-chain strength / 供应链强度矩阵
    style_matrix     : (N, N) style-factor correlation / 风格因子相关矩阵
    prices_window    : (T, N) historical price matrix / 历史价格矩阵
    """
    features:           np.ndarray
    target:             np.ndarray
    sector_matrix:      np.ndarray
    supplychain_matrix: np.ndarray
    style_matrix:       np.ndarray
    prices_window:      np.ndarray


@dataclass
class PortfolioResult:
    """
    Output of portfolio optimization step.
    组合优化步骤的输出。
    """
    weights:           np.ndarray    # (N,) portfolio weights / 组合权重
    expected_alpha:    float         # α^T w / alpha 期望值
    annual_volatility: float         # √(w^T Σ w × 252) / 年化波动率
    gross_exposure:    float         # ‖w‖₁ / 总杠杆
    long_count:        int           # number of long positions / 多头股票数
    short_count:       int           # number of short positions / 空头股票数


@dataclass
class PricingResult:
    """
    Monte Carlo pricing output for Project 2.
    Project 2 的 Monte Carlo 定价输出。
    """
    price:    float   # estimated option price / 期权价格估计
    std_error: float  # standard error of MC estimate / MC 估计的标准误
    ci_low:   float   # 95% confidence interval lower bound / 95% 置信区间下界
    ci_high:  float   # 95% confidence interval upper bound / 95% 置信区间上界


@dataclass
class ExecutionEvaluation:
    """
    Execution quality metrics for Project 3.
    Project 3 的执行质量指标。
    """
    name:            str    # agent name / agent 名称
    mean_is_bps:     float  # mean implementation shortfall in bps / 平均实施成本（bps）
    p95_is_bps:      float  # 95th percentile IS in bps / IS 的 95 分位数（bps）
    completion_rate: float  # fraction of episodes with full order completion / 完整完成率


@dataclass
class PipelineReport:
    """
    Unified output container for all three project pipelines.
    三个项目 pipeline 的统一输出容器。
    """
    name:    str            # pipeline identifier / pipeline 标识符
    metrics: Dict[str, float]  # key metrics dictionary / 关键指标字典
