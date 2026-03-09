from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CrossSectionData:
    features: np.ndarray
    target: np.ndarray
    sector_matrix: np.ndarray
    supplychain_matrix: np.ndarray
    style_matrix: np.ndarray
    prices_window: np.ndarray


@dataclass
class PortfolioResult:
    weights: np.ndarray
    expected_alpha: float
    annual_volatility: float
    gross_exposure: float
    long_count: int
    short_count: int


@dataclass
class PricingResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float


@dataclass
class ExecutionEvaluation:
    name: str
    mean_is_bps: float
    p95_is_bps: float
    completion_rate: float


@dataclass
class PipelineReport:
    name: str
    metrics: Dict[str, float]
