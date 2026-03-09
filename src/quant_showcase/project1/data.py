from __future__ import annotations

import numpy as np

from quant_showcase.core.types import CrossSectionData


def make_synthetic_cross_section(
    n_stocks: int = 400,
    n_factors: int = 220,
    lookback_days: int = 120,
    seed: int = 42,
) -> CrossSectionData:
    rng = np.random.default_rng(seed)

    prices = 100.0 * np.exp(np.cumsum(rng.standard_normal((lookback_days, n_stocks)) * 0.015, axis=0))
    volumes = np.exp(rng.standard_normal((lookback_days, n_stocks)) + 12.0)

    base_factors = rng.standard_normal((n_stocks, n_factors)).astype(np.float32)
    momentum = prices[-1] / (prices[-21] + 1e-12) - 1.0
    value = -np.log(prices[-1])
    quality = rng.standard_normal(n_stocks) * 0.2
    base_factors[:, 0] = momentum
    base_factors[:, 1] = value
    base_factors[:, 2] = quality

    true_beta = np.zeros(n_factors)
    true_beta[:20] = rng.standard_normal(20) * 0.08
    target = (base_factors @ true_beta + rng.standard_normal(n_stocks) * 0.05).astype(np.float32)

    sectors = rng.integers(0, 20, size=n_stocks)
    sector_matrix = (sectors[:, None] == sectors[None, :]).astype(float)
    np.fill_diagonal(sector_matrix, 0.0)

    supplychain_matrix = rng.random((n_stocks, n_stocks)) * (rng.random((n_stocks, n_stocks)) > 0.985)
    style_matrix = np.corrcoef(base_factors[:, :30])
    style_matrix = np.nan_to_num(style_matrix, nan=0.0)

    return CrossSectionData(
        features=base_factors,
        target=target,
        sector_matrix=sector_matrix,
        supplychain_matrix=supplychain_matrix,
        style_matrix=style_matrix,
        prices_window=prices,
    )
