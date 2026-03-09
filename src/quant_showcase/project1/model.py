from __future__ import annotations

import numpy as np


class RidgeForecaster:
    """Deterministic ridge forecaster with closed-form solution."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.coef_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        xtx = x.T @ x
        reg = self.alpha * np.eye(xtx.shape[0])
        self.coef_ = np.linalg.solve(xtx + reg, x.T @ y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model has not been fit.")
        return x @ self.coef_
