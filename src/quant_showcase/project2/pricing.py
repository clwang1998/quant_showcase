from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass
class AsianOptionParams:
    s0: float = 100.0
    k: float = 100.0
    t: float = 1.0
    r: float = 0.03
    q: float = 0.00
    sigma: float = 0.25
    n_steps: int = 252
    option_type: str = "call"


class AsianOptionEngine:
    def __init__(self, params: AsianOptionParams, n_paths: int = 80_000) -> None:
        self.p = params
        self.n_paths = n_paths

    def simulate_paths(self, antithetic: bool = False, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        dt = self.p.t / self.p.n_steps
        drift = (self.p.r - self.p.q - 0.5 * self.p.sigma ** 2) * dt
        vol = self.p.sigma * np.sqrt(dt)

        if antithetic:
            half = self.n_paths // 2
            z = rng.standard_normal((half, self.p.n_steps))
            z = np.vstack([z, -z])
        else:
            z = rng.standard_normal((self.n_paths, self.p.n_steps))

        log_s = np.log(self.p.s0) + np.cumsum(drift + vol * z, axis=1)
        s = np.exp(log_s)
        s = np.hstack([np.full((s.shape[0], 1), self.p.s0), s])
        return s

    def _discounted_payoff(self, avg: np.ndarray) -> np.ndarray:
        if self.p.option_type == "call":
            payoff = np.maximum(avg - self.p.k, 0.0)
        else:
            payoff = np.maximum(self.p.k - avg, 0.0)
        return np.exp(-self.p.r * self.p.t) * payoff

    def geometric_closed_form(self) -> float:
        m = self.p.n_steps
        sigma_g = self.p.sigma * np.sqrt((2 * m + 1) / (6 * (m + 1)))
        mu_g = 0.5 * (self.p.r - self.p.q - 0.5 * self.p.sigma ** 2) + 0.5 * sigma_g ** 2

        d1 = (np.log(self.p.s0 / self.p.k) + (mu_g + 0.5 * sigma_g ** 2) * self.p.t) / (sigma_g * np.sqrt(self.p.t) + 1e-12)
        d2 = d1 - sigma_g * np.sqrt(self.p.t)

        disc = np.exp(-self.p.r * self.p.t)
        fwd = self.p.s0 * np.exp(mu_g * self.p.t)
        nd1 = _norm_cdf(d1)
        nd2 = _norm_cdf(d2)
        if self.p.option_type == "call":
            return float(disc * (fwd * nd1 - self.p.k * nd2))
        return float(disc * (self.p.k * _norm_cdf(-d2) - fwd * _norm_cdf(-d1)))

    def price_plain(self, seed: int = 42) -> tuple[float, float]:
        paths = self.simulate_paths(antithetic=False, seed=seed)
        avg = np.mean(paths[:, 1:], axis=1)
        y = self._discounted_payoff(avg)
        return float(np.mean(y)), float(np.std(y) / np.sqrt(len(y)))

    def price_antithetic(self, seed: int = 42) -> tuple[float, float]:
        paths = self.simulate_paths(antithetic=True, seed=seed)
        avg = np.mean(paths[:, 1:], axis=1)
        y = self._discounted_payoff(avg)
        half = len(y) // 2
        pair = 0.5 * (y[:half] + y[half:])
        return float(np.mean(pair)), float(np.std(pair) / np.sqrt(len(pair)))

    def price_control_variate(self, seed: int = 42) -> tuple[float, float, float]:
        paths = self.simulate_paths(antithetic=True, seed=seed)
        arith = np.mean(paths[:, 1:], axis=1)
        geo = np.exp(np.mean(np.log(paths[:, 1:] + 1e-12), axis=1))

        y = self._discounted_payoff(arith)
        c = self._discounted_payoff(geo)

        b = np.cov(y, c)[0, 1] / (np.var(c) + 1e-12)
        c0 = self.geometric_closed_form()
        y_cv = y - b * (c - c0)

        price = float(np.mean(y_cv))
        se = float(np.std(y_cv) / np.sqrt(len(y_cv)))
        vr = float(1.0 - np.var(y_cv) / (np.var(y) + 1e-12))
        return price, se, vr

    def implied_vol(self, market_price: float, seed: int = 42) -> float:
        def objective(sig: float) -> float:
            p = AsianOptionParams(**{**self.p.__dict__, "sigma": sig})
            px, _ = AsianOptionEngine(p, n_paths=min(40_000, self.n_paths)).price_antithetic(seed=seed)
            return px - market_price

        lo, hi = 1e-4, 3.0
        f_lo = objective(lo)
        f_hi = objective(hi)
        if f_lo * f_hi > 0:
            return float("nan")

        for _ in range(40):
            mid = 0.5 * (lo + hi)
            f_mid = objective(mid)
            if abs(f_mid) < 1e-6:
                return float(mid)
            if f_lo * f_mid <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid
        return float(0.5 * (lo + hi))

    def greeks_fd(self, bump: float = 1e-3) -> dict[str, float]:
        base, _, _ = self.price_control_variate()

        h_s = self.p.s0 * bump
        up_s = AsianOptionEngine(AsianOptionParams(**{**self.p.__dict__, "s0": self.p.s0 + h_s}), n_paths=self.n_paths // 2)
        dn_s = AsianOptionEngine(AsianOptionParams(**{**self.p.__dict__, "s0": self.p.s0 - h_s}), n_paths=self.n_paths // 2)
        p_up, _, _ = up_s.price_control_variate(seed=7)
        p_dn, _, _ = dn_s.price_control_variate(seed=7)
        delta = (p_up - p_dn) / (2 * h_s)
        gamma = (p_up - 2 * base + p_dn) / (h_s ** 2)

        h_v = self.p.sigma * bump
        up_v = AsianOptionEngine(AsianOptionParams(**{**self.p.__dict__, "sigma": self.p.sigma + h_v}), n_paths=self.n_paths // 2)
        dn_v = AsianOptionEngine(AsianOptionParams(**{**self.p.__dict__, "sigma": self.p.sigma - h_v}), n_paths=self.n_paths // 2)
        vega = (up_v.price_control_variate(seed=9)[0] - dn_v.price_control_variate(seed=9)[0]) / (2 * h_v)

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
        }


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
