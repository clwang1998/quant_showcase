"""
project2/pricing.py
───────────────────
Asian option pricing engine.
亚式期权定价引擎。

[OpenClaw] AsianOptionEngine framework: simulation, Plain/Antithetic/CV, Greeks FD, IV
[OpenClaw] AsianOptionEngine 框架：模拟、Plain / Antithetic / CV、Greeks FD、IV 反推
[Ours]     Heston stochastic volatility paths (Euler-Milstein)
[我们]     Heston 随机波动率路径（Euler-Milstein 离散化）
[Ours]     Pathwise delta estimator (lower variance than pure FD)
[我们]     Pathwise delta estimator（比有限差分方差更低）
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass
class AsianOptionParams:
    s0:          float = 100.0
    k:           float = 100.0
    t:           float = 1.0
    r:           float = 0.03
    q:           float = 0.00
    sigma:       float = 0.25
    n_steps:     int   = 252
    option_type: str   = "call"


class AsianOptionEngine:
    """
    Asian option Monte Carlo pricing engine with variance reduction.
    亚式期权 Monte Carlo 定价引擎，包含方差缩减方法。

    Variance reduction methods (in order of effectiveness):
    方差缩减方法（效果递增）：
      Plain MC         → baseline / 基准
      Antithetic       → exploit Brownian motion symmetry: pair (Z, −Z)
                         利用布朗运动对称性，配对路径 (Z, −Z)
      Control Variate  → use geometric Asian closed-form as control mean
                         用几何均值亚式（有解析解）作控制变量
                         b* = Cov(f_arith, f_geo) / Var(f_geo)
                         Reduces SE by ~99% in practice / 实测 SE 降低约 99%
    """

    def __init__(self, params: AsianOptionParams, n_paths: int = 80_000) -> None:
        self.p       = params
        self.n_paths = n_paths

    # ── Path Simulation ───────────────────────────────────────────────────────
    # ── 路径模拟 ──────────────────────────────────────────────────────────────

    def simulate_paths(self, antithetic: bool = False, seed: int = 42) -> np.ndarray:
        """
        Simulate GBM paths under risk-neutral measure Q.
        在风险中性测度 Q 下模拟 GBM 路径。

        Returns / 返回：shape (n_paths, n_steps + 1)
        """
        rng   = np.random.default_rng(seed)
        dt    = self.p.t / self.p.n_steps
        # Risk-neutral drift under Q: (r - q - σ²/2)dt
        # 风险中性漂移项：(r - q - σ²/2)dt
        drift = (self.p.r - self.p.q - 0.5 * self.p.sigma ** 2) * dt
        vol   = self.p.sigma * np.sqrt(dt)

        if antithetic:
            # Generate half the paths then mirror: (Z, −Z) pairs
            # 生成一半路径再镜像：(Z, −Z) 配对，消除奇数阶矩的方差
            half = self.n_paths // 2
            z    = rng.standard_normal((half, self.p.n_steps))
            z    = np.vstack([z, -z])
        else:
            z    = rng.standard_normal((self.n_paths, self.p.n_steps))

        log_s = np.log(self.p.s0) + np.cumsum(drift + vol * z, axis=1)
        s     = np.hstack([np.full((z.shape[0], 1), self.p.s0), np.exp(log_s)])
        return s

    def simulate_heston_paths(
        self,
        v0: float    = 0.0625,   # initial variance σ₀² = 0.25² / 初始方差
        kappa: float = 2.0,      # mean-reversion speed / 均值回归速度
        theta: float = 0.0625,   # long-run variance / 长期均值方差
        xi: float    = 0.3,      # vol-of-vol / 波动率的波动率
        rho: float   = -0.7,     # correlation dW_S ↔ dW_v / 布朗运动相关系数
        seed: int    = 42,
    ) -> np.ndarray:
        """
        Heston stochastic volatility paths via Euler-Milstein discretization.
        用 Euler-Milstein 离散化模拟 Heston 随机波动率路径。

        SDEs / 随机微分方程：
          dS = (r−q)S dt + √v · S · dW_S
          dv = κ(θ−v) dt + ξ√v · dW_v,  corr(dW_S, dW_v) = ρ

        Milstein correction term for v (second-order accuracy):
        对 v 的 Milstein 修正项（二阶精度）：
          Δv_Milstein = ¼ξ² dt (Z² − 1)
        This reduces discretization bias when v approaches 0, where
        Euler scheme can produce negative variance values.
        当 v 接近 0 时，纯 Euler 离散化会产生负方差值，Milstein 修正可避免这一问题。
        """
        rng    = np.random.default_rng(seed)
        dt     = self.p.t / self.p.n_steps
        rho2   = np.sqrt(max(1 - rho ** 2, 0))
        N      = self.n_paths

        S = np.full((N, self.p.n_steps + 1), self.p.s0)
        v = np.full((N, self.p.n_steps + 1), v0)

        for t in range(self.p.n_steps):
            Z1 = rng.standard_normal(N)
            # Correlated Brownian increments via Cholesky decomposition
            # 通过 Cholesky 分解生成相关布朗运动增量
            Z2 = rho * Z1 + rho2 * rng.standard_normal(N)
            vp = np.maximum(v[:, t], 0.0)   # floor at 0 to avoid sqrt of negative
            sv = np.sqrt(vp)                 # √v 进行后续计算

            # Milstein step for variance process
            # 对方差过程做 Milstein 步
            v[:, t + 1] = np.maximum(
                v[:, t]
                + kappa * (theta - vp) * dt
                + xi * sv * np.sqrt(dt) * Z2
                + 0.25 * xi ** 2 * dt * (Z2 ** 2 - 1),   # Milstein correction / Milstein 修正
                0.0,
            )
            # Log-Euler for stock price (exact in GBM limit)
            # 对股价做 Log-Euler（在 GBM 极限下精确）
            S[:, t + 1] = S[:, t] * np.exp(
                (self.p.r - self.p.q - 0.5 * vp) * dt + sv * np.sqrt(dt) * Z1
            )
        return S

    # ── Payoff Utilities ──────────────────────────────────────────────────────
    # ── Payoff 工具 ───────────────────────────────────────────────────────────

    def _disc_payoff(self, avg: np.ndarray) -> np.ndarray:
        """Discounted arithmetic average payoff. / 折现算术平均 payoff。"""
        if self.p.option_type == "call":
            return np.exp(-self.p.r * self.p.t) * np.maximum(avg - self.p.k, 0.0)
        return np.exp(-self.p.r * self.p.t) * np.maximum(self.p.k - avg, 0.0)

    # ── Geometric Asian Closed-Form (Control Variate Anchor) ─────────────────
    # ── 几何亚式解析解（控制变量的精确均值）──────────────────────────────────

    def geometric_closed_form(self) -> float:
        """
        Closed-form price for geometric average Asian option (Kemna & Vorst, 1990).
        几何均值亚式期权的解析解（Kemna & Vorst, 1990）。

        The key insight: the geometric average of log-normal variables is itself
        log-normal, allowing a Black-Scholes-like formula with adjusted parameters.
        核心思路：对数正态变量的几何均值仍是对数正态分布，
        因此可用调整参数后的 Black-Scholes 公式计算。
        """
        m      = self.p.n_steps
        # Adjusted volatility: accounts for averaging effect on variance
        # 调整后的波动率：考虑了平均过程对方差的压缩效果
        sig_g  = self.p.sigma * np.sqrt((2 * m + 1) / (6 * (m + 1)))
        mu_g   = 0.5 * (self.p.r - self.p.q - 0.5 * self.p.sigma ** 2) + 0.5 * sig_g ** 2
        d1     = (np.log(self.p.s0 / self.p.k) + (mu_g + 0.5 * sig_g ** 2) * self.p.t) \
                 / (sig_g * np.sqrt(self.p.t) + 1e-12)
        d2     = d1 - sig_g * np.sqrt(self.p.t)
        disc   = np.exp(-self.p.r * self.p.t)
        fwd    = self.p.s0 * np.exp(mu_g * self.p.t)
        if self.p.option_type == "call":
            return float(disc * (fwd * _ncdf(d1) - self.p.k * _ncdf(d2)))
        return float(disc * (self.p.k * _ncdf(-d2) - fwd * _ncdf(-d1)))

    # ── Pricing Methods ───────────────────────────────────────────────────────
    # ── 定价方法 ──────────────────────────────────────────────────────────────

    def price_plain(self, seed: int = 42) -> tuple[float, float]:
        """Plain Monte Carlo. Returns (price, standard_error). / 纯 MC，返回 (价格, 标准误)。"""
        paths = self.simulate_paths(antithetic=False, seed=seed)
        y     = self._disc_payoff(paths[:, 1:].mean(axis=1))
        return float(y.mean()), float(y.std() / np.sqrt(len(y)))

    def price_antithetic(self, seed: int = 42) -> tuple[float, float]:
        """
        Antithetic variates: pair (Z, −Z), average each pair.
        对偶变量：配对 (Z, −Z)，每对取平均值。

        Var reduction: roughly 50% when path payoff is monotone in Z.
        方差缩减：当路径 payoff 对 Z 单调时，理论上可缩减约 50%。
        """
        paths = self.simulate_paths(antithetic=True, seed=seed)
        y     = self._disc_payoff(paths[:, 1:].mean(axis=1))
        half  = len(y) // 2
        pair  = 0.5 * (y[:half] + y[half:])   # paired average / 配对均值
        return float(pair.mean()), float(pair.std() / np.sqrt(len(pair)))

    def price_control_variate(self, seed: int = 42) -> tuple[float, float, float]:
        """
        Antithetic + control variate (combined method).
        对偶变量 + 控制变量（联合方法）。

        Control variate formula / 控制变量公式：
          b*    = Cov(f_arith, f_geo) / Var(f_geo)    (optimal coefficient / 最优系数)
          Y_cv  = f_arith − b*(f_geo − E[f_geo])

        E[f_geo] is exact (closed-form), so the correction is unbiased.
        E[f_geo] 是精确值（解析解），所以修正项是无偏的。

        Returns / 返回：(price, std_error, variance_reduction_ratio)
        """
        paths  = self.simulate_paths(antithetic=True, seed=seed)
        arith  = paths[:, 1:].mean(axis=1)
        geo    = np.exp(np.log(paths[:, 1:] + 1e-12).mean(axis=1))

        y = self._disc_payoff(arith)
        c = self._disc_payoff(geo)

        # Pair antithetic samples before computing CV coefficient
        # 先配对对偶样本，再计算控制变量系数
        half   = len(y) // 2
        yp     = 0.5 * (y[:half] + y[half:])
        cp     = 0.5 * (c[:half] + c[half:])

        # Optimal b* via OLS: minimizes Var(Y_cv)
        # 最优系数 b*：通过 OLS 最小化 Var(Y_cv)
        b      = np.cov(yp, cp)[0, 1] / (np.var(cp) + 1e-12)
        c0     = self.geometric_closed_form()   # exact control mean / 精确控制均值
        y_cv   = yp - b * (cp - c0)

        price  = float(y_cv.mean())
        se     = float(y_cv.std() / np.sqrt(len(y_cv)))
        # Variance reduction ratio: 1 − Var(Y_cv)/Var(Y_plain)
        # 方差缩减比：1 − Var(Y_cv)/Var(Y_plain)
        vr     = float(1.0 - np.var(y_cv) / (np.var(y) + 1e-12))
        return price, se, vr

    def price_heston(self, seed: int = 42, **heston_kwargs) -> tuple[float, float]:
        """Asian option price under Heston stochastic vol model. / Heston 模型下亚式期权价格。"""
        paths = self.simulate_heston_paths(seed=seed, **heston_kwargs)
        y     = self._disc_payoff(paths[:, 1:].mean(axis=1))
        return float(y.mean()), float(y.std() / np.sqrt(len(y)))

    # ── Greeks ────────────────────────────────────────────────────────────────

    def delta_pathwise(self, seed: int = 42) -> float:
        """
        Pathwise delta estimator — lower variance than finite difference.
        Pathwise delta estimator——比有限差分方差更低。

        Derivation / 推导：
          Δ = dV/dS₀ = E[ d/dS₀ { e^{-rT} (A_T − K)⁺ } ]
            = E[ e^{-rT} · 1{A_T > K} · A_T / S₀ ]    (chain rule + indicator)
                                                          （链式法则 + 指示函数）

        Valid because the Asian payoff is Lipschitz continuous in S₀.
        成立条件：亚式 payoff 对 S₀ 是 Lipschitz 连续的。
        Finite-difference delta requires two sets of paths; pathwise needs only one.
        有限差分 delta 需要两组路径，pathwise 只需一组。
        """
        paths = self.simulate_paths(antithetic=True, seed=seed)
        avg   = paths[:, 1:].mean(axis=1)
        disc  = np.exp(-self.p.r * self.p.t)
        if self.p.option_type == "call":
            delta = disc * (avg > self.p.k) * (avg / (self.p.s0 + 1e-12))
        else:
            delta = -disc * (avg < self.p.k) * (avg / (self.p.s0 + 1e-12))
        return float(delta.mean())

    def greeks_fd(self, bump: float = 1e-3) -> dict[str, float]:
        """
        Finite-difference Greeks using central differences (bump = 0.1%).
        有限差分 Greeks，使用中心差分（bump = 0.1%）。

        Delta and Gamma: bump S₀ up/down
        Delta 和 Gamma：对 S₀ 做正负 bump

        Vega: bump σ up/down
        Vega：对 σ 做正负 bump

        Central difference reduces bias from O(h) to O(h²).
        中心差分将偏差从 O(h) 降低到 O(h²)。
        """
        base, _, _ = self.price_control_variate()
        h_s = self.p.s0 * bump

        def _price_cv(s0=None, sigma=None):
            p = AsianOptionParams(**{**self.p.__dict__,
                                     **({'s0': s0}     if s0    else {}),
                                     **({'sigma': sigma} if sigma else {})})
            return AsianOptionEngine(p, self.n_paths // 4).price_control_variate(seed=7)[0]

        p_up  = _price_cv(s0=self.p.s0 + h_s)
        p_dn  = _price_cv(s0=self.p.s0 - h_s)
        # Delta: (V(S+h) − V(S−h)) / 2h
        delta = (p_up - p_dn) / (2 * h_s)
        # Gamma: (V(S+h) − 2V(S) + V(S−h)) / h²
        gamma = (p_up - 2 * base + p_dn) / h_s ** 2

        h_v  = self.p.sigma * bump
        # Vega: (V(σ+h) − V(σ−h)) / 2h
        vega = (_price_cv(sigma=self.p.sigma + h_v)
                - _price_cv(sigma=self.p.sigma - h_v)) / (2 * h_v)

        return {"delta": float(delta), "gamma": float(gamma),
                "vega": float(vega), "delta_pathwise": self.delta_pathwise(seed=42)}

    # ── Implied Volatility (Brent Bisection) ─────────────────────────────────
    # ── 隐含波动率反推（Brent 二分法）────────────────────────────────────────

    def implied_vol(self, market_price: float, seed: int = 42) -> float:
        """
        Invert MC price to find implied vol via Brent bisection.
        通过 Brent 二分法反推隐含波动率。

        50 iterations give convergence to ~1e-6 in σ under typical conditions.
        50 次迭代在常规情况下可收敛至 σ 误差约 1e-6。

        Returns NaN if market price is outside the no-arbitrage bounds.
        如果市场价格超出无套利范围则返回 NaN。
        """
        def obj(sig: float) -> float:
            p  = AsianOptionParams(**{**self.p.__dict__, "sigma": sig})
            px = AsianOptionEngine(p, min(30_000, self.n_paths)).price_antithetic(seed=seed)[0]
            return px - market_price

        lo, hi = 1e-4, 3.0
        f_lo, f_hi = obj(lo), obj(hi)
        if f_lo * f_hi > 0:
            # Price not achievable in [lo, hi] vol range — arbitrage or bad input
            # 在 [lo, hi] 波动率范围内无法达到该价格——可能存在套利或输入有误
            return float("nan")
        for _ in range(50):
            mid   = 0.5 * (lo + hi)
            f_mid = obj(mid)
            if abs(f_mid) < 1e-6:
                return float(mid)
            if f_lo * f_mid <= 0:
                hi, f_hi = mid, f_mid
            else:
                lo, f_lo = mid, f_mid
        return float(0.5 * (lo + hi))


def _ncdf(x: float) -> float:
    """Standard normal CDF via error function. / 用误差函数计算标准正态 CDF。"""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))
