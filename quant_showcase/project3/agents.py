"""
project3/agents.py
──────────────────
Three execution agents — all lightweight NumPy, no PyTorch required.
三个执行 Agent——全部为轻量 NumPy 实现，无需 PyTorch。

1. PPOStyleAgent               — GAE advantage + clipped surrogate
                                  GAE 优势估计 + clipped surrogate 目标
2. SACStyleAgent               — Stochastic policy + auto-tuned temperature α
                                  随机策略 + 自动调节温度 α
3. DecisionTransformerStylePolicy — Return-conditioned scheduling
                                    条件调度策略（基于目标完成率）

[OpenClaw] LinearPolicy skeleton and basic agent structure
[OpenClaw] LinearPolicy 骨架和 Agent 基础结构
[Ours]     PPO GAE correction, SAC log_std + tanh Jacobian, DT VWAP-U schedule
[我们]     PPO GAE 修正，SAC log_std 参数化 + tanh Jacobian 修正，DT VWAP-U 调度
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class LinearPolicy:
    """
    Linear sigmoid policy: π(s) = σ(w^T s + b)
    线性 sigmoid 策略：π(s) = σ(w^T s + b)

    Output ∈ (0, 1) interpreted as fraction of remaining qty to trade.
    输出 ∈ (0, 1)，解释为当前剩余数量中要交易的比例。
    """
    def __init__(self, state_dim: int, seed: int = 42) -> None:
        rng    = np.random.default_rng(seed)
        # Small init scale prevents early saturation of sigmoid
        # 小初始化尺度防止 sigmoid 早期饱和
        self.w = rng.standard_normal(state_dim).astype(np.float32) * 0.05
        self.b = 0.0

    def act(self, state: np.ndarray) -> float:
        z = float(np.dot(state, self.w) + self.b)
        return float(1.0 / (1.0 + np.exp(-z)))


# ── 1. PPOStyleAgent ──────────────────────────────────────────────────────────

@dataclass
class PPOStyleAgent:
    """
    PPO key components (lightweight NumPy implementation).
    PPO 核心要素（NumPy 轻量实现）。

    Advantage estimation: GAE-λ
    优势估计：GAE-λ
      δ_t  = r_t + γ V(s_{t+1}) − V(s_t)        (TD residual / TD 残差)
      A_t  = Σ_{k≥0} (γλ)^k · δ_{t+k}           (weighted sum / 加权求和)

      λ=0 → pure TD (low variance, high bias)    λ=0 → 纯 TD（低方差，高偏差）
      λ=1 → pure MC (high variance, zero bias)   λ=1 → 纯 MC（高方差，零偏差）
      λ=0.95 balances both                        λ=0.95 在两者之间折中

    Policy update: REINFORCE with GAE advantage (clipped surrogate approximated)
    策略更新：带 GAE 优势的 REINFORCE（近似 clipped surrogate）

    Entropy bonus: encourages exploration by penalizing deterministic policies
    熵奖励：通过惩罚确定性策略来鼓励探索
    """
    state_dim:    int
    lr:           float = 0.02
    gamma:        float = 0.99
    gae_lambda:   float = 0.95
    clip_eps:     float = 0.2
    entropy_coef: float = 0.01
    seed:         int   = 42

    def __post_init__(self) -> None:
        self.policy = LinearPolicy(self.state_dim, self.seed)

    def _gae(self, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        Compute GAE-λ advantages.
        计算 GAE-λ 优势。

        Runs backwards through trajectory for efficiency.
        为了效率，从轨迹末端向前递推。
        """
        T   = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            v_next = values[t + 1] if t + 1 < len(values) else 0.0
            # TD residual / TD 残差
            delta  = rewards[t] + self.gamma * v_next - values[t]
            # GAE recursive formula / GAE 递推公式
            gae    = delta + self.gamma * self.gae_lambda * gae
            adv[t] = gae
        return adv

    def act(self, state: np.ndarray, deterministic: bool = True) -> float:
        """
        Select action with minimum-rate constraint.
        在最低执行速率约束下选择动作。

        The constraint ensures the agent cannot fall arbitrarily behind schedule:
        该约束确保 Agent 不会过度落后于进度：
          min_rate = max(schedule_rate, qty_needed_per_step)
        """
        p        = self.policy.act(state)
        rem_qty  = float(state[0])
        rem_t    = max(float(state[1]), 1e-6)
        steps_l  = max(1, int(np.ceil(rem_t * 78.0)))
        min_rate = min(1.0, max(1.5 / steps_l, rem_qty / steps_l))
        if deterministic:
            return float(np.clip(max(p, min_rate), 0.0, 1.0))
        return float(np.clip(max(np.random.normal(p, 0.05), min_rate), 0.0, 1.0))

    def update(self, states: np.ndarray, actions: np.ndarray,
               rewards: np.ndarray) -> None:
        """
        Policy gradient update with GAE advantages.
        带 GAE 优势的策略梯度更新。

        Gradient of log π(a|s) for Bernoulli policy:
        Bernoulli 策略的 log π(a|s) 梯度：
          ∇_w log π = (a − π(s)) · s

        Gradient is then weighted by normalized advantage.
        梯度再乘以归一化优势。
        """
        T       = len(rewards)
        values  = np.zeros(T + 1)
        g       = 0.0
        returns = np.zeros(T)
        for t in reversed(range(T)):
            g          = rewards[t] + self.gamma * g
            returns[t] = g
        # Use Monte Carlo returns as value baseline
        # 用 MC 累计奖励作为 value baseline
        values[:T] = returns
        adv = self._gae(rewards, values)
        # Normalize advantages: zero mean, unit variance
        # 归一化优势：零均值，单位方差
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        probs     = np.array([self.policy.act(s) for s in states])
        grad_logp = (actions - probs)[:, None] * states
        grad      = np.mean(grad_logp * adv[:, None], axis=0)

        # Entropy bonus gradient: d/dw H(π) = d/dw [−π log π − (1−π)log(1−π)]
        # 熵奖励梯度
        grad += self.entropy_coef * np.mean((0.5 - probs)[:, None] * states, axis=0)
        self.policy.w += self.lr * grad


# ── 2. SACStyleAgent ──────────────────────────────────────────────────────────

@dataclass
class SACStyleAgent:
    """
    SAC key components.
    SAC 核心要素。

    Stochastic policy with learnable log_std:
    带可学习 log_std 的随机策略：
      a = tanh(μ(s) + σ(s) · ε),  ε ~ N(0,1)
    Output squashed to (0, 1) via tanh shift: 0.5*(tanh(·)+1)
    通过 tanh 变换将输出压缩到 (0, 1)

    Entropy-regularized objective:
    带熵正则化的目标：
      J(π) = E[Q(s,a) − α · log π(a|s)]

    Auto-tuned temperature α:
    自动调节温度 α：
      α ← α · exp( lr_α · (−E[log π(a|s)] − H_target) )
      H_target = target_entropy_ratio × dim(A)
      Forces average entropy to match target — prevents entropy collapse.
      强制平均熵与目标匹配，防止策略熵崩溃（过早收敛到确定性）。

    tanh Jacobian correction (our addition):
    tanh Jacobian 修正（我们的改进）：
      log π(a|s) = log π_unc(a') − Σ_i log(1 − tanh²(a'_i))
    Without this, entropy is underestimated → α overheats → unstable training.
    没有这一项，熵会被低估 → α 过热 → 训练不稳定。
    """
    state_dim:            int
    alpha:                float = 0.2
    lr:                   float = 0.01
    target_entropy_ratio: float = -1.0   # target_entropy = ratio × action_dim
    seed:                 int   = 42

    def __post_init__(self) -> None:
        self.policy    = LinearPolicy(self.state_dim, self.seed + 7)
        # log_std parameterization: more numerically stable than std directly
        # log_std 参数化：比直接参数化 std 更稳定
        self.log_std   = float(np.log(0.08))
        self.log_alpha = float(np.log(self.alpha))
        self.target_entropy = self.target_entropy_ratio   # scalar action space

    def act(self, state: np.ndarray, deterministic: bool = True) -> float:
        mu  = self.policy.act(state)
        if deterministic:
            return mu
        std = float(np.exp(self.log_std))
        raw = np.random.normal(mu, std)
        # Squash raw action to (0, 1) via shifted tanh
        # 通过移位 tanh 将原始动作压缩到 (0, 1)
        return float(0.5 * (np.tanh(raw) + 1.0))

    def _log_prob(self, state: np.ndarray, action: float) -> float:
        """
        Log probability with tanh Jacobian correction.
        带 tanh Jacobian 修正的对数概率。

        For action a = 0.5*(tanh(a') + 1), the change-of-variables formula gives:
        对于 a = 0.5*(tanh(a') + 1)，变量替换公式给出：
          log p(a) = log p(a') − log |da/da'|
                   = log p_gauss(a') − log(1 − tanh²(a'))
                   (the 0.5 factor in squashing adjusts to: log(1 − (2a−1)²))
        """
        mu  = self.policy.act(state)
        std = float(np.exp(self.log_std))
        a   = 2 * action - 1  # map (0,1) → (−1,1) for tanh space
        a   = np.clip(a, -0.9999, 0.9999)
        raw = np.arctanh(a)
        # Gaussian log-prob in pre-squash space
        # 压缩前空间的高斯对数概率
        log_p  = -0.5 * ((raw - mu) / (std + 1e-8)) ** 2 - np.log(std + 1e-8)
        # Jacobian correction: subtract log|det(d tanh/d raw)|
        # Jacobian 修正：减去 log|det(d tanh/d raw)|
        log_p -= np.log(1 - a ** 2 + 1e-8)
        return float(log_p)

    def update_alpha(self, log_probs: np.ndarray) -> None:
        """
        Auto-tune temperature α based on current policy entropy.
        根据当前策略熵自动调节温度 α。

        If entropy < H_target: increase α to encourage more exploration.
        如果熵 < H_target：增大 α，鼓励更多探索。
        If entropy > H_target: decrease α to allow more exploitation.
        如果熵 > H_target：减小 α，允许更多利用。
        """
        # entropy_diff > 0 means current entropy below target → need more α
        # entropy_diff > 0 表示当前熵低于目标 → 需要更大的 α
        entropy_diff   = -np.mean(log_probs) - self.target_entropy
        self.log_alpha += 0.005 * entropy_diff
        # Clamp log_alpha to prevent extreme values
        # 截断 log_alpha，防止极端值
        self.alpha     = float(np.exp(np.clip(self.log_alpha, -5, 2)))


# ── 3. DecisionTransformerStylePolicy ────────────────────────────────────────

@dataclass
class DecisionTransformerStylePolicy:
    """
    Decision Transformer core idea: condition on target return R̂.
    Decision Transformer 核心思想：以目标累计奖励 R̂ 为条件生成动作。

    Full DT uses a causal transformer over (R̂, s, a) triplets.
    完整 DT 在 (R̂, s, a) 三元组序列上使用因果 Transformer。

    Here we use an analytic VWAP-style schedule as a lightweight proxy
    that captures the same return-conditioning idea without GPT weights.
    这里用解析 VWAP 调度作为轻量代理，无需 GPT 权重，
    体现相同的 return-conditioning 思路。

    Three schedule modes / 三种调度模式：
      linear     : TWAP — uniform execution / 匀速执行
      front_loaded: concentrate execution at open (high liquidity window)
                    前置执行（适合流动性集中在开盘的场景）
      vwap_u     : U-shaped intraday volume profile (high at open and close)
                   U 型日内成交量分布（开盘和收盘量大，盘中量小）
    """
    context_len: int = 20

    def act(
        self,
        rem_qty:            float,
        rem_t:              float,
        desired_completion: float = 1.0,
        schedule:           str   = "linear",
    ) -> float:
        if rem_t <= 1e-6:
            return 1.0   # time's up — trade everything remaining / 时间到——全部执行

        if schedule == "linear":
            # TWAP: uniform pacing / TWAP：匀速执行
            rate = max((desired_completion - (1.0 - rem_qty)) / rem_t, 0.0)

        elif schedule == "front_loaded":
            # Execute 50% faster early on to capture opening liquidity
            # 开盘阶段执行速度加快50%，捕获开盘流动性
            rate = max((desired_completion - (1.0 - rem_qty)) / rem_t * 1.5, 0.0)

        else:  # vwap_u
            # U-shaped volume weight: heavier at open (tod≈0) and close (tod≈1)
            # U 型成交量权重：开盘（tod≈0）和收盘（tod≈1）处权重更大
            tod  = 1.0 - rem_t
            u    = 1.0 + 0.5 * ((2 * tod - 1) ** 2)   # parabola with min at midday
            rate = max((desired_completion - (1.0 - rem_qty)) / rem_t * u / 1.25, 0.0)

        return float(np.clip(rate, 0.0, 1.0))
