"""
project3/env.py
───────────────
Futures execution environment with Almgren-Chriss market impact.
期货执行环境，使用 Almgren-Chriss 市场冲击模型。

[OpenClaw] ExecutionEnv framework with 5-dim state space and reward design
[OpenClaw] ExecutionEnv 框架（5维状态空间，奖励设计）
[Ours]     Extended to 13-dim state (LOB depth / momentum / fill history)
[我们]     扩展到 13 维状态（LOB 深度 / 价格动量 / 成交历史）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarketImpactModel:
    """
    Almgren-Chriss dual market impact model.
    Almgren-Chriss 双重市场冲击模型。

    Temporary impact: dissipates immediately after execution
    临时冲击：执行后立即消散
      I_temp = η · |v/ADV|^β · price

    Permanent impact: shifts the mid-price forever
    永久冲击：永久改变中间价
      I_perm = γ · v/ADV · price

    η (eta):   temporary impact coefficient / 临时冲击系数
    β (beta):  impact concavity (0.5 = square root law) / 冲击凹性（0.5=平方根法则）
    γ (gamma): permanent impact coefficient / 永久冲击系数
    """
    eta:   float = 0.002
    beta:  float = 0.6
    gamma: float = 0.0001

    def temporary(self, v: float, adv: float) -> float:
        """Temporary impact as fraction of price. / 临时冲击（价格的分数）。"""
        return self.eta * np.sign(v) * (abs(v) / (adv + 1e-12)) ** self.beta

    def permanent(self, v: float, adv: float) -> float:
        """Permanent impact as fraction of price. / 永久冲击（价格的分数）。"""
        return self.gamma * v / (adv + 1e-12)


class ExecutionEnv:
    """
    Futures execution MDP environment.
    期货执行 MDP 环境。

    State s ∈ ℝ^5 (OpenClaw compatible) or ℝ^13 (our extended version):
    状态 s ∈ ℝ^5（OpenClaw 兼容）或 ℝ^13（我们的扩展版本）：

      [0]  q_remaining    fraction of quantity remaining / 剩余数量比例
      [1]  t_remaining    fraction of time remaining / 剩余时间比例
      [2]  mid/arrival    normalized mid price / 归一化中间价
      [3]  spread/mid     relative bid-ask spread / 相对价差
      [4]  order_imbalance LOB order imbalance ∈ [−1, 1] / 订单簿不平衡

      Extended state (extended=True) adds:
      扩展状态（extended=True）额外增加：
      [5]  depth_bid      normalized bid-side depth / 归一化买方深度
      [6]  depth_ask      normalized ask-side depth / 归一化卖方深度
      [7]  mom_5          5-step price momentum / 5步价格动量
      [8]  mom_20         20-step price momentum / 20步价格动量
      [9]  vol_realized   rolling 10-step realized volatility / 滚动10步已实现波动率
      [10] fill_rate      completion ratio so far / 截至当前完成比例
      [11] avg_slip_bps   average slippage in bps so far / 截至当前平均滑点（bps）
      [12] time_of_day    intraday time 0→1 (affects VWAP shape) / 日内时间（影响 VWAP 曲线形状）

    Reward: negative implementation shortfall in bps, with completion penalty.
    奖励：实施成本的负数（bps），完成不足时有额外惩罚。
    """

    def __init__(
        self,
        total_qty: float      = 10_000.0,
        horizon: int          = 78,           # 78 × 5min = 6.5h one trading day / 一个交易日
        adv: float            = 5_000_000.0,
        extended_state: bool  = False,
        seed: int             = 42,
    ) -> None:
        self.total_qty      = total_qty
        self.horizon        = horizon
        self.adv            = adv
        self.extended_state = extended_state
        self.state_dim      = 13 if extended_state else 5
        self.rng            = np.random.default_rng(seed)
        self.impact         = MarketImpactModel()
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state. / 重置环境到初始状态。"""
        self.t         = 0
        self.done_qty  = 0.0
        self.arrival   = 100.0          # arrival price (benchmark) / 到达价（基准价）
        self.mid       = self.arrival
        self.fills:    list[float] = []
        self.qty_hist: list[float] = []
        self.mid_hist: list[float] = [self.mid]
        return self._state()

    def _state(self) -> np.ndarray:
        """Construct current state observation. / 构建当前状态观测。"""
        rem_qty   = (self.total_qty - self.done_qty) / (self.total_qty + 1e-12)
        rem_t     = (self.horizon - self.t) / self.horizon
        spread    = 0.01 + 0.005 * abs(self.rng.standard_normal())
        imbalance = float(np.clip(self.rng.standard_normal() * 0.2, -1, 1))

        s = [rem_qty, rem_t, self.mid / self.arrival, spread / self.mid, imbalance]

        if self.extended_state:
            # LOB depth: exponentially distributed (heavier tail than Gaussian)
            # LOB 深度：指数分布（比高斯分布尾部更重）
            depth_bid = max(0.0, self.rng.exponential(0.3))
            depth_ask = max(0.0, self.rng.exponential(0.3))

            # Short-term and medium-term momentum signals
            # 短期和中期价格动量信号
            mom5  = (self.mid / (self.mid_hist[-5]  + 1e-12) - 1) if len(self.mid_hist) >= 5  else 0.0
            mom20 = (self.mid / (self.mid_hist[-20] + 1e-12) - 1) if len(self.mid_hist) >= 20 else 0.0

            # Realized volatility from recent mid-price changes
            # 由近期中间价变化计算的已实现波动率
            recent = np.array(self.mid_hist[-11:])
            vol_r  = float(np.std(np.diff(recent) / recent[:-1])) if len(recent) >= 2 else 0.0

            fill_rate = float(self.done_qty / self.total_qty)
            avg_slip  = float((np.mean(self.fills) - self.arrival) / self.arrival * 1e4) \
                        if self.fills else 0.0
            tod = self.t / self.horizon   # time of day normalized to [0, 1] / 日内时间归一化到 [0,1]
            s  += [depth_bid, depth_ask, mom5, mom20, vol_r, fill_rate, avg_slip, tod]

        return np.array(s, dtype=np.float32)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        """
        Execute one trading interval.
        执行一个交易时间段。

        Action: fraction of remaining quantity to trade ∈ [0, 1]
        动作：当前剩余数量中要交易的比例 ∈ [0, 1]

        Reward = − implementation_shortfall_bps × qty / total_qty
        奖励 = − 实施成本（bps）× 成交量 / 总量

        Completion penalty if order not filled by end of horizon:
        如果到期未完成订单，施加惩罚：
          penalty = 5.0 × (1 − fill_ratio)
        """
        action    = float(np.clip(action, 0.0, 1.0))
        rem       = max(self.total_qty - self.done_qty, 0.0)
        qty       = rem * action

        # Apply market impact to mid price
        # 对中间价施加市场冲击
        temp      = self.impact.temporary(qty, self.adv) * self.mid
        perm      = self.impact.permanent(qty, self.adv) * self.mid
        self.mid += perm + self.mid * 0.001 * self.rng.standard_normal()
        self.mid_hist.append(self.mid)

        # Fill price = mid + temporary impact
        # 成交价 = 中间价 + 临时冲击
        fill = self.mid + temp
        self.fills.append(fill)
        self.qty_hist.append(qty)
        self.done_qty += qty
        self.t        += 1

        done   = self.t >= self.horizon or self.done_qty >= self.total_qty * 0.999
        is_bps = (fill - self.arrival) / self.arrival * 1e4
        reward = -is_bps * qty / (self.total_qty + 1e-12)

        # Late completion penalty: simulate broker forced execution at market impact
        # 未完成惩罚：模拟 broker 在期末强制成交的冲击成本
        if done and self.done_qty < self.total_qty * 0.999:
            reward -= 5.0 * (1.0 - self.done_qty / self.total_qty)

        info = {
            "qty":        qty,
            "avg_fill":   float(np.mean(self.fills)) if self.fills else self.arrival,
            "completion": float(self.done_qty / self.total_qty),
            "is_bps":     float(is_bps),
        }
        return self._state(), float(reward), done, info
