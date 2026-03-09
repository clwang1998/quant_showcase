from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarketImpactModel:
    eta: float = 0.002
    beta: float = 0.6
    gamma: float = 0.0001

    def temporary(self, v: float, adv: float) -> float:
        return self.eta * np.sign(v) * (abs(v) / (adv + 1e-12)) ** self.beta

    def permanent(self, v: float, adv: float) -> float:
        return self.gamma * v / (adv + 1e-12)


class ExecutionEnv:
    def __init__(
        self,
        total_qty: float = 10_000.0,
        horizon: int = 78,
        adv: float = 5_000_000.0,
        seed: int = 42,
    ) -> None:
        self.total_qty = total_qty
        self.horizon = horizon
        self.adv = adv
        self.rng = np.random.default_rng(seed)
        self.impact = MarketImpactModel()
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = 0
        self.done_qty = 0.0
        self.arrival = 100.0
        self.mid = self.arrival
        self.fills: list[float] = []
        return self._state()

    def _state(self) -> np.ndarray:
        rem_qty = (self.total_qty - self.done_qty) / (self.total_qty + 1e-12)
        rem_t = (self.horizon - self.t) / self.horizon
        spread = 0.01 + 0.005 * abs(self.rng.standard_normal())
        imbalance = float(np.clip(self.rng.standard_normal() * 0.2, -1, 1))
        return np.array([rem_qty, rem_t, self.mid, spread / self.mid, imbalance], dtype=np.float32)

    def step(self, action: float) -> tuple[np.ndarray, float, bool, dict]:
        action = float(np.clip(action, 0.0, 1.0))
        rem = max(self.total_qty - self.done_qty, 0.0)
        qty = rem * action

        temp = self.impact.temporary(qty, self.adv) * self.mid
        perm = self.impact.permanent(qty, self.adv) * self.mid
        self.mid += perm + self.mid * 0.001 * self.rng.standard_normal()

        fill = self.mid + temp
        self.fills.append(fill)
        self.done_qty += qty
        self.t += 1

        done = self.t >= self.horizon or self.done_qty >= self.total_qty * 0.999
        reward = -(fill - self.arrival) * qty / (self.total_qty * self.arrival + 1e-12) * 1e4
        if done and self.done_qty < self.total_qty * 0.999:
            reward -= 5.0 * (1.0 - self.done_qty / self.total_qty)

        info = {
            "qty": qty,
            "avg_fill": float(np.mean(self.fills)) if self.fills else self.arrival,
            "completion": float(self.done_qty / self.total_qty),
        }
        return self._state(), float(reward), done, info
