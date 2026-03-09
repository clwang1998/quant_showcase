from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class LinearPolicy:
    def __init__(self, state_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w = rng.standard_normal(state_dim) * 0.05
        self.b = 0.0

    def act(self, state: np.ndarray) -> float:
        z = float(np.dot(state, self.w) + self.b)
        return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class PPOStyleAgent:
    state_dim: int
    lr: float = 0.02
    entropy_coef: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        self.policy = LinearPolicy(self.state_dim, self.seed)

    def act(self, state: np.ndarray, deterministic: bool = True) -> float:
        p = self.policy.act(state)
        rem_qty = float(state[0])
        rem_t = max(float(state[1]), 1e-6)
        steps_left = max(1, int(np.ceil(rem_t * 78.0)))
        schedule_rate = min(1.0, 1.5 / steps_left)
        min_rate = min(1.0, max(schedule_rate, rem_qty / steps_left))
        if deterministic:
            return float(np.clip(max(p, min_rate), 0.0, 1.0))
        sampled = float(np.random.normal(p, 0.05))
        return float(np.clip(max(sampled, min_rate), 0.0, 1.0))

    def update(self, states: np.ndarray, actions: np.ndarray, advantages: np.ndarray) -> None:
        probs = np.array([self.policy.act(s) for s in states])
        grad_logp = (actions - probs)[:, None] * states
        grad = np.mean(grad_logp * advantages[:, None], axis=0)
        self.policy.w += self.lr * grad


@dataclass
class SACStyleAgent:
    state_dim: int
    alpha: float = 0.2
    lr: float = 0.01
    seed: int = 42

    def __post_init__(self) -> None:
        self.policy = LinearPolicy(self.state_dim, self.seed + 7)

    def act(self, state: np.ndarray, deterministic: bool = True) -> float:
        p = self.policy.act(state)
        if deterministic:
            return p
        noise = np.random.normal(0.0, 0.08)
        return float(np.clip(p + noise, 0.0, 1.0))


@dataclass
class DecisionTransformerStylePolicy:
    context_len: int = 20

    def act(self, rem_qty: float, rem_t: float, desired_completion: float = 1.0) -> float:
        if rem_t <= 1e-6:
            return 1.0
        target_rate = max((desired_completion - (1.0 - rem_qty)) / rem_t, 0.0)
        return float(np.clip(target_rate, 0.0, 1.0))
