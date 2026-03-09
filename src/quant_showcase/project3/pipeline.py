from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_showcase.core.types import ExecutionEvaluation, PipelineReport
from .agents import DecisionTransformerStylePolicy, PPOStyleAgent, SACStyleAgent
from .env import ExecutionEnv


def _evaluate_policy(name: str, action_fn, n_episodes: int = 200, seed: int = 42) -> ExecutionEvaluation:
    rng = np.random.default_rng(seed)
    is_list = []
    completions = []

    for _ in range(n_episodes):
        env = ExecutionEnv(seed=int(rng.integers(1_000_000)))
        state = env.reset()
        done = False
        while not done:
            action = action_fn(state)
            state, _, done, info = env.step(action)

        if env.fills:
            avg_fill = float(np.mean(env.fills))
            is_bps = (avg_fill - env.arrival) / env.arrival * 1e4
            is_list.append(is_bps)
            completions.append(info["completion"])

    return ExecutionEvaluation(
        name=name,
        mean_is_bps=float(np.mean(is_list)),
        p95_is_bps=float(np.percentile(is_list, 95)),
        completion_rate=float(np.mean(completions)),
    )


RESEARCH_NARRATIVE = {
    "problem": "Intraday execution is a sequential control problem under impact, uncertainty, and completion constraints.",
    "hypothesis": "Combining online policy improvement and offline sequence-style scheduling improves execution quality.",
    "method_stack": [
        "Market simulator with Almgren-Chriss style impact",
        "PPO-style online policy improvement loop",
        "SAC-style stochastic actor baseline",
        "Decision Transformer-style conditional schedule policy",
        "Implementation shortfall based evaluation",
    ],
}


@dataclass(frozen=True)
class Project3Config:
    seed: int = 42
    state_dim: int = 5
    context_len: int = 20
    train_episodes: int = 60
    eval_episodes: int = 200


def run(seed: int = 42, config: Project3Config | None = None) -> PipelineReport:
    cfg = config or Project3Config(seed=seed)
    ppo = PPOStyleAgent(state_dim=cfg.state_dim, seed=cfg.seed)
    sac = SACStyleAgent(state_dim=cfg.state_dim, seed=cfg.seed)
    dtp = DecisionTransformerStylePolicy(context_len=cfg.context_len)

    # Lightweight policy-improvement loop for PPO-style model.
    train_env = ExecutionEnv(seed=cfg.seed)
    for _ in range(cfg.train_episodes):
        states = []
        actions = []
        rewards = []
        s = train_env.reset()
        done = False
        while not done:
            a = ppo.act(s, deterministic=False)
            ns, r, done, _ = train_env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = ns
        adv = np.array(rewards) - np.mean(rewards)
        ppo.update(np.array(states), np.array(actions), adv)

    eval_ppo = _evaluate_policy(
        "ppo_style",
        lambda s: ppo.act(s, deterministic=True),
        n_episodes=cfg.eval_episodes,
        seed=cfg.seed,
    )
    eval_sac = _evaluate_policy(
        "sac_style",
        lambda s: sac.act(s, deterministic=True),
        n_episodes=cfg.eval_episodes,
        seed=cfg.seed + 1,
    )
    eval_dt = _evaluate_policy(
        "dt_style",
        lambda s: dtp.act(rem_qty=float(s[0]), rem_t=float(s[1]), desired_completion=1.0),
        n_episodes=cfg.eval_episodes,
        seed=cfg.seed + 2,
    )

    metrics = {
        "ppo_mean_is_bps": eval_ppo.mean_is_bps,
        "sac_mean_is_bps": eval_sac.mean_is_bps,
        "dt_mean_is_bps": eval_dt.mean_is_bps,
        "ppo_completion": eval_ppo.completion_rate,
        "sac_completion": eval_sac.completion_rate,
        "dt_completion": eval_dt.completion_rate,
    }
    return PipelineReport(name="project3_rl_execution", metrics=metrics)
