"""
project3/pipeline.py
────────────────────
三个 Agent 的评估 pipeline，对比 IS bps 和完成率。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quant_showcase.core.types import ExecutionEvaluation, PipelineReport
from .agents import DecisionTransformerStylePolicy, PPOStyleAgent, SACStyleAgent
from .env import ExecutionEnv


RESEARCH_NARRATIVE = {
    "problem": (
        "Intraday execution is a sequential control problem under market impact, "
        "uncertainty, and hard completion constraints."
    ),
    "hypothesis": (
        "Combining online policy improvement (PPO) and offline sequence scheduling (DT) "
        "reduces implementation shortfall vs TWAP/VWAP baselines."
    ),
    "method_stack": [
        "Almgren-Chriss market impact: temporary η|v/ADV|^β + permanent γv/ADV",
        "MDP state: 5-dim (compatible) or 13-dim (extended: depth/momentum/fill history)",
        "PPO: GAE-λ advantage + clipped surrogate L_CLIP + entropy bonus",
        "SAC: tanh-squashed Gaussian + auto-tuned temperature α",
        "Decision Transformer: return-conditioned TWAP/VWAP schedule",
        "Evaluation: implementation shortfall in bps, completion rate",
    ],
}


def _evaluate(name: str, action_fn, n_episodes: int = 200,
              seed: int = 42, extended: bool = False) -> ExecutionEvaluation:
    rng      = np.random.default_rng(seed)
    is_list  = []
    comps    = []
    for _ in range(n_episodes):
        env   = ExecutionEnv(seed=int(rng.integers(1_000_000)),
                             extended_state=extended)
        s     = env.reset()
        done  = False
        while not done:
            s, _, done, info = env.step(action_fn(s))
        if env.fills:
            avg_fill = float(np.mean(env.fills))
            is_bps   = (avg_fill - env.arrival) / env.arrival * 1e4
            is_list.append(is_bps)
            comps.append(info["completion"])

    return ExecutionEvaluation(
        name=name,
        mean_is_bps=float(np.mean(is_list)),
        p95_is_bps=float(np.percentile(is_list, 95)),
        completion_rate=float(np.mean(comps)),
    )


@dataclass(frozen=True)
class Project3Config:
    seed:            int  = 42
    state_dim:       int  = 5
    context_len:     int  = 20
    train_episodes:  int  = 60
    eval_episodes:   int  = 200
    extended_state:  bool = False


def run(seed: int = 42, config: Project3Config | None = None) -> PipelineReport:
    cfg = config or Project3Config(seed=seed)

    ppo = PPOStyleAgent(state_dim=cfg.state_dim, seed=cfg.seed)
    sac = SACStyleAgent(state_dim=cfg.state_dim, seed=cfg.seed)
    dtp = DecisionTransformerStylePolicy(context_len=cfg.context_len)

    # PPO 训练循环（GAE 优势）
    train_env = ExecutionEnv(seed=cfg.seed, extended_state=cfg.extended_state)
    for ep in range(cfg.train_episodes):
        states, actions, rewards = [], [], []
        s    = train_env.reset()
        done = False
        while not done:
            a = ppo.act(s, deterministic=False)
            ns, r, done, _ = train_env.step(a)
            states.append(s); actions.append(a); rewards.append(r)
            s = ns
        ppo.update(np.array(states), np.array(actions), np.array(rewards))

        # SAC 温度自适应（用最近 episode 的 log_prob 估计熵）
        log_probs = np.array([sac._log_prob(s, a)
                              for s, a in zip(states, actions)])
        sac.update_alpha(log_probs)

    eval_ppo = _evaluate("ppo_gae",    lambda s: ppo.act(s, True),
                         cfg.eval_episodes, cfg.seed,     cfg.extended_state)
    eval_sac = _evaluate("sac_auto_α", lambda s: sac.act(s, True),
                         cfg.eval_episodes, cfg.seed + 1, cfg.extended_state)
    eval_twap = _evaluate("dt_twap",
                          lambda s: dtp.act(float(s[0]), float(s[1]), schedule="linear"),
                          cfg.eval_episodes, cfg.seed + 2, cfg.extended_state)
    eval_vwap = _evaluate("dt_vwap_u",
                          lambda s: dtp.act(float(s[0]), float(s[1]), schedule="vwap_u"),
                          cfg.eval_episodes, cfg.seed + 3, cfg.extended_state)

    metrics = {
        "ppo_mean_is_bps":   eval_ppo.mean_is_bps,
        "sac_mean_is_bps":   eval_sac.mean_is_bps,
        "dt_twap_is_bps":    eval_twap.mean_is_bps,
        "dt_vwap_u_is_bps":  eval_vwap.mean_is_bps,
        "ppo_completion":    eval_ppo.completion_rate,
        "sac_completion":    eval_sac.completion_rate,
        "dt_completion":     eval_twap.completion_rate,
        "sac_alpha":         sac.alpha,
    }
    return PipelineReport(name="project3_rl_execution", metrics=metrics)
