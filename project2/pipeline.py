from __future__ import annotations
from dataclasses import dataclass
from quant_showcase.core.types import PipelineReport
from .pricing import AsianOptionEngine, AsianOptionParams

RESEARCH_NARRATIVE = {
    "problem": "Path-dependent Asian options require Monte Carlo; naive pricing has high variance.",
    "hypothesis": "Combining antithetic variates + geometric control variate achieves >99% variance reduction.",
    "method_stack": [
        "Plain MC baseline",
        "Antithetic variates: pair (Z, -Z) paths",
        "Control variate: geometric Asian closed-form (Kemna-Vorst 1990)",
        "Heston stochastic vol extension (Euler-Milstein)",
        "Pathwise delta estimator + finite-difference Gamma/Vega",
        "Implied vol inversion (Brent bisection, 50 iter)",
    ],
}

@dataclass(frozen=True)
class Project2Config:
    seed: int = 42; n_paths: int = 80_000
    s0: float = 100.0; k: float = 100.0; t: float = 1.0
    r: float = 0.03; q: float = 0.0; sigma: float = 0.25
    n_steps: int = 252; option_type: str = "call"

def run(seed: int = 42, config: Project2Config | None = None) -> PipelineReport:
    cfg    = config or Project2Config(seed=seed)
    params = AsianOptionParams(**{f: getattr(cfg, f) for f in
                                   ['s0','k','t','r','q','sigma','n_steps','option_type']})
    engine = AsianOptionEngine(params, n_paths=cfg.n_paths)

    plain, plain_se         = engine.price_plain(seed=cfg.seed)
    anti,  anti_se          = engine.price_antithetic(seed=cfg.seed)
    cv,    cv_se, vr        = engine.price_control_variate(seed=cfg.seed)
    heston, heston_se       = engine.price_heston(seed=cfg.seed)
    greeks                  = engine.greeks_fd()

    metrics = {
        "plain_price": plain, "plain_se": plain_se,
        "antithetic_price": anti, "antithetic_se": anti_se,
        "control_variate_price": cv, "control_variate_se": cv_se,
        "variance_reduction": vr,
        "heston_price": heston, "heston_se": heston_se,
        "delta": greeks["delta"], "delta_pathwise": greeks["delta_pathwise"],
        "gamma": greeks["gamma"], "vega": greeks["vega"],
    }
    return PipelineReport(name="project2_asian_options", metrics=metrics)
