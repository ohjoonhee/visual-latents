"""Loss-coefficient curriculum.

Cosine-warmup schedules per `docs/inherited/AUX_LOSS_AND_ARCH_DESIGN.md`:
- NLL coefficient ramps from `start` to `end` over `warmup_steps`.
- L_norm coefficient ramps from `start` to `end` over `warmup_steps`.
- L_concept coefficient is fixed at `LossConfig.w_concept` (no warmup).

Pure functions of step ∈ [0, warmup_steps]; deterministic.
"""

import math


def cosine_ramp(step: int, warmup: int, start: float, end: float) -> float:
    """Cosine ramp from `start` (step=0) to `end` (step=warmup); flat at `end` after."""
    if step <= 0:
        return start
    if step >= warmup:
        return end
    # gamma in [0, 1]: 0 at step=0, 1 at step=warmup
    gamma = 0.5 * (1.0 - math.cos(math.pi * step / warmup))
    return start + gamma * (end - start)


def nll_weight(step: int, cfg) -> float:
    return cosine_ramp(
        step, cfg.curriculum_warmup_steps, cfg.nll_weight_start, cfg.nll_weight_end
    )


def norm_weight(step: int, cfg) -> float:
    return cosine_ramp(
        step, cfg.curriculum_warmup_steps, cfg.norm_weight_start, cfg.norm_weight_end
    )


def concept_weight(step: int, cfg) -> float:
    """Constant; no warmup. Returns `cfg.w_concept` regardless of step."""
    del step
    return cfg.w_concept
