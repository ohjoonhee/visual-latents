"""Tests for vl.curriculum — cosine-warmup schedules."""

import math

import pytest

from vl.config import LossConfig
from vl.curriculum import concept_weight, cosine_ramp, nll_weight, norm_weight


def test_cosine_ramp_endpoints():
    assert cosine_ramp(0, 100, 0.1, 1.0) == pytest.approx(0.1)
    assert cosine_ramp(100, 100, 0.1, 1.0) == pytest.approx(1.0)
    # Past warmup stays at end
    assert cosine_ramp(150, 100, 0.1, 1.0) == pytest.approx(1.0)


def test_cosine_ramp_midpoint():
    # At step = warmup/2, cosine ramp gives gamma = 0.5, so value = start + 0.5*(end-start)
    midpoint = cosine_ramp(50, 100, 0.0, 1.0)
    assert midpoint == pytest.approx(0.5, abs=1e-6)


def test_cosine_ramp_monotonic():
    vals = [cosine_ramp(s, 200, 0.0, 1.0) for s in range(0, 201, 10)]
    for a, b in zip(vals, vals[1:], strict=False):
        assert a <= b + 1e-9


def test_nll_weight_curriculum():
    cfg = LossConfig(curriculum_warmup_steps=200, nll_weight_start=0.1, nll_weight_end=1.0)
    assert nll_weight(0, cfg) == pytest.approx(0.1)
    assert nll_weight(200, cfg) == pytest.approx(1.0)
    assert nll_weight(1000, cfg) == pytest.approx(1.0)
    # Mid-warmup
    assert 0.1 < nll_weight(50, cfg) < 1.0


def test_norm_weight_curriculum():
    cfg = LossConfig(curriculum_warmup_steps=200, norm_weight_start=0.0, norm_weight_end=0.1)
    assert norm_weight(0, cfg) == pytest.approx(0.0)
    assert norm_weight(200, cfg) == pytest.approx(0.1)


def test_concept_weight_constant():
    cfg = LossConfig(w_concept=0.3)
    # Should be constant regardless of step
    for step in (0, 50, 200, 1000):
        assert concept_weight(step, cfg) == pytest.approx(0.3)


def test_negative_step():
    # Defensive: negative or zero step returns start
    assert cosine_ramp(-1, 100, 0.1, 1.0) == pytest.approx(0.1)
    assert cosine_ramp(0, 100, 0.1, 1.0) == pytest.approx(0.1)


def test_zero_warmup_disabled():
    # warmup=0 → step >= warmup immediately, returns end
    # NB: avoid div-by-zero by short-circuiting on step >= warmup
    assert cosine_ramp(1, 0, 0.0, 1.0) == pytest.approx(1.0)


def test_cosine_formula():
    """Sanity: at step=warmup/4, gamma = (1 - cos(pi/4)) / 2."""
    val = cosine_ramp(25, 100, 0.0, 1.0)
    expected_gamma = 0.5 * (1.0 - math.cos(math.pi / 4))
    assert val == pytest.approx(expected_gamma, abs=1e-6)
