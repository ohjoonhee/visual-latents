"""Tests for vl.config — every YAML config in configs/ must load cleanly."""

from pathlib import Path

import pytest

from vl.config import (
    AnchorsConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    Round3Config,
    TrainerConfig,
    VLPOConfig,
    WandBConfig,
    load_config,
)

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
ALL_CONFIGS = sorted(CONFIGS_DIR.rglob("*.yaml"))


def test_configs_dir_nonempty():
    assert ALL_CONFIGS, f"no YAML configs found under {CONFIGS_DIR}"


@pytest.mark.parametrize("config_path", ALL_CONFIGS, ids=lambda p: p.relative_to(CONFIGS_DIR).as_posix())
def test_config_loads(config_path):
    cfg = load_config(config_path)
    assert isinstance(cfg, Round3Config)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.anchors, AnchorsConfig)
    assert isinstance(cfg.loss, LossConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.trainer, TrainerConfig)
    assert isinstance(cfg.wandb, WandBConfig)


def test_round3_C1_full():
    cfg = load_config(CONFIGS_DIR / "round3" / "C1_full.yaml")
    assert cfg.cell == "C1"
    assert cfg.model.K == 16
    assert cfg.model.lora_r == 32
    assert cfg.loss.K_q == 3
    assert cfg.loss.w_concept == 0.3
    assert cfg.loss.w_norm == 0.1
    assert cfg.loss.target_norm == 57.86
    assert len(cfg.anchors.paths) == 2
    assert cfg.trainer.variant == "A"
    assert cfg.vlpo is None


def test_round3_C5_random_control_flag():
    cfg = load_config(CONFIGS_DIR / "round3" / "C5_random_control.yaml")
    assert cfg.data.shuffle_qa_within_image is True


def test_round3_C4_no_concept():
    cfg = load_config(CONFIGS_DIR / "round3" / "C4_no_concept.yaml")
    assert cfg.loss.w_concept == 0.0


def test_variant_b_pilot_has_vlpo():
    cfg = load_config(CONFIGS_DIR / "variant_b" / "pilot.yaml")
    assert cfg.trainer.variant == "B"
    assert isinstance(cfg.vlpo, VLPOConfig)
    assert cfg.vlpo.sigma == 5.0
    assert cfg.vlpo.beta_latent == 0.04
    assert cfg.vlpo.beta_text == 0.0
