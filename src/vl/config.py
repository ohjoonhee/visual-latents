"""Config dataclasses + YAML loader.

Configs are YAML files under `configs/`. Each YAML maps 1:1 to a `Round3Config`
instance. Loading is hand-rolled (no Hydra, no Pydantic) to keep the path
between YAML and code trivially readable.

Variant B (GRPO/VLPO) adds a `vlpo` section to the same Round3Config — see the
field at the bottom. For Variant A runs the `vlpo` field is None.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


# =============================================================================
# Component configs
# =============================================================================
@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    K: int = 16
    new_token: str = "<|latent|>"
    stage1_attention_mask: bool = True
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class AnchorsConfig:
    """Frozen anchor models (formerly 'readers'). R = len(paths)."""
    paths: list[str] = field(
        default_factory=lambda: [
            "Qwen/Qwen2.5-VL-7B-Instruct",
            # Round-3 cell C2 ablates to a single anchor; M1 may extend to R=3
            # by adding e.g. "TIGER-Lab/VL-Rethinker-7B"
        ]
    )


@dataclass
class LossConfig:
    w_concept: float = 0.3
    w_norm: float = 0.1
    target_norm: float = 57.86           # natural visual-token norm for Qwen2.5-VL-7B
    K_q: int = 3                          # multi-Q per image
    curriculum_warmup_steps: int = 200
    nll_weight_start: float = 0.1
    nll_weight_end: float = 1.0
    norm_weight_start: float = 0.0
    norm_weight_end: float = 0.1
    concept_bottleneck_dim: int = 1792    # D/2 for Qwen2.5-VL-7B (D=3584)


@dataclass
class DataConfig:
    mix: dict[str, float] = field(
        default_factory=lambda: {"gqa": 0.65, "clevr": 0.25, "tallyqa": 0.10}
    )
    n_samples: int = 10000
    image_max_pixels: int = 256 * 28 * 28
    shuffle_qa_within_image: bool = False  # True → random-control cell C5


@dataclass
class TrainerConfig:
    variant: Literal["A", "B"] = "A"
    batch_size: int = 4
    gradient_checkpointing: bool = True
    max_steps: int = 1000
    lr_lora: float = 5e-5
    lr_token: float = 5e-3
    optim: str = "adamw_torch"
    eval_every_steps: int = 100
    ckpt_every_steps: int = 200
    max_time: Optional[str] = None        # e.g. "23h" for chained jobs
    resume: bool = False
    seed: int = 0
    bf16: bool = True


@dataclass
class WandBConfig:
    project: str = "visual-latents"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    enabled: bool = True


@dataclass
class VLPOConfig:
    """Variant B only. Gaussian reparameterization on h positions."""
    sigma: float = 5.0
    beta_latent: float = 0.04
    beta_text: float = 0.0
    group_size: int = 8
    reward_kind: Literal["exact_match", "judge"] = "exact_match"
    multi_anchor_reward: bool = True       # sum reward over all anchors
    random_control_negative_weight: float = 0.5


# =============================================================================
# Top-level config
# =============================================================================
@dataclass
class Round3Config:
    name: str = "unnamed"
    cell: str = ""                                           # e.g. "C1_full"
    model: ModelConfig = field(default_factory=ModelConfig)
    anchors: AnchorsConfig = field(default_factory=AnchorsConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    vlpo: Optional[VLPOConfig] = None                        # Variant B only


# =============================================================================
# YAML loader (explicit; no recursion magic)
# =============================================================================
def load_config(path: str | Path) -> Round3Config:
    """Load YAML and construct a Round3Config. Missing sections use defaults."""
    with Path(path).open() as f:
        d = yaml.safe_load(f) or {}

    return Round3Config(
        name=d.get("name", Path(path).stem),
        cell=d.get("cell", ""),
        model=ModelConfig(**d.get("model", {})),
        anchors=AnchorsConfig(**d.get("anchors", {})),
        loss=LossConfig(**d.get("loss", {})),
        data=DataConfig(**d.get("data", {})),
        trainer=TrainerConfig(**d.get("trainer", {})),
        wandb=WandBConfig(**d.get("wandb", {})),
        vlpo=VLPOConfig(**d["vlpo"]) if "vlpo" in d else None,
    )
