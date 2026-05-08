"""Config dataclasses + YAML loader.

Configs are YAML files under `configs/`. Each YAML maps 1:1 to a `Round3Config`
instance. Loading is hand-rolled (no Hydra, no Pydantic) to keep the path
between YAML and code trivially readable.

Variant B (GRPO/VLPO) adds a `vlpo` section to the same Round3Config — see the
field at the bottom. For Variant A runs the `vlpo` field is None.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

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
    max_time: str | None = None        # e.g. "23h" for chained jobs
    resume: bool = False
    seed: int = 0
    bf16: bool = True


@dataclass
class WandBConfig:
    project: str = "visual-latents"
    entity: str | None = None
    run_name: str | None = None
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


@dataclass
class InterleavedConfig:
    """Interleaved variant only. Coconut-style recurrence within latent spans.

    Per `docs/INTERLEAVED_LATENT_DESIGN.md`:
      - T_blocks: number of alternating latent spans (each preceded by a text segment).
      - k_latent: number of latent positions per span; K_total = T_blocks * k_latent.
      - templates_per_step: not used in the POC (templates sampled per example);
        kept here for forward-compat with template-mixing curricula.
    """

    T_blocks: int = 2
    k_latent: int = 4
    # Optional: the trainer can use synthetic (template-driven) traces with
    # placeholder data instead of MixedDataset. POC defaults to synthetic so
    # the gradient probe is self-contained.
    use_synthetic_data: bool = True
    # Design §11.3 control: when true, build the trace text from a DIFFERENT
    # template than the one that produced the (question, answer) the reader
    # scores against. The latents are therefore conditioned on a mismatched
    # trace; if NLL still drops as fast as the natural pairing, the model is
    # learning trace-template surface form rather than using the latents.
    permute_template: bool = False
    # Round-2 norm-diagnostic C2 (docs/INTERLEAVED_POC_RESULTS.md §9). When
    # True, the previous latent's hidden state is `.detach()`ed before being
    # used as the next position's input embedding inside a latent block. This
    # decouples the recurrence's scale propagation from autograd — purely a
    # diagnostic for the self-sustaining-scale hypothesis. BREAKS the design
    # (gradient no longer flows through the recurrence chain). Do not use as
    # a default.
    detach_recurrence_input: bool = False
    # Round-3: switch the per-step example sampler. "synthetic" (default) keeps
    # round-2 behaviour (solid-colour images + 5 hand-written templates).
    # "gqa" loads real GQA records via `vl.data.gqa.load_gqa` and uses the
    # generic trace template (`traces.GENERIC_TEMPLATE`) so the natural-
    # language GQA questions do not need to fit category-specific placeholders.
    data_source: Literal["synthetic", "gqa", "shapes", "viscot", "grid"] = "synthetic"
    # Visual-CoT (round-4): the HF Hub repo with the preprocessed 50K subset
    # (cluster build via slurm/preprocess_viscot.sbatch). Default points at the
    # project owner's namespace; override per-machine via env if needed.
    viscot_hub_repo: str = "ohjoonhee/visual-cot-50k-poc"
    viscot_vsem_train: str = "data/viscot/viscot_50k_train_vsem.parquet"
    viscot_vsem_eval: str = "data/viscot/viscot_1k_eval_vsem.parquet"
    # Stage-1 → Stage-2 curriculum warmup (steps). w_grounding decays 1.0→0.3,
    # w_nll ramps 0.0→1.0 over this window. Active only when vsem_full+vsem_crop
    # are provided (data_source="viscot" with V_sem features available).
    stage1_warmup_steps: int = 500
    # When data_source="gqa", how many GQA train images to load into the pool
    # the trainer samples from each step. Held-out eval uses a separate small
    # set of testdev records (10 by construction), unaffected by this.
    n_samples: int = 100
    # Round-3 norm-diagnostic fix (b): apply the L_norm penalty ONLY to the
    # FIRST latent of each block. Hypothesis: the recurrent positions inherit
    # scale from the previous hidden state through the input-embedding feed,
    # creating a self-sustaining magnitude that resists the scalar norm penalty.
    # The first latent of each block is fed by the standard <|latent_start|>
    # embedding (not by recurrence), so it should be steerable via the standard
    # penalty. If THIS converges to target while baseline does not, the
    # recurrence is the cause. Implemented in trainer.run_one_step by zeroing
    # the standard norm contribution and adding back a custom term scoped to
    # first-of-block positions. Diagnostic only; do not use as default.
    first_latent_norm_only: bool = False
    # Round-3 binding test (POC results §11.4): when True AND data_source=
    # "shapes", each step samples K_q DIFFERENT (q, a) pairs from the same
    # scene's GT instead of duplicating one. This restores the q-invariance
    # pressure that the parallel method's `nll_multi_anchor` loss assumes —
    # without it, h can specialise to encode just one (q, a) per step,
    # producing the natural-vs-perm gap collapse observed in §11.3.
    multi_q_per_image: bool = False
    # Overnight 2026-05-04: at end of training, run a latent-position ablation
    # over the held-out set and write `results/<run>/ablation_eval.jsonl`. Each
    # row scores reader-NLL with a SUBSET (or perturbation) of the K_total latent
    # positions present, the rest replaced with zeros / random noise. Tests
    # whether information is uniformly spread across positions (evidence-style)
    # or concentrated in a small subset (compression-to-final-state, the
    # predicted failure mode for state-tracking tasks under reader-NLL alone).
    final_ablation_eval: bool = False


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
    vlpo: VLPOConfig | None = None                        # Variant B only
    interleaved: InterleavedConfig | None = None          # Interleaved variant only


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
        interleaved=InterleavedConfig(**d["interleaved"]) if "interleaved" in d else None,
    )
