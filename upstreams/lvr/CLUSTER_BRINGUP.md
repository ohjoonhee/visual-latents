# LVR (arXiv:2509.24251) — vendored upstream + cluster bring-up

This directory is the **LVR repo** (github.com/VincentLeebang/lvr, Apache-2.0)
vendored into `visual-latents` so the whole 3B reproduction is managed in **one
git repo** and reaches the cluster by `git pull` (no rsync). Upstream `src/`
model/trainer/loss code is **UNTOUCHED**; our changes are paths + 4-GPU knobs +
eval scaffolding only.

## What differs from pristine upstream (the only edits)
- `evaluation/evaluation.py` — drop the Oracle-Cloud (OCI) creds block;
  `CHKPT_PATHS` → released LVR-7B; `flash_attention_2`→`sdpa` (no flash-attn at
  eval; identical greedy/steps math); base data path → `lvr_eval/data`; benchmark
  loop reordered. (Phase-1 anchor only; eval runs locally, not on the cluster.)
- `data/meta_data_lvr_sft_stage1.json` — **new** SFT manifest (cluster paths
  `/data/joonhee/lvr/...`).
- `slurm/*.sbatch` — **new** cluster wrappers (below). They reproduce the released
  `scripts/finetune_lvr_stage{1,2}_3b.sh` args **verbatim** + cluster adaptations.
  `REPO` is derived from `$SLURM_SUBMIT_DIR`, so they run from wherever
  `visual-latents` is checked out — **submit from this directory**.
- `slurm/uv_overrides.txt` — **new**. The env build applies it via
  `uv pip install --override` to fix two broken pins in the upstream freeze
  (`av==14.3.0` yanked from PyPI; `huggingface-hub==0.30.2` contradicts
  `transformers==4.54.0`). `requirements.txt` is left byte-identical to upstream.
- `.gitignore`, `logs/.gitkeep` — housekeeping.

Documented deviations vs the released scripts: (1) Stage-1 `grad_accum 8→16`
(4 GPUs vs the script's 8 → eff. batch `1×4×16 = 64`, matching `1×8×8`);
(2) cluster paths; (3) `--online_checkpoint False` (local save, no OCI — `boto3`
is installed so the top-level import resolves, `oci_handler` is never
instantiated, **no code edit**); (4) **attention = sdpa** (`--disable_flash_attn2
True`, both stages). flash-attn 2.8.3's prebuilt wheels for torch 2.6 (both
`abiTRUE`/`abiFALSE`) demand the ABI-1 `__cxx11` c10 symbol, but torch 2.6.0+cu124
is ABI-0 (`cxx11_abi=False`) from PyPI **and** download.pytorch.org, and a source
build also came out ABI-1 — no compatible flash-attn exists for this torch. sdpa
is exact attention (numerically equivalent) and is the impl the eval anchor used to
reproduce the paper, so this is faithful. (env_build still installs a flash-attn
wheel + has `FA_ABI`/`TORCH_INDEX` knobs if a future torch makes it loadable.)

## Faithful hyperparameters (verbatim from the released scripts)
- **Stage-1 SFT**: `Qwen2.5-VL-3B-Instruct`, `max_steps 2500`, `lr 1e-5` cosine,
  `warmup 0.03`, `wd 0.1`, `loss_lvr_fct mse`, `loss_lvr_lambda 0.1`,
  `lvr_head False`, vision+merger frozen, LLM full FT, data-packing
  `max_packed_tokens 16384`, ZeRO-3-offload, `save_steps 500`.
- **Stage-2 GRPO_latent RL**: init = Stage-1 `checkpoint-2500`, `lr 5e-6`,
  `temperature 0.9`, `2 epochs`, `num_generations 8`, `decoding_strategy steps`,
  `lvr_steps 8`, KL 0.04, ZeRO-2. Rollouts are **in-process HF `generate`**
  (no vLLM); latent hidden states are recorded → no-grad teacher-force replay →
  patched back via `torch.where(lvr_mask, lvr_states, comp_embeds)`.

## Cluster bring-up — git-based (user submits every `sb`, §3-A: no agent-submit)

```
# 0. get the code onto the cluster via git (replaces rsync)
git clone https://github.com/ohjoonhee/visual-latents.git ~/projects/visual-latents
#   (or, if already cloned:)  cd ~/projects/visual-latents && git pull
cd ~/projects/visual-latents/upstreams/lvr
#   put .env here (HF_TOKEN, SLACK_WEBHOOK_URL) — gitignored, sourced by each sbatch

# 1. build the training env  (cpu-short)  -> ./.venv
sb slurm/lvr_env_build.sbatch

# 2. stage data  (cpu-short)  -> /data/joonhee/lvr/{data,images}
sb slurm/lvr_data_stage.sbatch      # JSONs + Visual-CoT 142GB + ViRL39K 1.8GB

# 3. Stage-1 SFT  (gpu-4farm 4xH100)
#    SMOKE FIRST (20 steps, isolated _SMOKE run dir): loss + loss_lvr finite &
#    decreasing, a checkpoint writes, no OOM-loop.
SMOKE=1 sb slurm/lvr_stage1_3b.sbatch
#    then the full run:
sb slurm/lvr_stage1_3b.sbatch       # full 2500;  RESUME=1 sb ...  to continue

# 4. upload SFT ckpt -> eval (this is the RL baseline)
#    upload via scripts/cluster/upload_ckpt_to_hf.py ; eval locally via lvr_eval
#    (V*/MMVP, steps {4,8,16})

# 5. Stage-2 GRPO_latent RL  (gpu-4farm 4xH100)
#    SMOKE FIRST (8 steps; inits from the latest stage-1 ckpt — a stage-1 SMOKE
#    ckpt is fine): rollouts emit <answer>, rewards > 0, GRPO loss finite.
SMOKE=1 sb slurm/lvr_stage2_3b.sbatch
#    then the full run — 2 epochs, heavy (8 gens x teacher-force replay/prompt),
#    budget 2-3 resume jobs under the 24h cap:
sb slurm/lvr_stage2_3b.sbatch

# 6. upload RL ckpt -> eval ; report the SFT->RL gain vs paper 3B
#    (V* ~65-67, MMVP ~55-58)
```

**On first run, verify:** Visual-CoT untar layout (`lvr_data_stage` auto-detects
the `flickr30k` dir and symlinks `images/viscot`); `train_lvr` resume semantics
(`--checkpoint_name`); GRPO memory on 4×H100-80GB.

Local Phase-1 anchor (validates the eval harness) and eval-data details live in
`visual-latents/lvr_eval/UPSTREAM_EDITS.md`. Released LVR-7B reproduced the paper:
V\* 80.63/81.68/80.63, MMVP 72.00/72.00/71.67 (steps 4/8/16).
