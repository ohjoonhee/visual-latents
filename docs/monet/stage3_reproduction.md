# Monet SFT Stage 3 — Our Exact Reproduction

> This document describes **exactly what we ran**, pulled verbatim from the
> upstream code (`github.com/NOVAglow646/Monet`, branch on cluster identical
> to `main` at the model-upload commit) and our submit scripts. Every flag,
> formula, and shape below is what was actually executed — not a paraphrase.
> Use it to diff against the paper.

---

## 0. What Stage 3 is (one paragraph)

Monet trains a VLM to emit **latent "abstract visual tokens"** (K slots) in
place of the explicit auxiliary images a visual-chain-of-thought model would
normally generate. Training is 3 SFT stages:

| Stage | Objective | What it learns |
|---|---|---|
| **Stage 1** | Next-token prediction (NTP) SFT | Fluently emit the Monet special tokens + answer format |
| **Stage 2** | Alignment + emphasized-CE, latents **conditioned on the real aux image** | Latent slots that encode the aux-image content |
| **Stage 3** | Same alignment + CE, but latents are produced **from the question alone** (aux images removed); supervised to match the **Stage-2 teacher's** latent geometry | Generate the useful latents **without ever seeing the aux image** |

**Stage 3 = distillation:** a *student* (init from Stage 1) is taught to
produce, from the question only, the same per-layer latent representations
that the *teacher* (released Stage 2) produced when it **could** see the aux
images. The supervision is a cosine-alignment loss to precomputed teacher
latents, plus a (weighted) language-modeling CE.

---

## 1. The exact pipeline we ran

```
released Monet-SFT-7B/stage2 ──(precompute, latent_mode forward)──► 124,165 teacher latent files
                                                                       (all 28 layers × K=8 slots)
                                                                              │
released Monet-SFT-7B/stage1 ──(init)──► STUDENT ──train (3 epochs)──► our Stage-3 checkpoint
                                            ▲                                  │
                                            └── alignment loss to teacher ─────┘
```

- **Init checkpoint (student):** released `Monet-SFT-7B/stage1`
- **Teacher (for precompute):** released `Monet-SFT-7B/stage2`
- **Teacher latents:** 124,165 `.pt` files, one per training sample
- **Data:** the full released `Monet-SFT-125K` mix, all 6 subsets
  (`Visual_CoT`, `CogCoM`, `ReFocus`, `Zebra_CoT_count`,
  `Zebra_CoT_visual_search`, `Zebra_CoT_geometry`)

---

## 2. Step A — Teacher-latent precompute (verbatim)

Runs upstream `src.precompute_teacher_latents` **unmodified**:

```bash
torchrun --nproc-per-node=4 -m src.precompute_teacher_latents \
  --bsz 1 \
  --data_path <the 6 Monet-SFT-125K train.json files> \
  --load_model_path  <released Monet-SFT-7B/stage2> \
  --save_model_path  <upstream_teacher_latents/> \
  --dataset_root     <Monet-SFT-125K> \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size 8 \
  --output_hidden_states \
  --resume
```

The core forward (`src/precompute_teacher_latents.py`):

```python
inputs = {
    'latent_mode': True,                 # teacher runs WITH its aux images
    'input_ids': batch['input_ids'],
    'attention_mask': batch['attention_mask'],
    'pixel_values': batch['pixel_values'],
    'image_grid_thw': batch['image_grid_thw'],
    'labels': None,
    'loss_type': [],
}
inputs['output_hidden_states'] = True    # capture ALL layers
outputs = model(**inputs, return_dict=True)
teacher_reps = outputs.hidden_states     # per-sample: [num_layers, K, hidden]
# saved as latent_all_layers_<dataset>_<sample_id>.pt
torch.save({'metadata_info': ..., 'latent': teacher_reps[b].detach().cpu()}, save_path)
```

So each teacher file holds the Stage-2 hidden states at the **K=8 latent
positions across all 28 layers** — this is the alignment target.

---

## 3. Step B — Student data construction (`collate_fn_sft_stage3`, verbatim)

This is the heart of "Stage 3 = from the question only." For each sample:

1. Render the chat with `apply_chat_template`.
2. **Aux images → latent slots.** `replace_img_pad_with_latent_pad(texts,
   latent_size=8, "<abs_vis_token_pad>")` replaces each assistant aux-image
   placeholder with a **block of K=8 `<abs_vis_token_pad>` tokens**
   (id `151665`), wrapped by `<abs_vis_token>` / `</abs_vis_token>`
   (`151666` / `151667`).
3. **Strip aux images from the student input.** `remove_auxiliary_images(...)`
   drops every assistant-turn image; only the **question image(s)** remain.
   The student therefore never sees the aux images — it must *generate* the
   latents from the question.
4. Build tensors with the processor:
   ```python
   batch["student_input_ids"]       = student_batch["input_ids"]
   batch["student_attention_mask"]  = student_batch["attention_mask"]
   batch["student_pixel_values"]    = student_batch["pixel_values"]      # question imgs only
   batch["student_image_grid_thw"]  = student_batch["image_grid_thw"]
   ```
5. **Alignment positions** = the K latent-slot positions:
   ```python
   batch["student_alignment_poss"] = find_ids_poss(student_input_ids, answer_start_pattern, latent_pad_idx)
   ```
6. **Observation positions** (text between `<observation>`…`</observation>`)
   are collected — these get the **emphasized** CE weight.
7. **Labels** mask everything except the answer tokens (the special tokens,
   image pads, latent pads, observation markers are all ignored):
   ```python
   batch["student_labels"] = generate_labels_after_multi_token_start(
       student_input_ids, answer_start_pattern,
       ignore_ids=[img_pad, img_start, img_end, end_pad, latent_pad,
                   latent_end, observation_start, observation_end])
   ```

> Note: `--mask_latent` was **off** (default), so the optional 4-D
> latent-isolation attention mask was **not** applied — we used the model's
> standard causal mask, exactly as upstream's `sft_stage3.sh` does.

---

## 4. Step C — Stage-3 training loss (`CustomTrainerSFT_STAGE3.compute_loss`, verbatim)

Two forward passes per step:

```python
# Load the precomputed teacher latents for this batch (all layers, K slots)
teacher_latents = load_offline_tensor(self.teacher_latent_dir,
                    batch_metadata=inputs['metadata'],
                    alignment_layer=self.args.alignment_layer,   # 'all_layers'
                    rep_type="latent")

# ---- Forward 1: LATENT GENERATION (latent_mode=True) ----
# Student autoregressively produces the K latent embeddings from the
# question context (it has NO aux image). Returns where they go + their values.
inputs['latent_mode'] = True
inputs['input_ids']            = inputs['student_input_ids']
inputs['attention_mask']       = inputs['student_attention_mask']
inputs['pixel_values']         = inputs['student_pixel_values']
inputs['image_grid_thw']       = inputs['student_image_grid_thw']
inputs['alignment_poss']       = inputs['student_alignment_poss']
inputs['teacher_hidden_states_for_alignment'] = teacher_latents
inputs['loss_type'] = []
student_outputs_latent = model(**inputs)
#   → student_outputs_latent.ce_patch_pos, .ce_patch_vec  (the K latents)

# ---- Forward 2: CE + ALIGNMENT (latent_mode=False) ----
# Splice the generated latents back in; compute weighted CE on the answer
# AND the alignment loss to the teacher (all layers).
inputs['latent_mode'] = False
inputs['labels']            = inputs['student_labels']
inputs['ce_patch_pos']      = student_outputs_latent.ce_patch_pos
inputs['ce_patch_vec']      = student_outputs_latent.ce_patch_vec
inputs['ce_emphasize_factor'] = self.ce_emphasize_factor          # 4.0
inputs['ce_emphasize_poss']   = inputs['observation_poss']        # observation tokens ×4
inputs['loss_type'] = ['ce', 'alignment']
(student_ce_loss, student_outputs) = super().compute_loss(model, inputs, return_outputs=True, ...)

alignment_loss = student_outputs.loss_dict['alignment']
loss = student_ce_loss + self.alignment_weight * alignment_loss   # alignment_weight = 2.0
# (our VICReg term is guarded by lambda_reg > 0; lambda_reg = 0 here, so it is OFF and
#  this run is byte-identical to upstream.)
```

### The alignment loss formula (verbatim, `modeling…_monet.py`)

```python
def alignment_loss(teacher_hidden_states, student_hidden_states):
    total_loss = 0
    if teacher_hidden_states.dim() == 3:   # [num_layer, num_align, dim] — ALL layers (our case)
        total_loss = (1 - F.cosine_similarity(
            teacher_hidden_states.to(student_hidden_states.device),
            student_hidden_states)).mean()
    elif teacher_hidden_states.dim() == 1: # last-layer-only variant (not used)
        total_loss = 1 - F.cosine_similarity(student_hidden_states, teacher_hidden_states, 0)
    return total_loss
```

With `--alignment_layer all_layers`, the teacher tensor is
`[28 layers, K=8, hidden=3584]`; the loss is the **mean over all layers and
all K slots of `(1 − cosine_similarity)`** between student and teacher
latent hidden states.

### Emphasized CE (verbatim)

CE is computed per token; tokens at `observation_poss` are scaled by
`ce_emphasize_factor = 4.0`:

```python
weight = torch.ones_like(ce)
for b, poss in enumerate(ce_emphasize_poss):
    weight[b, torch.tensor(poss) - 1] = float(ce_emphasize_factor)   # 4.0
```

**Total objective:**  `loss = CE_emphasized  +  2.0 × alignment_loss`

---

## 5. Exact hyperparameters (our run)

The training launch (`slurm/upstream_stage3.sbatch`):

```bash
torchrun --nproc-per-node=4 -m src.main \
  --epochs 3 \
  --bsz 1 \
  --grad_accum_steps 32 \
  --stage sft_stage3 \
  --data_path <6 subsets> \
  --load_model_path  <released Monet-SFT-7B/stage1> \
  --save_model_path  <upstream_stage3_baseline_ep3> \
  --dataset_root     <Monet-SFT-125K> \
  --deepspeed ./deepspeed/ds_zero2_gpu.json \
  --latent_size 8 \
  --alignment_weight 2.0 \
  --ce_emphasize_factor 4.0 \
  --teacher_latent_dir <upstream_teacher_latents> \
  --alignment_layer all_layers \
  --lambda_reg 0.0
```

`SFTConfig` (upstream `src/main.py`, unchanged) supplies the rest:

| Knob | Value | Source |
|---|---|---|
| epochs | **3** | our flag (= released recipe at model-upload commit) |
| effective batch | **128** | 4 GPU × bsz 1 × grad_accum 32 |
| optimizer steps/epoch | **971** | 125 072 samples / 128 |
| total steps (`max_steps`) | **2 913** | 3 × 971 |
| learning rate | **1e-5** | `--lr` default |
| lr scheduler | **linear** | `SFTConfig` default (warmup → linear decay to 0) |
| warmup_steps | **10** | `SFTConfig` |
| weight_decay | **0.01** | `SFTConfig` |
| optimizer | **adamw_torch_fused** | `SFTConfig` |
| precision | **bf16** | `SFTConfig` |
| gradient_checkpointing | **True** | `SFTConfig` |
| parallelism | **DeepSpeed ZeRO-2** | `ds_zero2_gpu.json` |
| latent slots K | **8** | `--latent_size 8` |
| alignment_weight | **2.0** | `--alignment_weight` |
| ce_emphasize_factor | **4.0** | `--ce_emphasize_factor` |
| alignment_layer | **all_layers** | `--alignment_layer` |
| save_steps | **250** | `SFTConfig` |
| base model | **Qwen2.5-VL-7B** | architecture of all stages |

### Special-token vocabulary (from the checkpoint `config.json` / `added_tokens.json`)

| Token | ID | Role |
|---|---|---|
| `<abs_vis_token_pad>` | 151665 | the K=8 **latent slot** placeholder (`latent_token_id`) |
| `<abs_vis_token>` | 151666 | latent-block start (`latent_start_id`) |
| `</abs_vis_token>` | 151667 | latent-block end (`latent_end_id`) |
| `<observation>` / `</observation>` | 151668 / 151669 | mark the CE-emphasized answer span |
| `<|image_pad|>` | 151655 | question-image patches |
| `<|vision_start|>` / `<|vision_end|>` | 151652 / 151653 | image delimiters |

---

## 6. The ONE deviation from upstream

Upstream `sft_stage3.sh` step 2 used **8 GPUs × grad_accum 16**. We only have a
4-GPU partition, so we ran **4 GPUs × grad_accum 32**:

| | upstream | ours |
|---|---|---|
| GPUs (`--nproc-per-node`) | 8 | **4** |
| `--grad_accum_steps` | 16 | **32** |
| effective batch | 128 | **128** ✓ identical |
| optimizer steps/epoch | 971 | **971** ✓ identical |
| LR-decay horizon (`max_steps`) | 2 913 | **2 913** ✓ identical |

This is **batch-equivalent**: the effective batch size, total optimizer
steps, and the linear-LR-decay horizon are all identical. Everything else —
code, data, init, teacher, hyperparameters — is verbatim upstream.

> Walltime note: 3 epochs ≈ 40 h > the 24 h cap, so the run was split into
> two jobs sharing one epochs=3 schedule (`--resume_from_checkpoint`). This is
> purely an operational split; the schedule and `max_steps` are unchanged, so
> the resulting training is identical to a single uninterrupted 3-epoch run.

---

## 7. Result of this reproduction

Pairwise off-diagonal cosine similarity of the K=8 latent slots (the collapse
metric) and downstream utility, K=8, n=200 held-out:

| checkpoint | mean off-diag cos | utility (qwen-base reader) | self utility |
|---|---:|---:|---:|
| released **stage 2** (the teacher) | 0.377 | +2.05 | +2.78 |
| **our stage 3, epochs = 3** (step 2913) | **0.401** | **+1.02** | **+0.79** |
| our stage 3, epochs = 2 (step 1500) | 0.420 | +0.90 | −0.51 |
| released **stage 3** (the artifact we tried to match) | **0.871** | −0.05 | +0.20 |

**Finding:** run faithfully, Stage 3 converges to the **teacher's healthy
geometry (cos ≈ 0.38–0.40)** and stays useful. It does **not** reproduce the
released Stage-3 checkpoint's collapse (cos 0.871). Because the alignment loss
pulls the student toward the teacher, and the released Stage-2 teacher is
healthy (0.377), a faithful Stage 3 cannot collapse to 0.871 — that collapse
must originate from something not present in the public recipe + checkpoints.

---

*Generated from the exact upstream source + our submit scripts. Diff each
section against the paper to surface discrepancies.*
