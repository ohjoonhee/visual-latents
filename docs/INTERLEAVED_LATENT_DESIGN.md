# Interleaved Text-Latent Reasoning — POC Design

**Date:** 2026-05-03
**Status:** design contract for the next implementation agent
**Companion to:** `docs/INTERLEAVED_LITRECON.md` (litrecon),
  `docs/METHODS.md` (current parallel method, this is its sibling),
  `docs/inherited/ROUND3_POC_DESIGN.md` (5-cell sweep targeted by the parallel method).

Minimal feasibility POC for an interleaved text-latent variant. The
current method emits K continuous latents in one parallel pass; this
variant interleaves text with short latent spans in one autoregressive
trace, à la Coconut (litrecon §1) but with frozen-reader supervision
in place of NTP on post-latent text. Sibling, not replacement;
runnable on a single A6000 in ~2 h.

---

## 1. Goal and scope of POC

Three feasibility questions, in order:

(a) **Trainability.** Does a pre-tokenized interleaved trace with
short latent spans complete forward + backward without NaN / shape
mismatch / graph break across the recurrence?

(b) **Gradient flow through the recurrence.** Does the trailing
reader-NLL gradient reach (i) LoRA in *every* forward pass within the
trace, (ii) `new_emb`, (iii) the concept MLP? Specifically, does grad
on the *first* latent block survive, propagating back through the
within-block hidden-state recurrence? Nonzero `new_emb.grad` only on
the last block would mean a broken graph.

(c) **Reader-anchor signal shaping trace content.** Over ~50 steps,
does held-out reader-NLL on the same template family decrease while
`‖h‖` migrates toward μ=57.86 (`docs/METHODS.md` §3.3)? Directional,
not convergence.

**Out of scope:** round-3 5-cell sweep, all five ROUND3 §1 gates,
steering / random-control / 5K stress, multi-reader transfer (R≥2),
DDP, cluster submission, teacher-bootstrapped traces. POC: R=1, B=1,
hand-written templates, single GPU.

---

## 2. Trace structure and segmentation

### 2.1 Schema

Each example is one autoregressive trace, fully pre-tokenized:

```
PREFIX_TEXT (system + user question, with real image tokens)
  <|latent_start|> <|latent|>×k_latent <|latent_end|>
TEXT_SEGMENT_1
  <|latent_start|> <|latent|>×k_latent <|latent_end|>
TEXT_SEGMENT_2
  ...  (T_blocks alternations)  ...
SUFFIX_TEXT (assistant answer)
```

POC: **`T_blocks=2`, `k_latent=4` ⇒ `K_total=8`**. Same K_total as a
parallel `K=8` config, so we reuse `forward_anchor` 1:1.

**Why fixed alternation, not learned gating:** litrecon §3 — no
surveyed paper specifies a working "decide-when-to-emit" mechanism. A
learned gate adds a categorical latent that complicates gradient
attribution for a feasibility check. Defer to future work (§7).

### 2.2 Special tokens

Reuse `<|latent_start|>`, `<|latent|>`, `<|latent_end|>` from the
parallel method (`src/vl/model.py:24`). `new_emb` and
`_splice_new_emb` already exist; `<bot>`/`<eot>` would be cosmetic
and re-init two rows for no mechanistic gain. Preserves checkpoint
compatibility.

### 2.3 Worked example — "How many cubes are there?"

Image: CLEVR scene of mixed shapes. Gold answer: `3`.

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>How many cubes are there?<|im_end|>
<|im_start|>assistant
Let me look at the shapes in the image.
<|latent_start|><|latent|><|latent|><|latent|><|latent|><|latent_end|>
Now I will count the cubes specifically.
<|latent_start|><|latent|><|latent|><|latent|><|latent|><|latent_end|>
The answer is 3.<|im_end|>
```

The interstitial cues are hand-written *anchors*; teacher-forced,
excluded from loss (§5, §6.5). The 8 latent hidden states harvested
from the trace become `h ∈ ℝ^{8 × D}` for the reader.

---

## 3. Latent emission mechanism

### 3.1 Coconut-style hidden-state recurrence within latent spans

Inside a latent span, **the hidden state at position i is the input
embedding at position i+1** (litrecon §1, Coconut verbatim). For a
block starting at position p:

- Position p: `<|latent_start|>`. Input embed = `new_emb[1]`. Forward → `h_p`.
- Position p+1: first `<|latent|>`. Input embed = `h_p`. Forward → `h_{p+1}`.
- ... continue for `k_latent` steps ...
- Position p+k_latent+1: `<|latent_end|>`. Input embed = `new_emb[2]`.

Harvested: `(h_{p+1}, …, h_{p+k_latent})`.

**Critical:** `new_emb[0]` is **never read as an input embedding
inside the span** under Coconut recurrence — input embeddings are
always prior hidden states. Remains trainable (parallel method's
checkpoint surface) but is a near-spectator here; flagged in §7.

### 3.2 Recurrence across text boundaries

When the span ends, recurrence does **not** cross the boundary as an
input-embedding substitution: the first text-mode position takes the
standard token embedding of the next teacher-forced text token. What
*does* cross is the residual / attention path — every subsequent
position attends freely to all prior latent hidden states, the route
the trailing reader-NLL gradient takes back into each latent. Common
misreading of Coconut and Mirage (litrecon §2.1).

### 3.3 Forward-pass count

Coconut needs n+1 forward passes for n latent thoughts because each
thought's input embedding depends on the prior pass's output (litrecon
§1). We adopt the same pattern with no curriculum: per training
example,

`generator passes = 1 (prefix) + k_latent (block 1) + 1 (transition+seg2+start_2) + k_latent (block 2) = 2·(k_latent+1) = 10`.

We do **not** use `past_key_values` between passes within a block — the
recurrence-participating activations must remain in the autograd
graph, so we re-run the growing prefix each pass. Cheaper KV-cache-
with-grad-retention is a §11 fallback only.

### 3.4 Alternative considered and rejected

The parallel method's "shared placeholder" mechanism (every
`<|latent|>` slot fed `new_emb[0]`, differentiated only by positional
encoding) is fine for parallel emission because the K positions
co-attend in one pass; but **inside a `k_latent=4` block** it
degenerates to a positionally-modulated copy of `new_emb[0]` with no
recurrent information flow — exactly the failure mode Coconut §3 /
litrecon §1 describes. Out of scope for the POC.

---

## 4. Reader-anchor consumption

Novelty axis (litrecon §3: no surveyed interleaved-VLM-latent paper
uses a separate frozen reader for training signal).

### 4.1 Three options

(i) **Latents-only.** Collapse all `K_total=8` latents into `[B, 8, D]`;
splice into 8 `<|image_pad|>` slots in the reader prompt. Reader sees
no trace text. Reuses `forward_anchor` verbatim.

(ii) **Latents + text trace.** Reader prompt mirrors the full
interleaved trace, with hidden states spliced at latent positions.
Closest to the natural CoT interpretation; but Qwen2.5-VL's vision
merger expects vision tokens only inside `<|vision_start|>…<|vision_end|>`,
so latents elsewhere would be processed as text-mode embeddings (no
precedent — uncertain semantics).

(iii) **Per-segment scoring.** For each block t ∈ {1,…,T_blocks}, score
with only the first `t·k_latent` latents spliced; sum. Multiplies
reader passes by T_blocks.

### 4.2 POC choice

**(i) latents-only.** Reuses `readers.py` 1:1 with `K=K_total=8`,
isolates the novelty to the generator-side recurrent forward, and
gives full continuity with the existing parallel `K=8` test surface.
The §10.1 gradient-flow check is about the recurrence graph, not
reader prompt structure. Graduate to (ii) once (i) verified — 1-day
change. (iii) rejected: 4× reader passes vs. parallel without
proportionate diagnostic value.

---

## 5. Discrete text gradient strategy

Per litrecon §4, the only multi-precedent strategy is **§4.1 teacher-
forcing on bootstrapped traces.** POC adopts it. Implications:

- All text tokens are *inputs*, not sampled outputs; zero loss
  contribution from generator pass (§6.5).
- Interleaving pattern is *dictated*, not learned.
- Sidesteps discrete-gradient entirely.

### 5.1 Bootstrap source

**Recommend (a) hand-written templates** — 10 templates total (4
count + 3 color + 3 position) over 30 GQA images. Reason: under our
control, bias-free; failure is unambiguously a mechanism failure, not
a teacher-trace failure. (b) frozen-VLM teacher introduces a confound
for a feasibility check — graduate to it once (a) demonstrates
trainability. (c) external teacher out of POC scope.

Each template has form `{prefix} <bot1> {seg1} <bot2> {answer}` with
~5 hand-written interstitials per template; with random per-step
template selection, 10 × 30 = 300 unique seeds, ample for 50 steps × B=1.

---

## 6. Loss

Reuse the three-term decomposition from `src/vl/losses.py` verbatim:

```
L_total = w_NLL(t) · L_NLL_multi + w_concept · L_concept + w_norm(t) · L_norm
```

### 6.1 L_NLL_multi

`R=1` (generator-shared reader), `K_q=2`, `h ∈ [1, 8, D]` per §4.2.
Existing `nll_multi_anchor` is shape-agnostic in K; no code change.

### 6.2 L_concept

**Per-block, concatenated**: gather all 8 latents into `h ∈ [B, 8, D]`;
extract `V_sem` once via `get_v_sem`; existing modular indexing
`target_idx = arange(8) % T_v` assigns each latent a target. Identical
to parallel `concept_loss`. Per-block extraction was rejected: with
`k_latent=4` and T_v ≈ 200-400, blocks would map to near-identical
target slots, wasting K_total's diversity.

### 6.3 L_norm

All 8 latent positions, target μ=57.86. Identical to parallel.

### 6.4 Curriculum

`vl.curriculum` cosine warmup, **`T_warmup=10`** (vs. 200 in parallel,
scaled to the 50-step budget). `w_NLL`: 0.1 → 1.0; `w_norm`: 0 → 0.1;
`w_concept` constant at 0.3.

### 6.5 Why no text-NLL term

Excluded deliberately. (i) Text is hand-written, not labeled —
predicting our anchor cues teaches a template, not reasoning. (ii)
Masks whether the latent path is load-bearing (Coconut keeps text-NLL
*only* on post-latent text for this reason — litrecon §1). (iii)
Cleanest signal for §1(b): the *only* reasonable gradient source on
the latents is the trailing reader-NLL.

---

## 7. Trainable parameters

Identical surface to parallel (`docs/METHODS.md` §2.1):

| Component | Status | Approx. count (3B) |
|---|---|---|
| Vision tower + projector + merger | frozen | 675 M |
| LM base (non-LoRA) | frozen | 3 B |
| LM head | frozen | — |
| LoRA r=16, all targets | trainable | ~25 M |
| `new_emb` (3 × D) | trainable | ~6 K |
| Concept MLP (D ↔ D/2) | trainable | ~8 M |

**No mode controller** for POC — segmentation is teacher-forced.
Future work: binary CE head per text-mode position labeling
"enter-latent-now" against the template's segmentation positions.

**`new_emb[0]` is a near-spectator** (per §3.1) — it receives gradient
only via the `<|latent|>` token's positional contribution to attention,
not via any input-embedding read. Log `new_emb.grad[0].norm()`; small-
but-nonzero is correct, exactly-zero indicates a wiring bug.

---

## 8. Forward-pass schedule

Per training example, `T_blocks=2`, `k_latent=4`, B=1:

```python
# Forward 1 — encode prompt up to <|latent_start|>_1.
prompt_ids = tokenize(PREFIX_TEXT + "<|latent_start|>")
prompt_embeds = embed(prompt_ids)                    # [1, L0, D] (with _splice_new_emb)
H = model(inputs_embeds=prompt_embeds, ...)          # full VLM forward
h_prev = H.hidden_states[-1][..., -1, :]             # [D]

latent_states = []

# Forwards 2..k_latent+1 — Coconut recurrence inside block 1.
for i in range(k_latent):
    new_pos = h_prev.unsqueeze(0).unsqueeze(0)       # [1, 1, D]
    prompt_embeds = torch.cat([prompt_embeds, new_pos], dim=1)
    H = model(inputs_embeds=prompt_embeds, ...)
    h_prev = H.hidden_states[-1][..., -1, :]
    latent_states.append(h_prev)

# Forward k_latent+2 — append <|latent_end|> + TEXT_SEGMENT_1 + <|latent_start|>_2.
seg_ids = tokenize("<|latent_end|>" + TEXT_SEGMENT_1 + "<|latent_start|>")
seg_embeds = embed(seg_ids)
prompt_embeds = torch.cat([prompt_embeds, seg_embeds], dim=1)
H = model(inputs_embeds=prompt_embeds, ...)
h_prev = H.hidden_states[-1][..., -1, :]

# Forwards (k_latent+3)..(2·k_latent+2) — Coconut recurrence inside block 2.
for i in range(k_latent):
    new_pos = h_prev.unsqueeze(0).unsqueeze(0)
    prompt_embeds = torch.cat([prompt_embeds, new_pos], dim=1)
    H = model(inputs_embeds=prompt_embeds, ...)
    h_prev = H.hidden_states[-1][..., -1, :]
    latent_states.append(h_prev)

# (Final pass for SUFFIX_TEXT skipped: not used by reader (§4.2 (i)) or any text-NLL (§6.5).)

h = torch.stack(latent_states, dim=0).unsqueeze(0)   # [1, 8, D]

# Reader pass — same as parallel.
v_sem = get_v_sem(model, processor, images)          # 1 frozen forward, no grad
loss  = combined(h, batch, anchors, v_sem, concept_mlp, cfg, step).total
loss.backward()
```

### 8.1 Pass count and cost

- Generator recurrent passes: `1 + k_latent + 1 + k_latent = 10`.
- V_sem extraction: 1 frozen.
- Reader passes: `R · K_q = 1 · 2 = 2`.
- **Total: 12 forward passes/step** (vs. 4 for parallel at K=8, R=1, K_q=2).

Sequence length grows from `L_0 ≈ 250` (post-image expansion) to
`L_0 + 10 ≈ 260`; per-pass cost is comparable to one parallel-method
pass at the same length. Estimate: **3-5 s/step on A6000 with
Qwen2.5-VL-3B**, ~3-5 min for 50 steps. Even with 5× overhead from
recurrence wiring, well within the 2 h budget.

### 8.2 Backprop graph

- `h_prev` participates in two paths: input embedding to subsequent
  passes (recurrence) AND harvested `h` consumed by the reader. Both
  must remain in the autograd graph.
- **No `.detach()`** on `h_prev` between iterations. **No `no_grad()`**
  around recurrent forwards. V_sem is the only `no_grad` path.
- Memory: 10 forward graphs held simultaneously before backward. Peak
  ~12 GB on Qwen2.5-VL-3B; within 48 GB. Gradient checkpointing not
  required and best avoided — `docs/METHODS.md` §4.3 documents its
  interaction with `_splice_new_emb`.

### 8.3 `inputs_embeds` API note

The standard embedding lookup for `<|latent|>` competes with our
hidden-state injection, so we maintain a parallel `[L]` `input_ids`
tensor for attention-mask and image-pad merger only (using `<|latent|>`
IDs as sentinels the merger ignores), and build `inputs_embeds`
directly via concat — same pattern as `forward_generator`
(`src/vl/model.py:221-244`).

---

## 9. POC config

| Parameter | Value | Source |
|---|---|---|
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` | A6000 budget; matches `docs/METHODS.md` §8 mini |
| LoRA r / α | 16 / 32 | mini config |
| LoRA targets | q,k,v,o + gate,up,down | parallel default |
| `T_blocks` / `k_latent` / `K_total` | 2 / 4 / 8 | §2.1 |
| B / K_q / R | 1 / 2 / 1 (gen-shared) | A6000 budget |
| Max steps | 50 | feasibility budget |
| Curriculum warmup | 10 | §6.4 |
| `w_concept` / `w_norm` (post-warmup) | 0.3 / 0.1 | parallel default |
| `target_norm` (μ) | 57.86 | parallel default |
| Optimizer | AdamW(0.9,0.95); lr=5e-5 LoRA+MLP, lr=5e-3 `new_emb` | parallel |
| Grad clip | 1.0 | parallel |
| Dataset | 10 hand-written templates × 30 GQA images | §5.2 |
| Held-out set | 5 examples, same template family, unseen images | §10 |

---

## 10. Evaluation criteria for "POC works"

### 10.1 Gradient flow (hard requirement)

After step 1, before optimization:

- `new_emb.grad` non-`None`; `new_emb.grad[1].norm() > 0`
  (latent_start, dominant new-token gradient per §3.1) and
  `new_emb.grad[2].norm() > 0` (latent_end). Row 0 may be small; log
  but do not require nonzero.
- All `concept_mlp` params: nonzero `grad`.
- ≥ 50 % of LoRA params: nonzero `grad`. Specifically, LoRA layers in
  the *first* transformer block must have nonzero grad — binding test
  for §1(b).

### 10.2 Norm trajectory

`mean(‖h‖)` migrates from ~190 (random init, parallel §3.3 baseline)
toward μ=57.86. Threshold: `< 80` by step 30. No NaN at any step.

### 10.3 Held-out NLL trend

5-example held-out set: mean reader-NLL improves by ≥ 0.5 nat from
step 0 to step 50. Loose; main signal is "monotonically trending
down, not flat."

If all three pass, mechanism is sound; next agent advances to (ii)
reader prompt (§4) and bootstrapped traces (§5).

---

## 11. Known risks and falsification triggers

### 11.1 Cost blow-up

**Symptom:** > 30 s/step. Mitigation ladder: (a) `k_latent` 4→2;
(b) `T_blocks` 2→1 (defeats interleaving, isolates recurrence as
sole variable); (c) abandon true Coconut recurrence — fall back to
"parallel-within-block" (each block emits k latents in one pass,
recurrence only across blocks; passes drop 10→4). (c) is strictly
weaker but recovers something.

### 11.2 Gradient explosion through repeated forward passes

**Symptom:** `grad.norm() > 100` on any trainable param, or NaN in
loss. Existing `clip_grad_norm_(1.0)` is first line. If clipping
saturates every step, suspect bf16 instability through the recurrence:
switch the recurrence path to fp32 (store `h_prev` in fp32, cast to
bf16 at concat into `inputs_embeds`). ~2× memory on
recurrence-participating activations.

### 11.3 Trace template overfit

**Symptom:** §10.3 passes on the same template family but a
rephrased-cue control shows no improvement. **Acceptable for
feasibility POC** — no generalization claim. Document as a control to
run before over-reading the NLL improvement; promote to a hard
requirement at the next iteration with bootstrapped traces (§5 (b)).

### 11.4 Recurrence graph break

**Symptom:** `new_emb.grad[1]` or `[2]` exactly zero after step 1, OR
LoRA in the first transformer block has exactly-zero grad while later
layers are nonzero. Cause: a `.detach()` or `no_grad()` in the inner
loop. Diagnose with `torch.autograd.set_detect_anomaly(True)`. Hard
kill if not resolved within 1 day — the mechanism is non-functional
without the cross-pass graph.

---

## 12. What this POC explicitly does NOT do

- No cluster submission (per `MEMORY.md`: no agent-initiated job submission).
- No 7B model — Qwen2.5-VL-3B only.
- No multi-reader (R=1, generator-shared).
- No teacher-bootstrapped traces — hand-written templates only.
- No mode-switching head — segmentation is teacher-forced.
- No DDP / multi-GPU.
- No round-3 5-cell sweep, no steering, no random-control, no 5K stress.
- No reader-transfer evaluation.
- No text-NLL on the trace's teacher-forced text (§6.5).
- No inference / decoding pipeline — training-time gradient check only.

If §10's three thresholds pass, the next iteration scopes follow-on
work along the deferred axes. If §11.4 fires, the POC's job is to
surface that wiring failure within the 2 h window.
