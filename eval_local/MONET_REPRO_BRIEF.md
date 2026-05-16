# Monet Reproduction — Session A Kickoff Brief

> **You are the local Monet-reproduction agent.** Self-contained brief — you have no prior
> conversation context. Read this top to bottom, then START HERE (bottom section).
> Run everything on the **local A6000**. Do **not** touch the cluster, HF artifacts, or
> `visual-latents` project code (`cluster/`, `slurm/`, `scripts/`). Run from the **main
> `/mnt/ssd/Projects/visual-latents` checkout**, NOT a `.claude/worktrees/` copy.

---

## 1. Mission

Determine whether Monet's **paper V\* benchmark number is reproducible with their released
code + checkpoint**, and — equally important — whether earlier reproduction failures were a
**configuration error on our side** rather than upstream breakage.

This is the *faithful-baseline* gate for the whole project. The cluster track is blocked on
trusting (or correctly distrusting) our eval. Your output directly decides cluster strategy.

**Paper targets (released checkpoint, "Monet-SFT" row):**
- V\* = **82.20**
- HRBench4K = 68.50
- MME-RealWorld-Lite = 52.68

Primary objective: reproduce **V\*** to within a few points of 82.20 on *some* released
checkpoint with *some* documented config. Secondary benchmarks only after V\* is cracked.

---

## 2. What is already known (do not re-derive)

### 2a. The released checkpoint is almost certainly NOT broken

An independent, working code path — the vendored Monet model's `latent_mode=True` forward,
used by `cluster/eval.py` — was run on the released `stage2/` checkpoint and produced
**healthy, paper-consistent latent geometry**: mean off-diag cos **0.377**, qwen_base
utility **+2.05** (see `eval_local/monet_paper_stage2/REPORT.md`). A broken or wrong-config
checkpoint could not produce healthy latents on an independent path.

**Conclusion: the checkpoint weights are fine. The failure is in the vLLM *inference
configuration*, which is the most-likely-our-error surface. Treat "we misconfigured
inference" as the leading hypothesis, not "the paper is irreproducible."**

### 2b. What was tried and failed

VLMEvalKit + Monet's vLLM monkey-patch was run on released `stage2/` at `LATENT_SIZE=8`:
- Without `enforce_eager=True`: CJK garbage (CUDA graphs discard dynamic latent injection).
- With `enforce_eager=True` + Monet SamplingParams: coherent English but **V\* = 23.04%**,
  0/191 answers with `\boxed{}`, degenerate repetition.
- Monet's *own* `inference/vllm_inference_example.py` run verbatim (their code, their
  prompt, released `stage2/`, `LATENT_SIZE=8`, `vllm==0.10.0`) → same `CAS CAS CAS`
  degenerate garbage.

**Why this is NOT yet proof of upstream breakage:** that verbatim test still used
`stage2/` at `LATENT_SIZE=8` — it shares the two unverified assumptions below. It did not
isolate upstream breakage from our config error.

### 2c. The ranked open questions (this is your work)

1. **Checkpoint ↔ paper-row mapping is UNVERIFIED (highest risk).** "`stage2/` = the
   paper's V\*=82.20 row" was *inferred*, never confirmed. The Monet README inference
   section references the **RL `Monet-7B`** model, not the SFT checkpoints. Running the
   vLLM path on `stage2/` may be garbage *by design*.
2. **`LATENT_SIZE` is UNVERIFIED.** We used **8** (from the *training* script). The
   inference README uses **10**. The released ckpt's correct *inference-time* latent size
   is unknown. Wrong K → exactly the degenerate decode observed.
3. **The vLLM path may only support RL `Monet-7B`/`stage3`**, not SFT `stage2`.
4. Genuine upstream breakage — *least* likely given 2a.

---

## 3. Resources & locations (VERIFY each before relying on it — paths may be stale)

| What | Path | Notes |
|---|---|---|
| Released ckpts (local) | `/mnt/ssd/Projects/visual-latents/phase0_monet_probe/checkpoints/Monet-SFT-7B/` | `stage2/` confirmed present; check for `stage3/`. HF repo `NOVAglow646/Monet-SFT-7B` (has `stage2/`,`stage3/`, no `stage1/`). RL model = HF `Monet-7B` (separate repo — locate/confirm). |
| VLMEvalKit | `/mnt/ssd/Projects/VLMEvalKit/` | `sitecustomize.py` = the monkey-patch (forces `enforce_eager=True` + Monet engine kwargs; forces Monet SamplingParams: temp=0.1, top_k=50, top_p=0.8, rep_pen=1.01, max_tokens=4096, skip_special_tokens=False). `Monet_models/monet_gpu_model_runner.py`. |
| VLMEvalKit config | `/mnt/ssd/Projects/Monet-eval/monet_vstar_stage2.json` | class=Qwen2VLChat, use_vllm=true, max_pixels=8192*28*28, system prompt = Monet README §Note. |
| Eval venv | `/mnt/ssd/Projects/Monet-eval/.venv` | vllm 0.10.0, torch 2.7.1+cu126, transformers 4.54.0, trl 0.15.2, Python 3.10. LATENT_START_ID=151666 / END=151667. Set `VLLM_USE_FLASHINFER_SAMPLER=0` (no nvcc on box). |
| Upstream Monet repo | **LOCATE & RECORD** | Has `inference/vllm_inference_example.py`, `inference/apply_vllm_monet.patch()`. Likely under `/mnt/ssd/Projects/Monet*` or `phase0_monet_probe/`. Find it, confirm the README inference section + which HF model it targets. |
| Paper PDF/appendix | **LOCATE & RECORD** | Search `/mnt/ssd/Projects/` for the Monet paper. The appendix/eval-protocol section is the authoritative source for ckpt↔row mapping and inference `LATENT_SIZE`. |

---

## 4. Decisive experiments (cheap → expensive; record each in the LOG)

- **E1 — Mapping archaeology (no GPU, do first).** From the *paper appendix*, the upstream
  *repo README/eval scripts*, and the *HF model card*: which released folder
  (`stage2`/`stage3`/`Monet-7B`) corresponds to the paper's V\*=82.20 "Monet-SFT" row, and
  what inference-time `LATENT_SIZE` does the eval protocol specify? Resolve open Qs 1 & 2 on
  paper before burning GPU.
- **E2 — Validate the pipeline on the documented config.** Run the existing
  VLMEvalKit+patch pipeline on the checkpoint the inference docs actually target (likely RL
  `Monet-7B` and/or `stage3/`) at the documented `LATENT_SIZE` (likely 10). **If V\* lands
  ~82, the pipeline is VALIDATED** and the prior failure was wrong-ckpt/wrong-K — a config
  error on our side, exactly what the user wants to know.
- **E3 — `LATENT_SIZE` sweep.** Run Monet's own `inference/vllm_inference_example.py` on
  released `stage2/` across `LATENT_SIZE ∈ {4,8,10,12}`. Cheap, single-image, isolates K.
- **E4 — transformers-native fallback (only if vLLM path stays broken).** Build a minimal
  V\* harness around the vendored Monet `latent_mode` forward (the path that already works
  in `cluster/eval.py` and probes the released ckpt healthy). Removes vLLM as a variable
  entirely.

Stop and report as soon as V\* reproduces (~within a few points of 82.20) on any
checkpoint+config — that answers the mission. Note: rule-based-only judging (no secondary
LLM judge) costs ~1–2 pts and CANNOT explain a ~60-pt gap, so don't chase the judge.

---

## 5. Constraints

- **A6000 / local only.** Never submit cluster jobs, never touch cluster state, HF
  artifacts, or `visual-latents` code under `cluster/ slurm/ scripts/`. You may read repo
  code freely and write under `eval_local/`, `/mnt/ssd/Projects/Monet-eval/`,
  `/mnt/ssd/Projects/VLMEvalKit/`.
- Run from the **main `/mnt/ssd/Projects/visual-latents` checkout** (Session B holds a
  worktree; stay out of it to avoid git contention).
- **Verify before asserting** (project core rule): never trust a path/API/behavior from
  this brief or from memory without checking it live. When execution contradicts the brief,
  execution wins — and fix the brief.

---

## 6. Interface contract — the ONLY cross-session channel

Append every result to **`/mnt/ssd/Projects/visual-latents/eval_local/MONET_REPRO_LOG.md`**
as a dated entry: *config tested → metric → verdict → next action*. Session B (cluster
strategy) reads only that log, never your chat. Keep entries terse and decision-grade. Do
not rely on shared conversation context — the log is the contract.

---

## 7. START HERE

1. Read `eval_local/MONET_REPRO_LOG.md` (current known state seeded there).
2. Run **E1** (mapping archaeology) — pure reading, no GPU. Locate the paper PDF and the
   upstream Monet repo first; record their paths in the LOG.
3. Append the E1 verdict to the LOG (which ckpt = which paper row, at which `LATENT_SIZE`).
4. Proceed to **E2** with the now-correct ckpt+config. Append result.
5. Continue E3/E4 only as needed. Report back to the user when V\* reproduces or when all
   four experiments are exhausted with a clear verdict.
