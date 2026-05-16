# Monet Reproduction — Cross-Session Findings Log

Append-only. Each entry: **date — config tested → metric → verdict → next action**.
This file is the ONLY channel between Session A (local repro) and Session B (cluster
strategy). Keep entries terse and decision-grade. Newest at the bottom.

---

## Seed state (2026-05-16) — known before Session A starts

Internal-probe results (vendored Monet `latent_mode` forward via `cluster/eval.py`,
n=200, A6000). This probe is a *relative* collapse detector, NOT the paper's metric.

| ckpt | type | mean off-diag cos | qwen_base utility | reading |
|---|---|---:|---:|---|
| `Monet-SFT-7B/stage2` (released) | monet | **0.377** | **+2.05** | healthy; ≈ local-3B Monet S2 (0.375) |
| our `stage2_repro_step1500` | monet | 0.840 | −5.35 | partial collapse, harmful latents |
| our `stage1_sft_step1100` | pivot | 0.000 | 0.000 | no latent mechanism (pre-alignment, by design) |

**Established:** released `stage2` weights are healthy on an independent path → the
checkpoint is fine; the failure surface is the **vLLM inference configuration**.

**NOT established (Session A's job):**
- Which released ckpt (`stage2`/`stage3`/RL `Monet-7B`) is the paper's V\*=82.20 row.
- Correct inference-time `LATENT_SIZE` (training used 8; inference README says 10).
- Whether the vLLM path supports SFT ckpts at all or only RL `Monet-7B`.

**Prior failed attempt:** VLMEvalKit+patch on `stage2` @ K=8 → V\*=23.04%, 0/191
`\boxed{}`. Monet's own `vllm_inference_example.py` verbatim on same ckpt+K → identical
degenerate garbage. Does NOT isolate upstream-breakage from our-config-error (shares the
two unverified assumptions above).

**Open experiments:** E1 mapping archaeology (no GPU) → E2 validate pipeline on
documented ckpt+K → E3 LATENT_SIZE sweep → E4 transformers-native fallback. See
`MONET_REPRO_BRIEF.md` §4.

---

## Findings (Session A appends below)
