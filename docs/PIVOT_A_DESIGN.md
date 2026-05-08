# Pivot A — LVR + collapse-prevention regularizer (Experiment C design)

## 1. Problem statement

Phase 1 (mean-MSE, λ=1.0) and Phase 1.5 (sum-MSE, λ=1.0) both produce
**MARGINAL 2/4** verdicts on the 5K Visual-CoT 3B run. The dispositive
failure mode is *mean off-diagonal cosine across the K=8 latent slots*:
0.851 (Phase 1) → 0.987 (Phase 1.5) → 0.737 (Phase 2). The Monet stage-2
target is 0.38, and Phase 1's *targets themselves* (post-projector ROI
patches) sit at ≈0.46 mean off-diag cosine — i.e. the supervision signal
already has enough angular spread, the LLM just isn't using it. Phase 1.5's
loss-form sweep showed magnitude is not the cause: across ~4 orders of
LVR pressure the latents still collapse onto roughly one direction (see
`phase1_lvr/REPORT_SUMSSE.md`). Pure per-position MSE has no incentive
against many-to-one mappings; the LLM minimizes ‖h_t − v_t‖² by pulling
all h_t toward the *mean* of v_t (or the dominant component), which is
the cheapest collapsed solution. We need a second loss term whose minimum
is *not* compatible with h_t ≈ const.

## 2. Regularizer options surveyed

`h ∈ R^{B×K×D}` with K=8 latent rows per example, D=2048 (Qwen2.5-VL-3B).
Notation: `h_norm = h / ‖h‖₂` along D; `K=8`, `B≈grad_accum`.

### (a) Pairwise cosine hinge — `cos_penalty_loss`
```python
hn = F.normalize(h, dim=-1)              # [B,K,D]
C  = hn @ hn.transpose(-1,-2)            # [B,K,K]
off = C - torch.eye(K)[None]             # zero diagonal
L_cos = (off.clamp(min=tau) - tau).pow(2).mean()  # only push pairs above tau
```
- **Pros:** directly targets the failure metric (off-diag cosine). Hinge
  at τ allows the *natural* Monet-stage-2-like spread (0.38) without
  forcing artificial orthogonality. Cheap (K²=64 dot products / example).
- **Cons:** no per-dimension constraint — could collapse onto a 2-D plane
  with K vectors anti-aligned. K=8 in D=2048 makes that unlikely but
  possible.
- **Expected behavior:** mean off-diag cos drives toward τ; if τ is too
  low (e.g. 0.0) competes destructively with LVR (since v_roi targets sit
  at 0.46). Use τ ∈ {0.4, 0.5}.

### (b) VICReg variance + covariance — `vicreg_loss`
Bardes et al. 2022. The K stack is the "batch" dimension here.
```python
H = h.reshape(-1, D)                     # [B*K, D]
H = H - H.mean(dim=0, keepdim=True)
std = (H.var(dim=0) + 1e-4).sqrt()       # [D]
L_var = F.relu(gamma - std).mean()       # gamma=1.0 in paper
cov = (H.T @ H) / (H.shape[0] - 1)       # [D, D]
off = cov - torch.diag(torch.diagonal(cov))
L_cov = off.pow(2).sum() / D
L_vic = mu * L_var + nu * L_cov          # mu=25, nu=1 (paper)
```
- **Pros:** principled, well-cited collapse prevention; addresses both
  per-dim variance starvation AND inter-dim redundancy.
- **Cons:** designed for SSL projection heads with batch≫D; here B*K≈32 ≪
  D=2048, so the empirical covariance is rank-deficient and noisy. The
  variance term may be the only useful piece at our scale. The variance
  threshold γ depends on activation magnitude (Qwen residual stream
  ~50, not ~1) — γ must be tuned, not blindly imported.
- **Expected behavior:** variance term pulls every dim away from constant;
  covariance term decorrelates dims — both indirectly raise off-diag
  cosine spread.

### (c) Hard orthogonality — `ortho_penalty_loss`
```python
hn = F.normalize(h, dim=-1)              # [B,K,D]
G  = hn @ hn.transpose(-1,-2)            # [B,K,K] Gram of unit rows
I  = torch.eye(K, device=h.device)[None]
L_orth = (G - I).pow(2).sum(dim=(-1,-2)).mean() / (K*(K-1))
```
- **Pros:** strongest mathematical guarantee against collinearity; closed
  form, no hyperparameter besides λ.
- **Cons:** target geometry is *too aggressive* — Monet stage-2 sits at
  cos=0.38, not 0; hard ortho would over-correct and likely fight LVR
  (whose targets are at cos=0.46). Risk of NTP starvation analogous to
  Phase 1.5's sum-MSE issue.

### (d) Within-batch InfoNCE — `nce_h_v_loss`
Pull h_t to v_roi_t, push h_t away from h_{t'≠t} of the same example.
```python
hn = F.normalize(h, dim=-1).reshape(-1, D)       # [BK, D]
vn = F.normalize(v, dim=-1).reshape(-1, D)       # [BK, D]
logits = (hn @ vn.T) / temperature               # [BK, BK]
labels = torch.arange(BK)
L_nce = F.cross_entropy(logits, labels)
```
- **Pros:** unifies invariance + repulsion in one term; well-understood.
- **Cons:** replaces (rather than complements) the LVR MSE — changes the
  experimental baseline so we can no longer claim "LVR + reg" decomposition.
  Also K=8 is a tiny negative pool; quality of negatives matters.

### (e) Cosine on per-token *deviations from the mean* — `dev_cos_loss`
A novel-but-simple variant: penalize cos similarity AFTER subtracting the
per-example slot mean. Targets *the failure mode signature directly*:
Phase 1's collapse is "all slots ≈ same direction", i.e. zero deviation.
```python
mu = h.mean(dim=1, keepdim=True)         # [B,1,D]
d  = h - mu                              # [B,K,D] — purely the spread
dn = F.normalize(d, dim=-1)
C  = dn @ dn.transpose(-1,-2)            # [B,K,K]
L_dev = (C - torch.eye(K)[None]).pow(2).sum(dim=(-1,-2)).mean() / (K*(K-1))
```
Equivalent to ortho on the centered residuals; allows a shared component
(common context, attended to by all slots) while forcing the slot-specific
deltas to span the K-dim subspace. Cheap, and the geometry matches the
diagnostic.

## 3. Recommended variants for testing

We pick **two complementary variants** that test distinct hypotheses,
keep the LVR baseline intact, and stay implementable in <30 lines:

### Variant **C1 — pairwise cosine hinge** (option a)
- **Hypothesis:** the collapse metric and the loss target are the *same
  quantity*; the simplest fix is to penalize it directly with slack
  (τ=0.5) so the model can match v_roi geometry (cos=0.46) without
  fighting the regularizer.
- **Lambda starting point:** λ_reg = 1.0 (loss is a unitless cosine²).
  Sweep grid (only if first run is borderline): λ_reg ∈ {0.5, 1.0, 2.0}, τ=0.5 fixed.
- **Why over (b)/(c)/(d):** directly tests the failure metric; minimal
  recipe deviation from Phase 1; failure to PASS would *cleanly* implicate
  attention topology (Experiment B) as the load-bearing piece.

### Variant **C2 — VICReg variance + covariance** (option b)
- **Hypothesis:** collapse is a *representation pathology* (low variance
  + high inter-dim redundancy), not a geometric one — fixing it at the
  representation level is more robust than penalizing one statistic.
- **Lambda starting point:** μ_var=1.0, ν_cov=0.04, γ=1.0 in **z-scored
  hidden space**. We z-score `h` per-dim using a running batch estimate
  (see code sketch) to decouple γ from Qwen's ~50-norm residual stream.
  Single sweep if needed: μ_var ∈ {0.5, 1.0, 2.0}, ν_cov fixed.
- **Why over (c)/(d)/(e):** principled, established in SSL literature;
  the variance term is exactly what addresses "low-rank single-direction
  collapse" without imposing artificial geometry. (c) is too rigid; (d)
  is too far from the LVR baseline; (e) is interesting but unproven.

We deliberately train **C1 and C2 separately**, not jointly — Experiment C
needs a clean attribution (which mechanism worked), not a maximally tuned
recipe.

## 4. Code sketch

Drop into `phase1_lvr/loss.py` (or a new `phase1_lvr/reg.py`):

```python
# --- Variant C1 ---
def cos_penalty_loss(h: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Pairwise off-diagonal cosine hinge. h: [B, K, D]."""
    if h.dim() == 2:                    # [K, D] -> [1, K, D]
        h = h.unsqueeze(0)
    B, K, D = h.shape
    hn = torch.nn.functional.normalize(h.float(), dim=-1)
    C = hn @ hn.transpose(-1, -2)       # [B, K, K]
    eye = torch.eye(K, device=h.device, dtype=hn.dtype).unsqueeze(0)
    off = C - eye                       # zero diagonal
    excess = (off - tau).clamp(min=0)   # only penalize pairs above tau
    return excess.pow(2).sum(dim=(-1, -2)).mean() / (K * (K - 1))

# --- Variant C2 ---
def vicreg_loss(
    h: torch.Tensor,
    var_weight: float = 1.0,
    cov_weight: float = 0.04,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg variance+covariance on z-scored h. h: [B, K, D].

    z-score per-dim across the (B*K) sample axis, then variance hinge at
    gamma=1.0 is meaningful regardless of activation magnitude.
    """
    if h.dim() == 2:
        h = h.unsqueeze(0)
    B, K, D = h.shape
    H = h.reshape(B * K, D).float()
    H = H - H.mean(dim=0, keepdim=True)
    std = (H.var(dim=0, unbiased=True) + eps).sqrt()
    H_z = H / std                                            # z-scored
    L_var = torch.nn.functional.relu(gamma - std).mean()
    cov = (H_z.T @ H_z) / max(B * K - 1, 1)                  # [D, D]
    diag = torch.diagonal(cov)
    L_cov = (cov.pow(2).sum() - diag.pow(2).sum()) / D
    return var_weight * L_var + cov_weight * L_cov
```

Trainer wiring (in `phase1_lvr/trainer.py::lvr_step`, after computing `lvr`):
```python
if reg_kind == "cos":
    reg = cos_penalty_loss(h.unsqueeze(0).float(), tau=cfg["reg_tau"])
elif reg_kind == "vicreg":
    reg = vicreg_loss(h.unsqueeze(0).float(),
                      var_weight=cfg["reg_var_w"],
                      cov_weight=cfg["reg_cov_w"],
                      gamma=cfg["reg_gamma"])
else:
    reg = torch.zeros((), device=h.device)
total = ntp_loss + lambda_lvr * lvr + lambda_reg * reg
```
Config additions to `lvr_3b_5k.yaml` (clone to two new files):
```yaml
# pivot_a/configs/pivot_a_cos.yaml
reg_kind: cos
lambda_reg: 1.0
reg_tau: 0.5

# pivot_a/configs/pivot_a_vicreg.yaml
reg_kind: vicreg
lambda_reg: 1.0
reg_var_w: 1.0
reg_cov_w: 0.04
reg_gamma: 1.0
```
Everything else (lr, schedule, λ_LVR=1.0 mean-form, K=8, 1000 steps,
5K examples, seed=0) is identical to `phase1_lvr/configs/lvr_3b_5k.yaml`.

## 5. Acceptance + comparison plan

- **Held-out eval:** identical 200-example held-out set used in
  `phase1_lvr` ablation (`phase1_lvr/ablation_runner.py`). No data drift.
- **Reader:** `phase1_self` (ours) and `qwen_base` (frozen Qwen2.5-VL-3B),
  same as Phase 1.
- **Acceptance (PASS = 3 of 4):**
  | criterion | target |
  |---|---|
  | `compression_ratio` | ≥ 0.4 |
  | `mean_off_diag_cos` | ≤ 0.55 (**dispositive**) |
  | `n_helpful` (margin ≥ 0.05 nat) | ≥ 3 |
  | `qwen_base utility` | > 0 |

- **Decision rule:** mean cos ≤ 0.55 alone qualifies the variant for
  cluster-scale Phase 3; mean cos > 0.7 = same failure mode as Phase 1.
  The narrow band 0.55–0.7 is "improved but not target-geometry"; treat
  as MARGINAL.
- **Expected results:**
  - **C1 (cos hinge):** mean cos converges to ≈τ=0.5 by construction;
    PASS gate is whether `n_helpful` and `utility` follow. Predicted:
    mean cos 0.45–0.55, n_helpful 3–5, utility ~+0.1 nat.
  - **C2 (VICReg):** less directly targeted; predicted: mean cos
    0.5–0.7, n_helpful 2–4, utility ~+0.05–0.15 nat. If C2 PASSes
    but C1 doesn't, the failure mode is per-dim variance starvation, not
    geometry — informs Phase 3 choice.
  - **Both FAIL:** 3B/5K is too small for distributed latents under any
    direct regularization → escalate to cluster Phase 3 with Monet
    stage-2's full attention-mask topology (matches Experiment B path
    in the day plan).
- **Comparability:** runs land in `pivot_a/results/{cos,vicreg}/` and
  feed the same `phase1_lvr/build_report.py` so all numbers are directly
  cross-tabulable with Phase 1, 1.5, 2.

## 6. Sources

- [VICReg (Bardes, Ponce, LeCun, ICLR 2022) — arXiv 2105.04906](https://arxiv.org/abs/2105.04906) — variance hinge prevents per-dim collapse; covariance term decorrelates redundancy; defaults λ=μ=25, ν=1, γ=1, ε=1e-4.
- [VICReg official PyTorch implementation](https://github.com/facebookresearch/vicreg) — exact code: `relu(1 - std)` for variance; `off_diagonal(cov).pow(2).sum() / D` for covariance.
- [Barlow Twins (Zbontar et al., ICML 2021) — arXiv 2103.03230](https://arxiv.org/abs/2103.03230) — alternative redundancy-reduction loss via cross-correlation matrix → identity; complementary intuition for why decorrelation is collapse-preventive.
- [Orthogonality regularization in deep nets (NeurIPS 2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/bf424cb7b0dea050a42b9739eb261a3a-Paper.pdf) — the `‖WW^T − I‖_F²` family; rejected here as too rigid for our geometry.
- [DINO (Caron et al., 2021) centering+sharpening](https://arxiv.org/abs/2104.14294) — centering subtracts running mean to break trivial-constant collapse; the dev-cos variant (option e) is in this spirit but applied to slot residuals rather than teacher outputs.
- [Phase 0 REPORT.md](../phase0_monet_probe/REPORT.md) — Monet stage-2 mean off-diag cos = 0.38 is the natural target geometry; Phase 1 v_roi targets sit at 0.46, so τ=0.5 is well-calibrated.
- [Phase 1 REPORT.md](../phase1_lvr/REPORT.md) and [REPORT_SUMSSE.md](../phase1_lvr/REPORT_SUMSSE.md) — failure mode is robust across 4 orders of LVR pressure → magnitude is not the lever; representational structure of the loss must change.
