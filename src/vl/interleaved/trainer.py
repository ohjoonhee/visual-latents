"""Interleaved variant training loop.

Mirrors `vl.trainers.sft_anchor.train` but per-step does:
  1. Sample a template + (object, answer) -> Trace.
  2. Run `run_interleaved_forward` to get h ∈ [1, K_total, D] from the
     Coconut-recurrent forward (10 model calls for T_blocks=2, k_latent=4).
  3. Splice h into the reader's K_total `<|image_pad|>` slots and compute
     the standard combined loss (NLL_multi + concept + norm).
  4. Backward, clip, step.

POC scope (per `docs/INTERLEAVED_LATENT_DESIGN.md`): R=1, B=1, hand-written
templates, single GPU, no DDP, no checkpointing, no wandb-by-default.
Held-out eval and checkpoints exist but are minimal.

Public API:
    train(cfg: Round3Config) -> None
    run_one_step(...) -> dict   # exposed for the gradient probe
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .. import paths as vl_paths
from ..losses import combined, nll_multi_anchor
from ..model import get_v_sem
from ..readers import load_anchors
from .grounding import block_grounding_loss, stage1_curriculum_weights
from .model import build_interleaved_generator, run_interleaved_forward
from .traces import TEMPLATES, sample_generic_trace, sample_trace


# ---------------------------------------------------------------------------
# Synthetic-data sampler
# ---------------------------------------------------------------------------
# POC §5.1 says 10 templates × 30 GQA images = 300 unique seeds. The probe
# and quick smoke runs use synthetic (object, answer) pairs paired with a
# solid-color PIL image to remove the GQA-loader dependency from the
# feasibility check. Once the recurrence is verified the trainer can be
# graduated to real GQA images / labels via a separate code path (out of POC).
_SYNTH_OBJECTS = [
    "red cubes", "blue spheres", "green cylinders", "yellow cones",
    "people", "cars", "animals", "trees", "houses", "tables",
]
_SYNTH_ANSWERS_BY_CAT = {
    "count": ["3", "5", "1", "7", "2"],
    "color": ["red", "blue", "green", "yellow", "white"],
    "position": ["on the left", "in the center", "on the right", "in the back", "in front"],
    "presence": ["yes", "no"],
    "attribute": ["standing", "sitting", "running", "watching", "resting"],
}


def _sample_synthetic_example(rng: random.Random, *, permute_template: bool = False):
    """Sample one (image, trace, qa_override) tuple using synthetic placeholders.

    When ``permute_template`` is False (default), returns a trace whose
    (question, answer) match the template that drove the trace text — the
    natural pairing. The reader scores against `trace.question/answer`.

    When ``permute_template`` is True (design §11.3 control), the trace text
    is built from a DIFFERENT template than the one whose (question, answer)
    will be scored. We achieve this by sampling two templates: ``tpl_qa``
    supplies the QA the reader scores, ``tpl_trace`` (a different category)
    supplies the trace text the recurrent forward consumes. The returned
    trace's ``question``/``answer`` fields are overwritten to match
    ``tpl_qa`` so the trainer's downstream code (which reads
    ``trace.question, trace.answer``) is unchanged.
    """
    template_idx = rng.randrange(len(TEMPLATES))
    cat = TEMPLATES[template_idx]["category"]
    obj = rng.choice(_SYNTH_OBJECTS)
    ans = rng.choice(_SYNTH_ANSWERS_BY_CAT[cat])

    if not permute_template:
        trace = sample_trace(obj=obj, answer=ans, rng=rng, template_idx=template_idx)
    else:
        # Pick a different category for the trace text.
        other_idxs = [i for i in range(len(TEMPLATES)) if i != template_idx]
        trace_template_idx = rng.choice(other_idxs)
        trace_cat = TEMPLATES[trace_template_idx]["category"]
        trace_obj = rng.choice(_SYNTH_OBJECTS)
        trace_ans = rng.choice(_SYNTH_ANSWERS_BY_CAT[trace_cat])
        trace = sample_trace(
            obj=trace_obj, answer=trace_ans, rng=rng,
            template_idx=trace_template_idx,
        )
        # Build the natural-pairing question/answer from the original template
        # (so the reader sees a QA that has NO surface relation to the trace text).
        nat_q = TEMPLATES[template_idx]["question_template"].format(object=obj)
        trace.question = nat_q
        trace.answer = ans
        trace.category = f"{trace_cat}->{cat}"

    # A single solid-color image — vision content is a constant; the latents
    # have no "image grounding" to learn here. That is acceptable for the POC
    # whose goal is recurrence-graph integrity, not visual reasoning quality.
    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    img = Image.new("RGB", (224, 224), color=color)
    return img, trace


# ---------------------------------------------------------------------------
# GQA sampler (round-3)
# ---------------------------------------------------------------------------
# Each pre-loaded GQA record carries `image_loader` + `questions: list[(q,a)]`
# (see src/vl/data/gqa.py). The interleaved trainer needs ONE example per step
# of the form (image, trace). For GQA we use the "generic" template
# (traces.GENERIC_TEMPLATE) — no per-category routing — so any GQA (q, a) pair
# can drive a trace.
def _sample_gqa_example(
    records: list,
    rng: random.Random,
    *,
    permute_template: bool = False,
):
    """Sample one (image, trace) tuple from a pre-loaded GQA record list.

    Args:
        records: list of GQA records as returned by `vl.data.gqa.load_gqa`.
        rng: random.Random — controls record + (q, a) selection.
        permute_template: design §11.3 control. When True, pick (q, a) from
            ONE record for the trace text and (q, a) from a DIFFERENT record
            for the reader-scored question — analogous to the synthetic
            permute, except the "trace text" here is template-only (no
            object placeholder), so the only mismatch is the (image, q, a)
            triple seen by the reader vs. the trace's image grounding. We
            implement this by sampling the trace from one record's image
            and overwriting the trace's question/answer with a (q, a) pulled
            from a different record. The reader scores against the OVERWRITTEN
            (q, a), which has no relation to the image fed to the recurrence.

    Returns:
        (PIL.Image, Trace) — same shape as `_sample_synthetic_example`.
    """
    if not records:
        raise RuntimeError("GQA records list is empty.")

    rec = rng.choice(records)
    q, a = rng.choice(rec["questions"])
    img = rec["image_loader"]()

    if not permute_template:
        trace = sample_generic_trace(question=q, answer=a)
    else:
        # Pick a DIFFERENT record for the (q, a) the reader scores. The
        # generic trace template carries no object/question placeholder so
        # the only thing we mismatch is the reader-scored pair vs. the image.
        other_recs = [r for r in records if r["image_id"] != rec["image_id"]]
        if not other_recs:
            other_rec = rec
        else:
            other_rec = rng.choice(other_recs)
        q_alt, a_alt = rng.choice(other_rec["questions"])
        trace = sample_generic_trace(question=q_alt, answer=a_alt)
        trace.category = "gqa-perm"

    return img, trace


# ---------------------------------------------------------------------------
# Shapes sampler (round-3 pivot — GQA not cached locally; build a
# programmatic dataset with REAL visual structure that the latents can
# encode. CLEVR-spirit: countable colored shapes at known positions.
# ---------------------------------------------------------------------------
_SHAPE_COLORS = {
    "red": (220, 40, 40),
    "blue": (40, 80, 220),
    "green": (40, 180, 60),
    "yellow": (240, 220, 40),
}
_SHAPE_KINDS = ["circle", "square", "triangle"]


def _draw_shape(draw: ImageDraw.ImageDraw, kind: str, color, cx: int, cy: int, r: int):
    if kind == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif kind == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=color)
    elif kind == "triangle":
        draw.polygon(
            [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color
        )
    else:
        raise ValueError(kind)


def _make_shapes_image_and_gt(rng: random.Random):
    """Generate a 224x224 PIL scene of N ∈ [1,5] colored shapes + ground truth.

    GT dict carries:
        n_total, by_color: {color: count}, by_shape: {shape: count},
        per_shape: list of (color, kind, cx, cy, r) for richer probes.
    """
    img = Image.new("RGB", (224, 224), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    n = rng.randint(1, 5)
    placed: list[tuple[str, str, int, int, int]] = []
    for _ in range(n):
        color_name = rng.choice(list(_SHAPE_COLORS))
        kind = rng.choice(_SHAPE_KINDS)
        r = rng.randint(18, 32)
        cx = rng.randint(r + 4, 224 - r - 4)
        cy = rng.randint(r + 4, 224 - r - 4)
        _draw_shape(draw, kind, _SHAPE_COLORS[color_name], cx, cy, r)
        placed.append((color_name, kind, cx, cy, r))
    by_color: dict[str, int] = {}
    by_shape: dict[str, int] = {}
    for c, k, *_ in placed:
        by_color[c] = by_color.get(c, 0) + 1
        by_shape[k] = by_shape.get(k, 0) + 1
    return img, {
        "n_total": n,
        "by_color": by_color,
        "by_shape": by_shape,
        "per_shape": placed,
    }


def _qa_from_shapes_gt(rng: random.Random, gt: dict) -> tuple[str, str, str, str]:
    """Pick a question type given the ground truth. Returns (q, a, category, object).

    Question types are aligned to the existing TEMPLATES so we can reuse
    the count / color / presence templates without per-category routing.
    """
    qtype = rng.choice(["count_total", "count_color", "presence_color_kind", "color_of_kind"])

    if qtype == "count_total":
        return (
            f"How many shapes are there in the image?",
            str(gt["n_total"]),
            "count",
            "shapes",
        )
    if qtype == "count_color":
        # Ask about a color that may or may not be present (so 0 is a valid answer).
        color = rng.choice(list(_SHAPE_COLORS))
        cnt = gt["by_color"].get(color, 0)
        return (
            f"How many {color} shapes are there in the image?",
            str(cnt),
            "count",
            f"{color} shapes",
        )
    if qtype == "presence_color_kind":
        color = rng.choice(list(_SHAPE_COLORS))
        kind = rng.choice(_SHAPE_KINDS)
        present = any(c == color and k == kind for c, k, *_ in gt["per_shape"])
        return (
            f"Is there a {color} {kind} in the image?",
            "yes" if present else "no",
            "presence",
            f"{color} {kind}",
        )
    # color_of_kind — only well-defined when there is exactly one shape of that kind
    kinds_with_one = [k for k, c in gt["by_shape"].items() if c == 1]
    if kinds_with_one:
        kind = rng.choice(kinds_with_one)
        color = next(c for c, k, *_ in gt["per_shape"] if k == kind)
        return (
            f"What color is the {kind} in the image?",
            color,
            "color",
            kind,
        )
    # Fallback: ask total count (always well-defined).
    return (
        f"How many shapes are there in the image?",
        str(gt["n_total"]),
        "count",
        "shapes",
    )


def _category_to_template_idx(cat: str) -> int:
    for i, t in enumerate(TEMPLATES):
        if t["category"] == cat:
            return i
    raise ValueError(f"no template for category {cat!r}")


def _sample_shapes_multiq_example(rng: random.Random, K_q: int, *, permute_template: bool = False):
    """Generate ONE shapes scene + K_q DIFFERENT (q, a) pairs from its GT.

    Returns: (image, trace, extra_qa_pairs) where:
      - image: PIL.Image (the scene, OR scene-B's image if permuted).
      - trace: Trace built from (q_0, a_0); used for the recurrent forward.
      - extra_qa_pairs: list[(q_i, a_i)] of length K_q-1 — additional (q, a)
        pairs from the SAME scene's GT for the reader to score against the
        same h. The reader scores `[trace.(q,a)] + extra_qa_pairs`, totalling
        K_q questions. h must encode image-general content to serve all of them.

    For perm: as in `_sample_shapes_example`, the image fed to the recurrence
    is scene-B's, while ALL K_q (q, a) pairs come from scene-A. The (q, a) the
    reader scores has no relation to the pixels; if the model still trains
    well, it's not using the latents for image content.
    """
    img_a, gt_a = _make_shapes_image_and_gt(rng)
    # Sample K_q distinct (q, a) pairs by retrying when we get duplicates.
    seen_q: set[str] = set()
    qa_pairs: list[tuple[str, str]] = []
    attempts = 0
    while len(qa_pairs) < K_q and attempts < 4 * K_q:
        attempts += 1
        q, a, cat, obj = _qa_from_shapes_gt(rng, gt_a)
        if q not in seen_q:
            seen_q.add(q)
            qa_pairs.append((q, a))
    # If we couldn't find K_q distinct ones (e.g., scene only supports 1
    # well-defined question), pad by cycling.
    while len(qa_pairs) < K_q:
        qa_pairs.append(qa_pairs[len(qa_pairs) % max(1, len(qa_pairs))])

    q0, a0 = qa_pairs[0]
    # Build trace using the first pair's category-aligned template. We re-derive
    # cat from q0 by trying each category — fall back to count.
    if "color is" in q0:
        cat = "color"
    elif "Is there" in q0:
        cat = "presence"
    else:
        cat = "count"
    obj = "shapes"  # generic placeholder; actual q is the prompt
    template_idx = _category_to_template_idx(cat)
    trace = sample_trace(obj=obj, answer=a0, rng=rng, template_idx=template_idx)
    from .traces import _build_prompt_with_image
    trace.prompt_text = _build_prompt_with_image(q0)
    trace.question = q0
    trace.answer = a0

    extra_qa = qa_pairs[1:]

    if not permute_template:
        return img_a, trace, extra_qa

    # Perm: image fed to recurrence is from a different scene; all (q, a) come
    # from scene-A. h sees the wrong image; reader is still asked scene-A's
    # questions. If h carries image content, NLL should NOT improve as much.
    img_b, _gt_b = _make_shapes_image_and_gt(rng)
    trace.category = f"shapes-perm:{cat}-multiq"
    return img_b, trace, extra_qa


def _sample_shapes_example(rng: random.Random, *, permute_template: bool = False):
    """Generate a shapes scene + matching (q, a) on a category-aligned template.

    Permutation control: generate TWO independent scenes; the trace is built
    from scene-A's (q, a, template), but the image fed to the recurrence is
    scene-B's. The reader scores against scene-A's (q, a) — which has no
    relation to scene-B's pixels. This is a stronger control than the
    synthetic version because the (q, a) is genuinely image-dependent.
    """
    # Scene A — drives (q, a, trace text).
    img_a, gt_a = _make_shapes_image_and_gt(rng)
    q, a, cat, obj = _qa_from_shapes_gt(rng, gt_a)
    template_idx = _category_to_template_idx(cat)
    trace = sample_trace(obj=obj, answer=a, rng=rng, template_idx=template_idx)
    # Override the prompt's question so it matches the image's actual content
    # (the synthetic template's `question_template` uses {object}; it already
    # produced the right question, but for "count_color" we want the natural-
    # language form). Use the q we built from gt directly:
    from .traces import _build_prompt_with_image
    trace.prompt_text = _build_prompt_with_image(q)
    trace.question = q
    trace.answer = a

    if not permute_template:
        return img_a, trace

    # Scene B — different image, fed to the recurrence; the reader still
    # scores scene-A's (q, a). The GT mismatch is genuine image content.
    img_b, _gt_b = _make_shapes_image_and_gt(rng)
    trace.category = f"shapes-perm:{cat}"
    return img_b, trace


# ---------------------------------------------------------------------------
# Visual-CoT sampler (round-4 — see docs/INTERLEAVED_DATASET_RECON.md)
# ---------------------------------------------------------------------------
# Returns (image, trace, extra_qa, vsem_full, vsem_crop) for one training step.
# The viscot dataset is loaded once at trainer init via vl.data.viscot.ViscotDataset.
# Per-image multi-Q grouping enables K_q-different sampling for q-invariance.
def _sample_viscot_example(
    viscot,
    rng: random.Random,
    K_q: int,
    *,
    permute_template: bool = False,
):
    """Sample one (image, trace, extras, vsem_full, vsem_crop) tuple.

    Args:
        viscot: vl.data.viscot.ViscotDataset instance.
        rng: random.Random.
        K_q: number of (q, a) pairs to sample for the same image (multi-Q).
        permute_template: when True, image and (q, a) come from DIFFERENT
            (source, image_id) groups. The trace text is built from group-B's
            first (q, a); the image fed to the recurrence is from group-A.
            V_sem targets follow the IMAGE (group-A), so the grounding signal
            pulls h toward image-A's content while reader-NLL is mismatched.

    Returns: (PIL image, Trace, extra_qa: list[(q, a)], vsem_full, vsem_crop).
    """
    if not viscot.image_keys:
        raise RuntimeError("viscot dataset is empty")

    if not permute_template:
        # Natural: one image group, K_q distinct questions about it
        key = rng.choice(viscot.image_keys)
        idxs = viscot.groups[key]
        if len(idxs) >= K_q:
            chosen = rng.sample(idxs, K_q)
        else:
            # Cycle to fill K_q
            chosen = list(idxs) + [idxs[i % len(idxs)] for i in range(K_q - len(idxs))]
        primary = viscot.get_record(chosen[0])
        extra_qa = []
        for i in chosen[1:]:
            m = viscot.get_meta(i)
            extra_qa.append((m["question"], m["answer"]))
        trace = sample_generic_trace(
            question=primary["question"], answer=primary["answer"],
        )
        trace.category = f"viscot:{primary['source']}"
        return primary["image"], trace, extra_qa, primary["vsem_full"], primary["vsem_crop"]

    # Perm: image from group-A, (q, a) from group-B (different image_id)
    key_a = rng.choice(viscot.image_keys)
    key_b = key_a
    while key_b == key_a:
        key_b = rng.choice(viscot.image_keys)

    # Pull metadata-only for the candidate qa pool
    idxs_b = viscot.groups[key_b]
    if len(idxs_b) >= K_q:
        chosen_b = rng.sample(idxs_b, K_q)
    else:
        chosen_b = list(idxs_b) + [idxs_b[i % len(idxs_b)] for i in range(K_q - len(idxs_b))]

    # Materialise A only (we need the image and its V_sem)
    rec_a = viscot.get_record(viscot.groups[key_a][0])
    rec_b_meta = viscot.get_meta(chosen_b[0])

    trace = sample_generic_trace(
        question=rec_b_meta["question"], answer=rec_b_meta["answer"],
    )
    trace.category = f"viscot-perm:{rec_a['source']}->{rec_b_meta['source']}"
    extra_qa = []
    for i in chosen_b[1:]:
        m = viscot.get_meta(i)
        extra_qa.append((m["question"], m["answer"]))
    return rec_a["image"], trace, extra_qa, rec_a["vsem_full"], rec_a["vsem_crop"]


# ---------------------------------------------------------------------------
# One training step (refactored out so the gradient probe can call it)
# ---------------------------------------------------------------------------
def run_one_step(
    *,
    model,
    new_emb,
    new_token_ids: dict,
    processor,
    concept_mlp,
    anchors,
    image: Image.Image,
    trace,
    cfg,
    step: int,
    extra_qa_pairs: list | None = None,
    vsem_full=None,
    vsem_crop=None,
) -> dict:
    """One forward + loss for the interleaved variant.

    Args:
        extra_qa_pairs: optional list of additional (q, a) tuples for the
            same image. If provided, the reader scores `[trace.(q,a)] +
            extra_qa_pairs` (truncated/padded to K_q). This is the multi-Q
            binding test (POC results §11.4): with K_q DIFFERENT (q, a)
            pairs sharing the same h, the model cannot specialise to one
            (q, a) — h must encode image-general content.
        vsem_full: optional pre-computed [D] V_sem of the full image.
            If provided alongside vsem_crop, the Stage-1 grounding loss
            (interleaved.grounding.block_grounding_loss) replaces the
            combined()-style loss. Curriculum: w_grounding 1.0→0.3,
            w_nll 0.0→1.0 over `cfg.interleaved.stage1_warmup_steps`.
        vsem_crop: optional pre-computed [D] V_sem of the bbox-cropped image.

    Returns a losses dict with `total` graph-attached. Caller does backward.
    """
    h = run_interleaved_forward(
        model=model,
        new_emb=new_emb,
        new_token_ids=new_token_ids,
        processor=processor,
        image=image,
        trace=trace,
        detach_recurrence_input=bool(getattr(cfg.interleaved, "detach_recurrence_input", False)),
    )  # [1, K_total, D] with grad

    # Build the batch shape that nll/combined() expects.
    K_q = max(1, cfg.loss.K_q)
    if extra_qa_pairs:
        all_pairs = [(trace.question, trace.answer)] + list(extra_qa_pairs)
        if len(all_pairs) >= K_q:
            qa_list = all_pairs[:K_q]
        else:
            qa_list = [all_pairs[i % len(all_pairs)] for i in range(K_q)]
    else:
        qa_list = [(trace.question, trace.answer)] * K_q
    batch_records = [{"questions": qa_list}]

    # === Stage-1 grounding path (Visual-CoT with pre-computed V_sem) ===
    if vsem_full is not None and vsem_crop is not None:
        import torch as _torch
        # Convert numpy → tensor on CUDA, batch dim = 1
        target_full = _torch.as_tensor(vsem_full, dtype=_torch.float32).cuda().unsqueeze(0)  # [1, D]
        target_crop = _torch.as_tensor(vsem_crop, dtype=_torch.float32).cuda().unsqueeze(0)  # [1, D]
        T_blocks = int(cfg.interleaved.T_blocks)
        k_latent = int(cfg.interleaved.k_latent)
        if T_blocks != 2:
            raise ValueError(
                f"Stage-1 grounding currently expects T_blocks=2 "
                f"(block 0 → full, block 1 → crop); got T_blocks={T_blocks}"
            )
        block_sizes = [k_latent] * T_blocks
        targets_per_block = [target_full, target_crop]
        grounding = block_grounding_loss(
            h=h,
            targets_per_block=targets_per_block,
            block_sizes=block_sizes,
            concept_mlp=concept_mlp,
        )
        # Reader-NLL (graph-attached, no curriculum applied here).
        nll = nll_multi_anchor(h, batch_records, anchors, K_q=cfg.loss.K_q)
        # Stage-1 → Stage-2 curriculum
        warmup = int(getattr(cfg.interleaved, "stage1_warmup_steps", 500))
        w_g, w_nll = stage1_curriculum_weights(step, warmup=warmup)
        total = w_g * grounding + w_nll * nll
        with _torch.no_grad():
            mean_h_norm = h.float().norm(dim=-1).mean()
        losses = {
            "total": total,
            "nll": nll.detach(),
            "concept": grounding.detach(),  # name "concept" for jsonl-schema compat
            "norm": h.new_zeros(()),
            "grounding": grounding.detach(),  # explicit Stage-1 alias
            "mean_h_norm": mean_h_norm,
            "cos_h_vsem": h.new_zeros(()),
            "w_nll": float(w_nll),
            "w_concept": float(w_g),  # "w_concept" name for jsonl compat
            "w_norm": 0.0,
            "w_grounding": float(w_g),
        }
        return losses, h

    # === Original (non-grounding) path ===
    if cfg.loss.w_concept != 0.0:
        v_sem = get_v_sem(model, processor, [image])
    else:
        v_sem = None

    losses = combined(
        h=h,
        batch=batch_records,
        anchors=anchors,
        v_sem=v_sem,
        concept_mlp=concept_mlp,
        cfg=cfg.loss,
        step=step,
    )

    # Round-3 norm-diagnostic fix (b): replace the standard mean-over-all-K
    # norm penalty with one scoped to FIRST-OF-BLOCK positions. We do this
    # post-hoc by recomputing total = w_nll*nll + w_c*concept + w_n*norm_first
    # because `combined()` already added the all-positions term to `total` with
    # graph-attached `norm`. Cleanest re-derivation: scale the existing graph-
    # attached `norm` away and add the first-only term back with the same w_n.
    if bool(getattr(cfg.interleaved, "first_latent_norm_only", False)):
        k_latent = int(cfg.interleaved.k_latent)
        T_blocks = int(cfg.interleaved.T_blocks)
        first_idxs = [b * k_latent for b in range(T_blocks)]
        # h is the same tensor used by combined() — its grad path is intact.
        h_first = h[:, first_idxs, :]
        norm_first = ((h_first.float().norm(dim=-1) - cfg.loss.target_norm) ** 2).mean()
        # combined() built `total = w_nll*nll + w_c*concept + w_n*norm_full`
        # where w_n is `losses['w_norm']` (a Python float). Subtract the
        # graph-attached full-norm contribution and add the first-only one.
        # We need the graph-attached `norm_full` value: re-derive from `losses`
        # by recomputing it ourselves (the value in `losses['norm']` is detached).
        norm_full = ((h.float().norm(dim=-1) - cfg.loss.target_norm) ** 2).mean()
        w_n = losses["w_norm"]
        losses["total"] = losses["total"] - w_n * norm_full + w_n * norm_first
        losses["norm"] = norm_first.detach()
        losses["norm_full_diagnostic"] = norm_full.detach()

    return losses, h


# ---------------------------------------------------------------------------
# Held-out eval (round-2 follow-up B)
# ---------------------------------------------------------------------------
def _eval_heldout(
    *,
    model,
    new_emb,
    new_token_ids: dict,
    processor,
    anchors,
    eval_records: list,
    cfg,
    step: int,
) -> dict:
    """Compute mean reader-NLL over a fixed held-out set.

    Mirrors the structure of `run_one_step` but:
      - calls `model.eval()` (and restores training mode after)
      - wraps the forward + reader call in `torch.no_grad()` so no graph is
        built — we ONLY want training-mode params untouched, so we never want
        autograd state here.

    The eval records are pre-sampled at trainer init from a SEPARATE RNG seed
    so the training loop's own RNG never produces them (by construction —
    different rng state stream).

    Returns: {"step": int, "eval_nll_mean": float, "n": int}.
    """
    if not eval_records:
        return {"step": step, "eval_nll_mean": float("nan"), "n": 0}

    # Eval reader-NLL only — concept/norm are training regularisers, not signal.
    # We compute the same anchor-NLL as in `combined.nll_multi_anchor` but
    # without building a graph.
    from ..readers import forward_anchor

    was_training = model.training
    model.eval()
    nlls: list[float] = []
    try:
        with torch.no_grad():
            for image, trace in eval_records:
                h = run_interleaved_forward(
                    model=model,
                    new_emb=new_emb,
                    new_token_ids=new_token_ids,
                    processor=processor,
                    image=image,
                    trace=trace,
                    detach_recurrence_input=bool(getattr(cfg.interleaved, "detach_recurrence_input", False)),
                )  # [1, K_total, D]
                K_total = int(h.shape[1])
                # Average over anchors (R) and K_q replicas (= 1 here, since we
                # always use the same held-out (q, a) — averaging duplicates is
                # a no-op).
                per_anchor = []
                for anchor in anchors:
                    loss = forward_anchor(
                        anchor, h, [trace.question], [trace.answer], K=K_total,
                    )
                    per_anchor.append(float(loss.item()))
                nlls.append(sum(per_anchor) / len(per_anchor))
    finally:
        if was_training:
            model.train()

    mean = sum(nlls) / len(nlls) if nlls else float("nan")
    return {"step": step, "eval_nll_mean": mean, "n": len(nlls)}


# ---------------------------------------------------------------------------
# Main training entry
# ---------------------------------------------------------------------------
def train(cfg) -> None:
    """Single-process interleaved-variant training loop.

    Caller is expected to have set MACHINE / HF_HOME via the env (vl.train.main).
    """
    if cfg.interleaved is None:
        raise ValueError(
            "Interleaved variant requires a 'interleaved' section in the config."
        )

    torch.manual_seed(cfg.trainer.seed)
    rng = random.Random(cfg.trainer.seed)

    # ---- Run dirs ----
    run_name = (cfg.wandb.run_name if cfg.wandb is not None else None) or cfg.name
    run_dir = vl_paths.checkpoint_root() / run_name
    results_dir = vl_paths.results_root() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[interleaved] run_name={run_name}")
    print(f"[interleaved] run_dir={run_dir}")
    print(f"[interleaved] results_dir={results_dir}")

    # ---- Build generator ----
    print("[interleaved] building generator...")
    bundle = build_interleaved_generator(cfg.model, dtype=torch.bfloat16)
    model = bundle["model"]
    processor = bundle["processor"]
    new_emb = bundle["new_emb"]
    new_token_ids = bundle["new_token_ids"]
    concept_mlp = bundle["concept_mlp"]

    # ---- Load anchors ----
    print(f"[interleaved] loading {len(cfg.anchors.paths)} anchor(s)...")
    anchors = load_anchors(
        cfg.anchors.paths, model, processor, dtype=torch.bfloat16
    )

    # NOTE on gradient checkpointing (design §8.2): the parallel method enables
    # it; the interleaved POC explicitly does NOT — design §8.2 says "Gradient
    # checkpointing not required and best avoided" because of its interaction
    # with the splice. Even if cfg.trainer.gradient_checkpointing is true we
    # leave it off for the interleaved variant.
    if cfg.trainer.gradient_checkpointing:
        print(
            "[interleaved] WARN: gradient_checkpointing requested but ignored — "
            "design §8.2 advises against it for the recurrent forward."
        )

    # ---- Optimizer (mirrors sft_anchor) ----
    lora_params = [
        p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad
    ]
    mlp_params = list(concept_mlp.parameters())
    token_params = [new_emb]

    if not lora_params:
        raise RuntimeError("no LoRA params found")
    if not mlp_params:
        raise RuntimeError("no concept-MLP params found")

    lr_lora = float(cfg.trainer.lr_lora)
    lr_token = float(cfg.trainer.lr_token)

    optim = torch.optim.AdamW(
        [
            {"params": lora_params + mlp_params, "lr": lr_lora, "weight_decay": 0.0},
            {"params": token_params, "lr": lr_token, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )

    n_lora = sum(p.numel() for p in lora_params)
    n_mlp = sum(p.numel() for p in mlp_params)
    n_tok = sum(p.numel() for p in token_params)
    print(f"[interleaved] trainable: lora={n_lora:,} mlp={n_mlp:,} token={n_tok:,}")

    from torch.optim.lr_scheduler import CosineAnnealingLR
    sched = CosineAnnealingLR(
        optim, T_max=max(1, cfg.trainer.max_steps), eta_min=0.1 * lr_lora
    )

    # ---- Logs ----
    losses_path = results_dir / "losses.jsonl"
    losses_jsonl = losses_path.open("a")
    eval_path = results_dir / "eval.jsonl"
    eval_jsonl = eval_path.open("a")

    # ---- Data source dispatch (round-3) ----
    # `data_source` is "synthetic" (default) or "gqa". Synthetic keeps round-2
    # behaviour exactly. GQA pre-loads a pool of real (image, q, a) records
    # via `vl.data.gqa.load_gqa` and pairs them with the generic trace template.
    data_source = str(getattr(cfg.interleaved, "data_source", "synthetic"))
    permute_template = bool(getattr(cfg.interleaved, "permute_template", False))

    gqa_train_records: list = []
    gqa_eval_records: list = []
    if data_source == "gqa":
        from ..data.gqa import load_gqa

        n_gqa = int(getattr(cfg.interleaved, "n_samples", 100))
        print(f"[interleaved] data_source=gqa — loading {n_gqa} GQA train images...")
        gqa_train_records = load_gqa(
            split="train",
            n_per_image=2,
            n_images=n_gqa,
            held_out_q=False,
            seed=cfg.trainer.seed,
        )
        print(f"[interleaved] loaded {len(gqa_train_records)} GQA train records")
        # Held-out: 10 testdev records — fixed seed for determinism, completely
        # disjoint from the training image pool.
        print("[interleaved] loading 10 GQA testdev images for held-out eval...")
        gqa_eval_records = load_gqa(
            split="testdev",
            n_per_image=2,
            n_images=10,
            held_out_q=False,
            seed=0,  # fixed across runs
        )
        print(f"[interleaved] loaded {len(gqa_eval_records)} GQA eval records")
    elif data_source == "viscot":
        from ..data.viscot import ViscotDataset

        hub_repo = getattr(cfg.interleaved, "viscot_hub_repo", "ohjoonhee/visual-cot-50k-poc")
        train_vsem = getattr(cfg.interleaved, "viscot_vsem_train", "data/viscot/viscot_50k_train_vsem.parquet")
        eval_vsem = getattr(cfg.interleaved, "viscot_vsem_eval", "data/viscot/viscot_1k_eval_vsem.parquet")
        print(f"[interleaved] data_source=viscot — loading {hub_repo}...")
        viscot_train = ViscotDataset(hub_repo=hub_repo, vsem_parquet=train_vsem, split="train")
        viscot_eval = ViscotDataset(hub_repo=hub_repo, vsem_parquet=eval_vsem, split="eval")
    elif data_source not in ("synthetic", "shapes"):
        raise ValueError(
            f"InterleavedConfig.data_source must be 'synthetic' | 'gqa' | 'shapes' | 'viscot'; "
            f"got {data_source!r}"
        )

    # ---- Held-out eval set (round-2 follow-up B) ----
    # For synthetic: sample N=10 (image, trace) tuples from a SEPARATE RNG seed
    # so the training loop's RNG cannot reproduce them.
    # For GQA: take the first 10 testdev records, with a fixed eval RNG so the
    # (q, a) chosen per record is deterministic across runs.
    n_eval = 10
    eval_rng = random.Random(cfg.trainer.seed + 9999)
    if data_source == "gqa":
        # Use a deterministic eval over the FIRST 10 testdev records so the
        # natural and perm runs see the SAME held-out (image, q, a) triples —
        # only the trainer's per-step sampler differs by `permute_template`.
        # We always score against natural (q, a) pairings, regardless of the
        # training-side flag: the held-out NLL is a reference for how well
        # the model answers the actual GQA question for the actual image,
        # NOT a permuted variant (which would be apples-to-oranges).
        # This also fixes the round-2 caveat where the perm flag affected
        # the eval RNG draws.
        eval_records = []
        eval_q_rng = random.Random(1234)
        for rec in gqa_eval_records[:n_eval]:
            q, a = eval_q_rng.choice(rec["questions"])
            img = rec["image_loader"]()
            trace = sample_generic_trace(question=q, answer=a)
            eval_records.append((img, trace))
    elif data_source == "shapes":
        # Held-out: 10 fresh shapes scenes scored against their NATURAL (q,a).
        # Even when the trainer is in permute mode, eval pairs every image
        # with its own ground truth — so eval is comparable across runs.
        eval_records = [
            _sample_shapes_example(eval_rng, permute_template=False)
            for _ in range(n_eval)
        ]
    elif data_source == "viscot":
        # Held-out: first n_eval records of the viscot eval split, paired
        # naturally (image with its own q, a). The eval split is image-disjoint
        # from train by construction (see scripts/cluster/process_viscot.py).
        # Eval scoring uses NATURAL (q, a) regardless of training-side perm flag.
        eval_records = []
        for i in range(min(n_eval, len(viscot_eval))):
            rec = viscot_eval.get_record(i)
            tr = sample_generic_trace(question=rec["question"], answer=rec["answer"])
            tr.category = f"viscot:{rec['source']}"
            eval_records.append((rec["image"], tr))
    else:
        eval_records = [
            _sample_synthetic_example(
                eval_rng,
                permute_template=permute_template,
            )
            for _ in range(n_eval)
        ]
    print(f"[interleaved] held-out eval: {len(eval_records)} records (data_source={data_source})")

    # ---- Loop ----
    print(
        f"[interleaved] beginning training: max_steps={cfg.trainer.max_steps}, "
        f"T_blocks={cfg.interleaved.T_blocks}, k_latent={cfg.interleaved.k_latent}"
    )
    start_time = time.monotonic()
    model.train()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    if permute_template:
        print(
            "[interleaved] permute_template=True — design §11.3 control: "
            "trace text from a DIFFERENT template than the (q, a) the reader scores."
        )

    try:
        multi_q = bool(getattr(cfg.interleaved, "multi_q_per_image", False))
        for step in range(cfg.trainer.max_steps):
            extra_qa: list | None = None
            vsem_full = None
            vsem_crop = None
            if data_source == "gqa":
                image, trace = _sample_gqa_example(
                    gqa_train_records, rng, permute_template=permute_template,
                )
            elif data_source == "shapes":
                if multi_q:
                    image, trace, extra_qa = _sample_shapes_multiq_example(
                        rng, K_q=max(1, cfg.loss.K_q),
                        permute_template=permute_template,
                    )
                else:
                    image, trace = _sample_shapes_example(
                        rng, permute_template=permute_template,
                    )
            elif data_source == "viscot":
                image, trace, extra_qa, vsem_full, vsem_crop = _sample_viscot_example(
                    viscot_train,
                    rng,
                    K_q=max(1, cfg.loss.K_q),
                    permute_template=permute_template,
                )
            else:
                image, trace = _sample_synthetic_example(
                    rng, permute_template=permute_template,
                )

            losses, h = run_one_step(
                model=model,
                new_emb=new_emb,
                new_token_ids=new_token_ids,
                processor=processor,
                concept_mlp=concept_mlp,
                anchors=anchors,
                image=image,
                trace=trace,
                cfg=cfg,
                step=step,
                extra_qa_pairs=extra_qa,
                vsem_full=vsem_full,
                vsem_crop=vsem_crop,
            )

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optim.param_groups for p in g["params"]],
                max_norm=1.0,
            )
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            record = {
                "step": step,
                "category": trace.category,
                "K_total": int(h.shape[1]),
                "total": float(losses["total"].item()),
                "nll": float(losses["nll"].item()),
                "concept": float(losses["concept"].item()),
                "norm": float(losses["norm"].item()),
                "mean_h_norm": float(losses["mean_h_norm"].item()),
                "w_nll": float(losses["w_nll"]),
                "w_concept": float(losses["w_concept"]),
                "w_norm": float(losses["w_norm"]),
                "elapsed_s": time.monotonic() - start_time,
            }
            losses_jsonl.write(json.dumps(record) + "\n")
            losses_jsonl.flush()

            if step % 5 == 0:
                print(
                    f"[step {step}/{cfg.trainer.max_steps}] cat={trace.category} "
                    f"total={record['total']:.3f} nll={record['nll']:.2f} "
                    f"||h||={record['mean_h_norm']:.1f}"
                )

            # ---- Held-out eval (round-2 follow-up B) ----
            # Run at step 0 and every eval_every_steps thereafter so the run
            # has a baseline + per-checkpoint NLL trajectory.
            if (
                cfg.trainer.eval_every_steps > 0
                and (step % cfg.trainer.eval_every_steps == 0 or step == cfg.trainer.max_steps - 1)
            ):
                eval_record = _eval_heldout(
                    model=model,
                    new_emb=new_emb,
                    new_token_ids=new_token_ids,
                    processor=processor,
                    anchors=anchors,
                    eval_records=eval_records,
                    cfg=cfg,
                    step=step,
                )
                eval_jsonl.write(json.dumps(eval_record) + "\n")
                eval_jsonl.flush()
                print(
                    f"[interleaved] eval @ step {step}: "
                    f"nll_mean={eval_record['eval_nll_mean']:.3f} (n={eval_record['n']})"
                )
    finally:
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"[interleaved] peak GPU memory allocated: {peak:.2f} GiB")
        import contextlib
        with contextlib.suppress(Exception):
            losses_jsonl.close()
            eval_jsonl.close()
