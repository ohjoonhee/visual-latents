"""Hand-written trace templates (design §5.1, option (a)).

Each template specifies a question form and a teacher-forced trace pattern
that alternates plain-text segments with `latent_block` markers. The trace
builder fills in placeholders (`{object}`, `{answer}`) and yields a list of
text segments + a list of per-block latent counts so the trainer can drive
the Coconut recurrence.

POC: 5 templates total (count, color, position, presence, attribute) — design
§5.1 calls for "10 templates" eventually; the 5 here cover the structural
pattern for all categories. Add more later without touching trainer code.

Templates are intentionally minimal and unambiguous. Per design §6.5, the
text is NEVER a loss target — it is only used to construct the recurrent
forward pass. Wording quality matters only insofar as it produces a coherent
prefix for the latents to attend over.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# k_latent per block; design §2.1 fixes this at 4. Trainer override possible
# via cfg.interleaved.k_latent — templates use this constant via sample_trace.
_DEFAULT_K_LATENT = 4

# 5 hand-written templates covering count / color / position / presence / attribute.
# Each is `[(kind, payload), ...]`; kind in {"text", "latent_block"}.
# `latent_block` payload is the number of latent positions in that block.
TEMPLATES: list[dict] = [
    {
        "category": "count",
        "question_template": "How many {object} are there in the image?",
        "trace_template": [
            ("text", "To count the {object}, I need to look at the image carefully."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "Now I will count them."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "The answer is {answer}."),
        ],
    },
    {
        "category": "color",
        "question_template": "What color is the {object}?",
        "trace_template": [
            ("text", "I need to identify the color of the {object}."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "Let me focus on its appearance."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "The {object} is {answer}."),
        ],
    },
    {
        "category": "position",
        "question_template": "Where is the {object} located?",
        "trace_template": [
            ("text", "I will locate the {object} in the scene."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "Now I will describe its position."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "The {object} is {answer}."),
        ],
    },
    {
        "category": "presence",
        "question_template": "Is there a {object} in the image?",
        "trace_template": [
            ("text", "I need to check whether a {object} is present."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "Let me scan the image."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "The answer is {answer}."),
        ],
    },
    {
        "category": "attribute",
        "question_template": "What is the {object} doing?",
        "trace_template": [
            ("text", "I need to identify the action of the {object}."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "Let me look at its pose."),
            ("latent_block", _DEFAULT_K_LATENT),
            ("text", "The {object} is {answer}."),
        ],
    },
]


# Strategy A from round-3 brief: a single generic template that requires NO
# {object} placeholder, so it can be applied to ANY (free-form question, answer)
# pair — e.g. GQA's natural-language questions which are not pre-categorised.
# Same T_blocks=2, k_latent=4 → K_total=8 shape as the synthetic templates,
# so configs (and the `Trace` -> `run_interleaved_forward` path) are unchanged.
GENERIC_TEMPLATE: dict = {
    "category": "generic",
    # No question_template — the question is supplied by the data source.
    "trace_template": [
        ("text", "Let me look at the image to answer the question."),
        ("latent_block", _DEFAULT_K_LATENT),
        ("text", "Based on what I see,"),
        ("latent_block", _DEFAULT_K_LATENT),
        ("text", "the answer is {answer}."),
    ],
}


@dataclass
class Trace:
    """A fully-instantiated training trace.

    Attributes:
        prompt_text: System + user prompt, ending with "<|im_start|>assistant\\n".
            Includes the image placeholder; `forward_interleaved` will pair it
            with the actual PIL image via the processor.
        text_segments: Plain-text segments in order. T_blocks+1 entries:
            [seg_0, seg_1, ..., seg_T].  seg_0 prefixes block 1; the trailing
            seg_T (the answer / closer) is included for completeness but is
            NOT used by the reader (latents-only consumption per design §4.2).
        latent_block_sizes: list of k_latent per block. len == T_blocks.
        question: instantiated question string (already inside prompt_text).
        answer: gold answer string (already inside trailing text segment).
        category: template category, for diagnostics.
    """

    prompt_text: str
    text_segments: list[str]
    latent_block_sizes: list[int]
    question: str
    answer: str
    category: str


def _build_prompt_with_image(question: str) -> str:
    """Mirror parallel method's prompt skeleton (one image, one question)."""
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def sample_trace(
    *,
    obj: str,
    answer: str,
    rng: random.Random | None = None,
    template_idx: int | None = None,
) -> Trace:
    """Sample a trace template and instantiate it for `(obj, answer)`.

    Args:
        obj: object phrase (e.g. "red cubes", "person on the left").
        answer: gold answer string.
        rng: optional Random; if None, uses module-level random.
        template_idx: optional explicit index; if None, sampled.

    Returns:
        Trace populated with prompt_text, text_segments, latent_block_sizes.
    """
    if rng is None:
        rng = random
    if template_idx is None:
        template_idx = rng.randrange(len(TEMPLATES))
    tpl = TEMPLATES[template_idx]

    question = tpl["question_template"].format(object=obj)
    prompt_text = _build_prompt_with_image(question)

    text_segments: list[str] = []
    latent_block_sizes: list[int] = []
    for kind, payload in tpl["trace_template"]:
        if kind == "text":
            seg = payload.format(object=obj, answer=answer)
            text_segments.append(seg)
        elif kind == "latent_block":
            latent_block_sizes.append(int(payload))
        else:
            raise ValueError(f"Unknown trace kind: {kind!r}")

    # Validate the alternation: text_segments has T_blocks + 1 entries when
    # the template starts with text and ends with text (the design §2.3 form).
    if len(text_segments) != len(latent_block_sizes) + 1:
        raise RuntimeError(
            f"Template '{tpl['category']}' has {len(text_segments)} text segments "
            f"and {len(latent_block_sizes)} latent blocks; expected text=blocks+1."
        )

    return Trace(
        prompt_text=prompt_text,
        text_segments=text_segments,
        latent_block_sizes=latent_block_sizes,
        question=question,
        answer=answer,
        category=tpl["category"],
    )


def sample_generic_trace(*, question: str, answer: str) -> Trace:
    """Instantiate the generic template for a free-form (question, answer) pair.

    Used by the GQA sampler (round-3): GQA questions are natural-language and
    NOT pre-categorised, so they do not fit the synthetic templates'
    `{object}` placeholder. The generic template carries no `{object}` slot —
    only `{answer}` in the closing segment. The reader scores against the
    given (question, answer); the trace text never references the object name.
    """
    tpl = GENERIC_TEMPLATE
    prompt_text = _build_prompt_with_image(question)

    text_segments: list[str] = []
    latent_block_sizes: list[int] = []
    for kind, payload in tpl["trace_template"]:
        if kind == "text":
            seg = payload.format(answer=answer)
            text_segments.append(seg)
        elif kind == "latent_block":
            latent_block_sizes.append(int(payload))
        else:
            raise ValueError(f"Unknown trace kind: {kind!r}")

    if len(text_segments) != len(latent_block_sizes) + 1:
        raise RuntimeError(
            f"Generic template has {len(text_segments)} text segments "
            f"and {len(latent_block_sizes)} latent blocks; expected text=blocks+1."
        )

    return Trace(
        prompt_text=prompt_text,
        text_segments=text_segments,
        latent_block_sizes=latent_block_sizes,
        question=question,
        answer=answer,
        category=tpl["category"],
    )
