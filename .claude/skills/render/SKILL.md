---
name: render
description: >
  Turn a markdown deliverable (a report, analysis, findings, design doc, plan)
  into a self-contained HTML view that is easier to review than a long linear
  document. Use when the user invokes /render, says "build HTML" / "render this
  as HTML", or asks to see a markdown document in a more reviewable form. Not
  for markdown the agent or tooling consumes as working context (instruction
  files, logs, append-only notes) — those stay markdown.
user-invocable: true
argument-hint: "[path to a .md deliverable, or which document to render]"
---

# Render — Markdown Deliverable → Reviewable HTML

## The intent

Take a markdown document that a person has to *review and act on*, and give
them the artifact they'd actually want in front of them to understand it and
decide.

Markdown is right for text read as working context. But a long document someone
must *judge* is the wrong shape: the things that carry the decision — numbers
across conditions, a trajectory, a claim sitting next to its evidence — are
spatial, and prose flattens them into a wall you skim. HTML is the chance to
give that content the shape it always deserved. That reshaping is the entire
point of this skill; everything below is in service of it.

## What you are actually making

Act as the person who has to *present this work* and make someone get it. You
read the whole thing, understood what it is really about, decided what the
load-bearing points are, and built an exhibit around them: the key evidence as
the centerpiece, the structure spatial and navigable, each claim next to the
thing that proves it. Prose supports; it is not the show.

The standard is the reader's comprehension, and you should spend real effort
there. If the content has equations, typeset them so they read as math, not
source. If it carries a trajectory of numbers, that trajectory wants to be a
figure the eye grasps at once, not a table it has to reconstruct. If it
describes a structure, a diagram beats three paragraphs. Always ask what makes
*this* content land fastest for the person reading it, and do that.

One honest test tells you whether you have done the job: a reader reaches the
document's conclusion **faster and with more confidence than by reading the
markdown**. If the truthful answer is "they'd learn the same thing, it just
looks nicer," the work is not finished.

Default form is an interactive **slide deck** — one idea per view, a headline
claim, the evidence beside it, a one-line "so what", keyboard nav, an overview.
Reach for scrollytelling or a dashboard when the content clearly wants it. There
is no template; designing the right view *is* the thinking.

## Why the obvious path produces nothing

You will arrive holding the markdown, and the cheapest move will be to render
each section, style it, and wrap it in a carousel. It matters that you
understand *why* that fails rather than just avoid it: it preserves the exact
linear order and section granularity that made the document hard to review in
the first place. A deck whose slides are the source's sections — bodies and all
— is the same wall with pagination, and a headline pasted on top does not change
that. The value you add is not formatting the document; it is the judgment, from
someone who now understands it, about what the reader needs to see and in what
shape.

## The contract that makes this safe

You have wide editorial latitude — selecting, re-sequencing, re-shaping — and it
is safe *because* of three guarantees that let the reader trust and verify what
you built:

- **Numbers and claims are the source's, verbatim.** You re-present and reshape
  them; you never recompute or reinterpret. Any interactivity is presentation
  only.
- **The verbatim source is one action away.** Every view links to the exact
  source section it was distilled from — a faithful conversion of that section,
  embedded in the same file — so a reader can check your editorial call against
  the original without leaving the page.
- **Markdown stays canonical; the HTML is disposable.** Regenerate, never
  hand-edit. Stamp the build with the source's hash and have it detect when the
  served source has since changed, so a stale view announces itself.

Producing this well is real work — write a small throwaway build script rather
than hand-typing HTML, so the conversion stays faithful and the build repeats.

## Make it viewable

A file on disk is useless if the reader cannot open it, and they are usually on
a different device with no patience for port-forwarding — often a phone. Serve
the directory over loopback with one long-lived static server (reused across
renders, started only if nothing is already serving), then expose it through a
public tunnel — `cloudflared` or similar — and hand back that link, so it opens
on any device with no setup. The tunnel URL is public and ephemeral; that is the
deliberate tradeoff for zero-setup access. Keep the local port stable across
renders.
