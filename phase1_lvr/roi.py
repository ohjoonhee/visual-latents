"""ROI patch-index gather for Qwen2.5-VL.

Given a normalized xyxy bbox + the processor's `image_grid_thw`, return the K=8
post-merger patch indices whose patch CENTER falls inside the bbox. If fewer
than K are inside, pad with the patches closest (Euclidean distance) to the
bbox center. If more than K are inside, take the first K in raster order.

Qwen2.5-VL geometry:
- Image is split into patches of size `patch_size` (=14).
- Patches are MERGED in groups of `merge_size × merge_size` (=2x2) before the
  vision projector.
- Post-merger token count = (grid_h * grid_w) / (merge_size**2).
- `image_grid_thw[0] = (T, grid_h, grid_w)` where T=1 for images and (grid_h,
  grid_w) are PRE-merge patch counts.
- The post-merger spatial grid is `(grid_h // merge_size, grid_w // merge_size)`.

Patch i in raster order (post-merger) corresponds to a (row, col) in the
post-merger grid; its NORMALIZED center is
`((col + 0.5) / merger_w, (row + 0.5) / merger_h)`.
"""
from __future__ import annotations

import torch


def select_roi_indices(
    image_grid_thw: torch.Tensor | tuple,
    bbox_xyxy_norm: tuple[float, float, float, float] | list[float],
    K: int = 8,
    merge_size: int = 2,
) -> list[int]:
    """Return K post-merger patch indices.

    Args:
        image_grid_thw: shape [num_images, 3] tensor or 1-image (T, H, W) tuple
            with PRE-merge grid dims.
        bbox_xyxy_norm: (x0, y0, x1, y1) in [0, 1].
        K: number of indices to return (paper-default 8).
        merge_size: 2 for Qwen2.5-VL.

    Returns:
        list[int] of length K.
    """
    if hasattr(image_grid_thw, "tolist"):
        thw = image_grid_thw.tolist()
        if isinstance(thw[0], (list, tuple)):
            thw = thw[0]
    else:
        thw = list(image_grid_thw)
    T, gh, gw = int(thw[0]), int(thw[1]), int(thw[2])
    if T != 1:
        raise NotImplementedError(f"only T=1 (single image) supported, got T={T}")
    if gh % merge_size != 0 or gw % merge_size != 0:
        raise ValueError(f"grid {gh}x{gw} not divisible by merge_size={merge_size}")

    mh = gh // merge_size  # post-merger rows
    mw = gw // merge_size  # post-merger cols
    n = mh * mw

    x0, y0, x1, y1 = bbox_xyxy_norm
    bcx, bcy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    inside: list[tuple[int, float]] = []  # (idx, dist_to_bbox_center)
    outside: list[tuple[int, float]] = []
    for i in range(n):
        row = i // mw
        col = i % mw
        cx = (col + 0.5) / mw
        cy = (row + 0.5) / mh
        d = ((cx - bcx) ** 2 + (cy - bcy) ** 2) ** 0.5
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            inside.append((i, d))
        else:
            outside.append((i, d))

    # Inside in raster order; outside in distance order (for padding).
    inside.sort(key=lambda t: t[0])
    outside.sort(key=lambda t: t[1])

    if len(inside) >= K:
        return [t[0] for t in inside[:K]]
    chosen = [t[0] for t in inside] + [t[0] for t in outside[: K - len(inside)]]
    return chosen[:K]


def post_merger_token_count(image_grid_thw, merge_size: int = 2) -> int:
    """Number of post-projector image tokens for a single-image grid_thw."""
    if hasattr(image_grid_thw, "tolist"):
        thw = image_grid_thw.tolist()
        if isinstance(thw[0], (list, tuple)):
            thw = thw[0]
    else:
        thw = list(image_grid_thw)
    T, gh, gw = int(thw[0]), int(thw[1]), int(thw[2])
    return T * (gh // merge_size) * (gw // merge_size)
