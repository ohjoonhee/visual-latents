"""Variant A trainer: trl.SFTTrainer subclass with multi-anchor compute_loss.

Overrides `compute_loss` to compute the combined loss (NLL_multi + L_concept +
L_norm with curriculum). Uses base SFTTrainer for: distributed setup,
optimizer, scheduler, checkpointing, logging.

TODO: implement SFTAnchorTrainer.
"""

# from trl import SFTTrainer  # noqa: ERA001
#
# class SFTAnchorTrainer(SFTTrainer):
#     def __init__(self, *, anchors, vl_config, **kwargs):
#         super().__init__(**kwargs)
#         self.anchors = anchors
#         self.vl_config = vl_config
#
#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         # 1. forward generator → h
#         # 2. for each anchor, compute NLL with h spliced in
#         # 3. compute L_concept (cosine to teacher V_sem)
#         # 4. compute L_norm (per-token norm matching)
#         # 5. weighted sum via curriculum coefficients at self.state.global_step
#         raise NotImplementedError
