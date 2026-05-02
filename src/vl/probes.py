"""Steering and random-control probes.

Mirrors POC round-2's `steering_probe.py` and round-3's `tune_random_control.py`
+ `evaluate_random_control_heldout.py`. Used by `eval.py` and by the cell C5
random-control training condition.

TODO: implement steering_probe, random_control_collator.
"""


def steering_probe():
    """Apply zero_pos / permute_within / permute_across / gauss_noise to h, score Δnll."""
    raise NotImplementedError("probes.steering_probe not yet implemented")


def random_control_collator():
    """Return a collator that shuffles (image, question, answer) triples within batch."""
    raise NotImplementedError("probes.random_control_collator not yet implemented")
