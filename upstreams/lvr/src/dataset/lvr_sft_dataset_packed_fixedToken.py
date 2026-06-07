"""Reconstruction stub for a file MISSING from the upstream LVR repo.

`src/dataset/__init__.py` imports `make_packed_supervised_data_module_lvr_fixedToken`
from this module, but `lvr_sft_dataset_packed_fixedToken.py` was never committed to
github.com/VincentLeebang/lvr (raw URL 404s). The published repo therefore cannot
`import src.dataset` as-is — the authors' working tree had this file locally.

The symbol is only *called* at `train_lvr.py:204`, inside
`if model_args.max_lvr_tokens is not None:` — the "Fixed Token" packed-data path.
The faithful 3B Stage-1 recipe (and the authors' own scripts) never set
`--max_lvr_tokens` (defaults to None), so this path is never taken. This stub exists
solely so `import src.dataset` succeeds; the regular `make_packed_supervised_data_module_lvr`
handles our packed-data training.

If you set `--max_lvr_tokens`, you need the authors' real implementation here.
"""


def make_packed_supervised_data_module_lvr_fixedToken(*args, **kwargs):
    raise NotImplementedError(
        "lvr_sft_dataset_packed_fixedToken is missing from the upstream LVR repo "
        "(never committed). It is only reached when --max_lvr_tokens is set, which "
        "the faithful 3B recipe does not use. Obtain the authors' implementation to "
        "enable the Fixed-Token packed-data path."
    )
