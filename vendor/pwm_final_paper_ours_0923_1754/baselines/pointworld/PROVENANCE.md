# Provenance

Vendored from **PointWorld** (NVlabs), Apache-2.0. `LICENSE` is upstream's; `ptv3/LICENSE`
covers the PTv3 backbone, which PointWorld itself vendored from Pointcept.

- Source checkout: `/data/user_data/mbronars/packages/PointWorld`, commit `7554c9b3e21a0e70f1aa2f79c142c8905f2ea7bb`
  (`origin` = git@github.com:NVlabs/PointWorld.git), copied 2026-09-23.
- Copied: the 32 files an inference call touches — the model (`pointworld/`), the PTv3 backbone
  (`ptv3/`, plus `ptv3_arch.yaml`), the preprocessing transforms (`dataset_components/`),
  `arguments.py`, `scene_featurizer.py`, `transform_utils.py`, `robot_sampler.py` and
  PointWorld's top-level `utils.py`.
- **Not** copied: training (`training/`, `train.py`), the evaluator (`evaluation/`), the
  data converters, the URDF and mesh assets, and everything else upstream carries.

## Changes made here

1. `utils.py` -> `pw_utils.py`, and the four `from utils import ...` lines that reach it
   (`pointworld/base.py`, `pointworld/losses.py`, `dataset_components/transforms.py`,
   `robot_sampler.py`) now say `from pw_utils import ...`. Without this, PointWorld's
   top-level `utils.py` would shadow this repo's `utils` package.
2. `infer.py` added — the inference entry point. Not upstream.
3. `norm_stats/` added — the normalization statistics of our runs. Not upstream.

Nothing else was edited: the model and transform files are byte-for-byte the source checkout's,
which itself carries our local modifications to PointWorld (that checkout is 12 tracked files
ahead of upstream `7554c9b3e21a0e70f1aa2f79c142c8905f2ea7bb`, most importantly `--pred_horizon`, which the 16-frame checkpoints
need and upstream does not have).

## Re-syncing

Copy the same file list from a newer PointWorld checkout and redo change 1; `infer.py` and
`norm_stats/` are ours and carry over.
