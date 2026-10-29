# Water-3D: the opt-in `water` domain

DeepMind's Learning to Simulate (Sanchez-Gonzalez et al., ICML 2020, [arXiv 2002.09405](https://arxiv.org/abs/2002.09405))
released Water-3D: SPH water sloshing in a closed unit box, 1,000 / 100 / 100 trajectories of
801 frames at 200 Hz (4 s), 4,352 to 13,248 particles, one particle type and no wall particles.
It is the only 3D liquid in that release (Sand-3D and Goop-3D are the other 3D sets). Here it is
the domain `water`: passive like MPM (no actuator, no rigid bodies), with its own material id,
frame, frame rate and normalization entry. It is never selected by default.

## Getting the data

```bash
bash scripts/unified/download_water3d.sh                      # 90 GB of TFRecords, sizes verified
PYTHONPATH=. python scripts/unified/materialize_water3d.py \
    --src /data/group_data/katefgroup/datasets/deepmind_lts/Water-3D \
    --out /data/group_data/katefgroup/datasets/deepmind_lts/Water-3D/materialized_v1
```

| | location |
|---|---|
| release | `/data/group_data/katefgroup/datasets/deepmind_lts/Water-3D/{metadata.json,train,valid,test}.tfrecord` (75.1 / 7.5 / 7.5 GB) |
| what the loader reads (`WATER_ROOT`) | `.../Water-3D/materialized_v1/{train,valid,test}/` — one float32 `[801, N, 3]` `.npy` per trajectory plus `manifest.json`, 84 GB |

A TFRecord is one sequential stream of whole trajectories, so random window access needs the
materialized form; the loader memory-maps a file and reads only the frames a window needs.
Conversion takes about 15 minutes for all three splits and refuses to overwrite. The splits are
the release's own: `train` trains, `valid` is the validation split, `test` stays held out.

## What the domain fixes (code-baked decisions)

| decision | value | where |
|---|---|---|
| frame | file `y` up → `z` up by `(x, y, z) → (x, −z, y)`, then the shift `(−0.5, 0.5, −0.125)`: the box becomes `x, y ∈ ±0.375 m`, `z ∈ [0, 0.75] m`, one box unit = 1 m | `CANONICAL_TRANSLATION["water"]`, `data/water3d.py:to_model_frame` |
| frame rate | every 8th file frame, 25 fps (next to MPM's 24) | `DOMAIN_DT["water"] = 0.04`, `WATER3D_STRIDE` |
| window | 4 context + 16 predicted frames | `data.context_len`, `DOMAIN_FUTURE_LEN["water"] = 16` |
| particles | 4,096 per window: a fresh random subset per training sample, one seeded subset per trajectory for validation; shorter trajectories are zero-padded and masked | `WATER3D_PARTICLES`, `WATER3D_SUBSET_SEED` |
| window sampling | training: uniform over trajectories and start frames. validation: one window per trajectory at a start drawn from its key. rollouts: the whole trajectory from frame 0 | `UnifiedWaterDataset` |
| material | `MAT_WATER = 6`, so water runs need `model.num_mat_types=7` (the default stays 6) | `data/unified_dataset.py` |
| physics features | density 1000 (water's); stiffness `1e4` and Poisson ratio `0.40` are placeholders, meaningless in a single-material liquid domain | `WATER3D_FILL_*` |
| actions and bodies | none: zero actions, no actuated particles, one object, no rigid sidecar | `water_batch` |
| walls | the box bounds only. Water-3D has no boundary particles, and the domain adds no wall feature | see Limitations |

## Normalization

Water's particle scale lives in its **own artifact**, `stats/normalization_water3d.json`: the frozen
`stats/normalization.json` plus a `particle.water` entry. Select it with
`normalization.artifact=stats/normalization_water3d.json`; a water run without it fails at config
time.

The frozen file is deliberately untouched. A run's normalization checksum covers the whole
artifact, and resume and saved fixtures compare that checksum, so adding an entry to the frozen
file would stop every existing run from resuming and every saved batch from loading strictly.
Water is also outside the global aggregation, so `xy_p99_global` keeps its four-domain values. That
gives a second, file-free option: under `normalization.profile=xy_p99_global` water uses the shared
reference scales, which are MPM's (±26.8 / ±96.8 mm per frame), about 2.6x tighter in x/y than its
own fit. On the one batch measured so far the water fit edges it, 22.8 against 23.6 mm; see below.

Refit (for a different frame rate, window or split):

```bash
PYTHONPATH=. python scripts/dev/water3d_stats.py --split train --policies prefix uniform \
    --out stats/my_water_candidate.json --artifact_version water-<date>
```

The estimator is the one in `data/stats_fitting.py`: per-component absolute p99 of the window's
future deltas, `xy = max(p99|dx|, p99|dy|)`, `z = p99|dz|`, componentwise maximum over the chosen
window policies (`prefix` = every trajectory's first window, the violent collapse; `uniform` =
every window start, which is what training draws).

## Commands

```bash
# train (transformer)
python train.py --config configs/transformer.yaml run_name=water_tf 'data.domains=[water]' \
    model.num_mat_types=7 normalization.artifact=stats/normalization_water3d.json

# train (UNet: it also needs a coarse-point budget for material 6)
python train.py --config configs/unet.yaml run_name=water_unet 'data.domains=[water]' \
    model.num_mat_types=7 normalization.artifact=stats/normalization_water3d.json \
    'model.unet_l1_by_mat="0:256,1:256,2:256,3:128,4:128,5:512,6:1024"'

# evaluate (fixed windows and 48-frame rollouts on the valid split)
PYTHONPATH=. python scripts/unified/eval_unified.py --checkpoint <ckpt> --domains water \
    --rollout --rollout_frames 48 --clips all --seed 0 --out eval.json

# overfit one saved batch, then score it per window
PYTHONPATH=. python scripts/dev/water3d_overfit_fixture.py --trajectory 0 --name water_valid0_c4f16b4 \
    --batch_dir <dir> --normalization_artifact stats/normalization_water3d.json
PYTHONPATH=. python scripts/unified/overfit_prod.py --config configs/unet.yaml --domain water \
    --batch_name water_valid0_c4f16b4 --batch_dir <dir> --require_saved_batches --steps 10000 \
    --eval_every 500 --warmup 200 --eval_rng legacy --save_final --out_root <out> --tag outputs \
    --override seed=42 normalization.artifact=stats/normalization_water3d.json data.batch_size=4 \
      data.context_len=4 data.future_len=16 "data.domains=[water]" train.compile=false model.num_mat_types=7
PYTHONPATH=. python scripts/unified/overfit_per_window.py --run <out>/outputs \
    --fixture <dir>/water_valid0_c4f16b4 --out per_window.json

# render it like the paper's Figure 1 (separate environment: env/render_water3d.requirements.txt)
PYTHONPATH=. python scripts/render_water3d/dump_overfit_windows.py --run <out>/outputs \
    --fixture <dir>/water_valid0_c4f16b4 --out_prefix dumps/arm
python scripts/render_water3d/render_rollout.py --npz dumps/arm_w0.npz --out renders/w0 --label "UNet overfit"
python scripts/render_water3d/render_points.py --npz dumps/arm_w0.npz --out renders/w0_points
python scripts/render_water3d/compose_page_gif.py renders/w0 4 w0.gif
```

## Where it stands

Everything measured so far is a **fixed-batch overfit**, not a benchmark: one saved batch of four
20-frame windows from valid trajectory 0, 4,096 particles, the August MPM-specialist protocol
(10,000 updates, 8-seed final). No full-corpus training run has happened yet.

| backbone | delta scale | targets | error, 8 seeds (mm) |
|---|---|---|---|
| UNet | MPM reference (±26.8 / ±96.8 mm per frame) | clipped | 39.0 |
| transformer | MPM reference | clipped | 40.5 |
| UNet | water, first-window fit (116 / 121 mm) | unclipped | 33.6 |
| transformer | water, first-window fit | unclipped | 62.4 |
| UNet | MPM reference | unclipped | 23.6 |
| UNet | water, every-window fit (69.8 / 86.3 mm) — this artifact | unclipped | **22.8** |

Three things came out of it.

**Clipping the targets was the real limit.** Against MPM's motion limits, 19.7% of the fast window's
targets were clipped, which alone capped that window at 74 mm. Removing the clipping, at the same
scale, took the batch from 39.0 to 23.6 mm.

**A scale set by the violent window costs precision everywhere else.** The first-window fit (116 mm,
set by the initial collapse) lands at 33.6 mm, worse than both unclipped alternatives; the calmest
window went from 9 to 21 mm under it. The every-window fit shipped here avoids that.

**Water's own scale and MPM's are close.** 22.8 ± 0.21 against 23.6 ± 0.29 mm, and they differ only
away from the collapse:

| window | water moves | water fit (shipped) | MPM reference scale |
|---|---|---|---|
| collapse | 62 mm/frame | 34.7 mm | 35.0 mm |
| second | 27 | 29.7 | 28.7 |
| third | 13 | **13.6** | 19.6 |
| calmest | 8 | 13.1 | **10.8** |

So the choice is not settled by a margin worth defending on one batch; both are reasonable, and the
alternative needs no file (`normalization.profile=xy_p99_global`). For reference, the same protocol on
an MPM batch of near-identical per-frame motion reaches 13 mm, so water remains markedly harder for
this model.

Caveat for anyone reproducing those numbers: they predate this domain. They ran through an adapter
that labelled water as MPM's sand (material 2), which also gave it sand's 256 coarse points in the
UNet. The batch itself is bit-identical to what `water3d_overfit_fixture.py` builds today.

Receipts, per-window breakdowns and the renders: `/data/user_data/robertre/scratchpad/liquid_water3d_0911/RESULTS.md`.

## Limitations and open questions

- **Walls are implicit.** Water-3D has no boundary particles, and the domain adds no
  distance-to-wall feature (the reference graph network in the paper has one). A model can still
  learn this box, because every trajectory uses the same one at the same place, but nothing
  transfers to another container. A pre-water checkpoint has no idea the walls exist: it pours
  water straight through them.
- **Particle subsets.** A window carries 4,096 of up to 13,248 particles, so the model never sees
  the full fluid; neighbours outside the subset act only through the ones inside it.
- **No zero-shot from the four-domain checkpoints.** They have a 6-row material table and no
  water entry in their stats.
- **The UNet's coarse-point budget for water** is unvalidated. The command above uses 1,024, which
  keeps MPM's ratio of about four particles per coarse point for a 4,096-particle window; every
  overfit arm so far ran at 256, inherited from sand.
