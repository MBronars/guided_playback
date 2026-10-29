# SIM1 folding collection: data format and training handoff

> **Status note, 2026-09-19.** Since this handoff was written the collection has been frozen for training as
> `pwm-data-v1/folding` (4,510 train clips, 236 test clips) and its thinned variant `folding_8mm`; the folding adapter,
> the index loader and the scales are on `robert/clean`. [DATASETS.md](DATASETS.md) is the current guide; this document
> remains the reference for the collection's format, fields and provenance.

Updated September 16, 2026 (Eastern time). This describes the **new 1,000-garment
collection**, not the legacy `cloth_curated_med` corpus. The schema below was
checked against a retained test export and the current exporter/loader code.

## 1. Quick start and readiness

Use the retained split **indices**, not a recursive glob of NPZ files. One record
is one complete action trajectory on one garment. All actions on a garment stay
in the same split. Files contain actual simulated particle positions, not model
predictions or RGB/depth observations.

The data are suitable for inspection and initial experiments with a verified
adapter. They are **not yet a drop-in, formally released replacement** for the
old training corpus. `accepted_for_training: false` is intentional: numerical
screening and candidate export do not certify a perfect fold or complete the
training integration. See sections 7 and 10 before launching a large run.

Collection completed cleanly at **09:26 EDT on September 16**. All 1,000 planned
garments were processed. Final retained membership is:

| Split | Clips | Garments with retained clips | Usable states |
|---|---:|---:|---:|
| Train | 4,510 | 939 | 4,068,020 |
| Test | 236 | 49 | 212,872 |
| Total | 4,746 | 988 | 4,280,892 |

Of 5,000 planned action slots, 254 were not retained. Twelve garments yielded no
retained clips. The final split indices were checked for unique clip keys,
disjoint source garment membership, and existence of every referenced file.
These metadata checks do not substitute for the model-level validation below.

See the [six-example GIF gallery](SIM1_FOLDING_GALLERY.md) to inspect saved test
motions across garment families and action variants.

A separate [10,000-cloth-point before/after preview](SIM1_CLOTH_10K_PREVIEW.md)
explores reducing model input size. It has **not** modified this full-resolution
dataset or its indices; the current maximum remains 32,739 cloth particles
(34,739 including both grippers).

**Final candidate root:**

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-noon-20260916-v4
```

Read these files below that root:

```text
production/train-INDEX.json
production/test-INDEX.json
production/STATUS.json
PLAN.json
```

`DONE.json` confirms `all_1000_attempted: true`; `REUSE-VERIFIED.json` records
successful inherited-data verification. The final indices include inherited
files by absolute path, including files in
v1/v2/v3 directories. Do not concatenate generation indices: that double-counts
inherited clips. Older directories must stay available while indices reference
them. `STATUS.json` alone is not a liveness check after a hard kill.

For reproducible experiments, copy the chosen index JSON to the experiment's
metadata directory, record its SHA-256 and the source `PLAN.json` hash, and keep
that snapshot fixed. This completed run's indices are no longer growing. During
an active collection, live indices grow and the two split files
are updated separately: if snapshotting during collection, read both and verify
disjoint garment/source membership against the plan rather than assuming one
atomic cross-split snapshot. Check permissions from the colleague's account;
directory placement alone does not prove read access.

## 2. Splits and action inventory

The frozen plan contains **950 train garments and 50 held-out test garments**,
with five attempted actions each: **5,000 attempts**, not 5,000 guaranteed retained
clips. Rejected plans or trajectories do not enter the retained indices. The
number of processed garments is not necessarily the number with retained clips.

| `world` | Planned action family | Variation |
|---|---|---|
| 0 | `left-inward` | 0 |
| 1 | `right-diagonal` | 0 |
| 2 | `two-hand-half` | 0 |
| 3 | `left-inward` | 2 |
| 4 | `right-diagonal` | 2 |

`world` is the action slot within the garment's five-world simulation batch,
**not** a globally unique episode ID. Use `(garment_id, world)` as the unique
clip key. Batched spatial offsets have already been removed from exports.

Do not randomly split clips or windows: actions from the same garment would
leak across splits. `PLAN.json` rows also record `source_id`; verify train/test
source identities are disjoint. There is no separate validation split in this
collection. For tuning, reserve a documented garment-level subset of **train**,
not the held-out test set. Fit normalization on train only.

## 3. Index contract

Each split index contains `split`, `records`, `native_fps: 30`, and
`accepted_for_training: false`. Important fields in each record:

| Field | Meaning |
|---|---|
| `file` | Absolute path to the selected `loader-candidates/action-N.npz` |
| `sha256` | SHA-256 of that NPZ file's bytes |
| `garment_id`, `split`, `world` | Identity and frozen split/action assignment |
| `frames` | 902 usable saved states |
| `particles` | Total point count: cloth plus both grippers |
| `cloth_particles` | Native garment vertex count; varies by garment |
| `gripper_points_per_hand` | 1,000 |
| `original_raw_frames` | 932 saved states before settling exclusion |
| `usable_native_start_index` | 30 |

Use `file` exactly; do not rebuild its path under the newest root. A completed
garment directory contains `RESULT.json` and `SELECTED.json`, including per-action
quality evidence. Other NPZ files in that directory may be rejected candidates,
raw chunks, templates, or diagnostic exports. File existence is not selection.

## 4. Selected NPZ schema

Load with `numpy.load(path, allow_pickle=False)`. Let `T=902`, `Nc=n_cloth`, and
`N=Nc+2000`. There is no batch dimension in the file.

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `particle_pos` | `[T, N, 3]` | float32 | Canonical-frame positions in **metres**, unnormalized |
| `obj_ids` | `[N]` | int64 | 0=cloth, 1=left gripper, 2=right gripper |
| `valid_particles` | `[N]` | bool | All true in these unpadded exports |
| `n_cloth` | scalar | int64 | Number of native cloth vertices |
| `dt` | scalar | float32 | `1/30` seconds |
| `state_time_s` | `[T]` | float64 | Absolute time in the active simulation |
| `source_state_index` | `[T]` | int64 | Native active-state indices, 30 through 931 inclusive |
| `ee_link_ids` | `[1000]` | int32 | Link membership for the point ordering of **each** gripper |

Particle order is `[all cloth, 1000 left-hand points, 1000 right-hand points]`.
Point identities persist over time. Do not independently resample/reorder every
frame. Each garment is one cloth object (`obj_ids == 0`), even if a visualization
colors its panels or material points differently. Object IDs do not encode
garment category, panel membership, physical parameters, or mesh connectivity.

`ee_link_ids` uses contiguous segments `[0 x 200, 1 x 400, 2 x 400]`: hand,
first finger, second finger. Apply the same ID ordering separately to each
1,000-point gripper block. The left and right points were sampled from each
arm's own geometry; do not assume identical coordinates across arms. Per-garment
`loader-candidates/gripper-template-1000.npz` preserves link-local points and
triangle/barycentric lineage. Cloth mesh topology is not in the selected NPZ;
follow the raw build/input provenance if a mesh renderer needs it.

Verified example: test garment `tee__tee_1MWCTRN0EU`, action 2 has `Nc=13659`,
`N=15659`, and `particle_pos.shape == (902, 15659, 3)`. The position array alone
is about 162 MiB when decompressed. Particle counts can exceed the old corpus's
12,000-cloth-point cap; do not copy that cap into a new adapter.

## 5. Coordinates, time, and simulation

The selected export has already applied:

```text
p_canonical_metres = p_world_metres + [-0.65, 0.0, -0.80]
```

The table top is at canonical `z=0`; particles can sit slightly above it due to
collision thickness. Axes are not rescaled or rotated. The canonical origin is
the legacy-compatible origin, not necessarily the enlarged table's center.
Do not translate a second time, recenter each frame, or interpret metres as mm.

The active run is native **30 Hz, 24 substeps, 16 VBD solver iterations** in
Newton/Warp. The physics substep is `1/720 s`. It saves 932 states for 931
transitions, covering `0..31.033333 s`. The first physical second is excluded
once at export, leaving states `30..931`, with times `1..31.033333 s`.

Thus each selected clip has **902 states, 901 observed transitions**, spanning
30.033333 seconds between its first and last states. It is already at 30 Hz:
**do not apply the old 60-to-30-Hz stride, or exclude another settling second.**
Time zero may be subtracted for plotting, but retain source time/index metadata.
Never create a transition across episode boundaries or beyond the terminal state.

This collection uses the reviewed 30-Hz throughput/quality tradeoff. It is not
claimed numerically equivalent to the older 60-Hz simulations.

## 6. Minimal verified read and window example

This is a NumPy inspection/window example, **not** a complete trainer adapter.
It leaves normalization and material/pose conditioning to explicit training code.
Change `INDEX` to the frozen manifest snapshot chosen for your experiment.

```python
import hashlib
import json
from pathlib import Path
import numpy as np

INDEX = Path('/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-noon-20260916-v4/production/test-INDEX.json')
manifest = json.loads(INDEX.read_text())
record = manifest['records'][0]
path = Path(record['file'])

# Verify once during dataset preparation; avoid rehashing the corpus every epoch.
h = hashlib.sha256()
with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
        h.update(chunk)
assert h.hexdigest() == record['sha256']

with np.load(path, allow_pickle=False) as z:
    p = z['particle_pos']
    ids, valid = z['obj_ids'], z['valid_particles']
    nc, dt = int(z['n_cloth']), float(z['dt'])
    times, source_indices = z['state_time_s'], z['source_state_index']
    links = z['ee_link_ids']

assert p.shape == (record['frames'], record['particles'], 3)
assert p.dtype == np.float32 and np.isfinite(p).all()
assert len(p) == 902 and p.shape[1] == nc + 2000 and valid.all()
assert np.array_equal(ids, np.repeat([0, 1, 2], [nc, 1000, 1000]))
assert np.array_equal(links, np.repeat([0, 1, 2], [200, 400, 400]))
assert np.array_equal(source_indices, np.arange(30, 932))
assert np.allclose(times, source_indices / 30) and np.isclose(dt, 1 / 30)

C, F, start = 2, 16, 0  # deterministic prefix example, not a full-motion evaluation
assert 0 <= start <= len(p) - (C + F)
window = p[start:start + C + F]                  # [18, N, 3], metres
deltas = window[1:] - window[:-1]               # [17, N, 3], metres/transition
target_deltas = deltas[C - 1:]                  # [16, N, 3]
action_mask = valid & (ids != 0)
actions = target_deltas * action_mask[None, :, None]
assert np.count_nonzero(actions[:, ids == 0]) == 0
print(record['garment_id'], record['world'], window.shape, actions.shape)
```

For independent loading tests, use train clips instead of treating this example
test clip as training data. Compressed NPZ is not a time-sliceable memory map:
accessing `particle_pos` decompresses that array. Avoid repeatedly decompressing
the same whole episode for every sampled window. Use a bounded cache or a tested
streaming/chunked conversion that preserves exact indices, values, and split IDs.

## 7. Training contract and current integration gaps

For a window of `C+F` states, the current `TrajectoryBatch` contract uses:

- `positions`: `[B, C+F, N, 3]` in canonical metres before normalization.
- `actions`: `[B, F, N, 3]`, achieved **gripper displacement**, zero on cloth.
- `action_mask`: `[B, N]`, true only on valid gripper points.
- `object_ids`, valid lengths/masks, explicit material fills and normalizers.
- Optional pose/link/articulation fields for models that require those channels.

For future step `f`, action is `p[C+f] - p[C+f-1]` on grippers only. This is
displacement in metres, **not velocity**: do not divide by `dt` for the existing
delta target. The recorded trajectory supports achieved-motion-conditioned
prediction; it is not automatically the same as conditioning on commanded IK
targets. Do not leak future cloth motion into actions. Prediction/loss masks
must follow the selected model contract; this guide does not redefine them.

The current main-branch `UnifiedClothDataset` still expects legacy WebDataset
`.pyd` fields and has legacy roots/sidecars. New indexed NPZ files are **not
automatically discovered** by pointing the existing WDS glob at their directory.
In particular:

1. Add a tested index-aware NPZ adapter or materialize a new WDS release while
   preserving units, clocks, keys and frozen membership. The old materializer
   would apply an extra stride/translation and imposes the old size cap.
2. Use this export's `ee_link_ids`. The old external sidecar is keyed by legacy
   episode names and documents 200/401/399 hand/finger counts; these exports use
   200/400/400. Old sidecar lookup may skip new clips entirely.
3. For articulated models, validate/build the correct reference hand and finger
   axes from the new templates. The old reference assumes its own point ordering
   and is not interchangeable merely because both hands have 1,000 points.
4. Handle ragged `N` by compatible bucketing or explicit padding/masks. Do not
   treat padding as cloth or let it contribute to losses/actions.

Cloth material constants are not stored in the selected NPZ. The current loader
uses explicit cloth/gripper fills; those are conditioning conventions, not
measured material labels. To condition on actual simulation parameters, obtain
them from the pinned generation/build provenance and define a tested mapping.

## 8. Normalization, sampling, and evaluation

Selected files are **not normalized** and have no synthetic training noise added.
Apply the experiment's resolved normalization exactly once. For checkpoint
evaluation or cleanup parity, preserve that checkpoint's numeric stats; do not
silently refit because the dataset changed. For a new-data experiment, train-only
calibration can produce a new versioned profile after checking coordinate/delta
ranges, clipping/saturation, and the enlarged workspace. Neither legacy per-domain
nor global scales are automatically certified for this collection. See
[the normalization contract](../stats/README.md).

Keep windows within one clip and retain context before each future interval.
The current loader uses motion-aware train sampling but deterministic prefix
evaluation. A prefix can be mostly approach/near-static motion; it is not a
full-fold benchmark. For meaningful comparisons, freeze garment/action membership,
window starts or full-rollout policy, seeds, stats, and physical error units.
If autoregressing with ground-truth future gripper motion, state that conditioning
explicitly. Test garments must not be used to fit stats or select checkpoints.

## 9. What the collection checks do—and do not—mean

Only selected records enter the indices. The pipeline checks complete raw
replays and hashes, finite states, substep speed/height/bounds guards, batched-world
isolation and contact capacity, plus per-action screening:

- Post-warmup stretch checked at every saved state: p99 stretch above 1.20 is
  a warning; above 1.30 is quarantined. A retained clip may have a warning.
- Global RMS-radius ratio below 0.10 is rejected as severe contraction.
- Proper/uncertain self-intersection pairs must be clear at sampled milestones.
- Each active hand must pass the near-finger lift/carry proxy (at least five
  material vertices, at least 20 mm lift and 30 mm carry).

These are operational thresholds, not proof of physical accuracy. Geometry
screening is discrete, not continuous; full robot-surface intersection is not
certified. Lift/carry is not successful, neat garment folding. Some retained clips
show partial folds or dragging. Review `RESULT.json`/`SELECTED.json` and rendered
motion if task success matters. Rejections must not be silently restored to meet
a target dataset size.

Production omitted routine GIF rendering. A [six-example test gallery](SIM1_FOLDING_GALLERY.md)
was rendered afterward from selected files. The earlier single test example is
available at:

```text
/data/user_data/robertre/particle_world_model/.worktrees/collection-scale/runs/folding_preview_20260915/test_tee_two_hand.gif
```

It shows the actual selected NPZ, no resimulation, fixed top/side cameras, 5-Hz
display at real-time speed. Its `RENDER.json` records the source hash and display
subsampling. Visualization colors are not additional object IDs.

## 10. Before sharing a training release

- Freeze the retained indices and source-plan hashes; check selected file hashes,
  unique clip keys, garment/source split disjointness, and colleague read access.
- Run the exact consumer adapter on representative train/test garments and size
  extremes; verify time alignment, masks, link/pose reconstruction and stats.
- Run one forward/backward training smoke test and a fixed-batch overfit/rollout
  check using the intended Transformer or UNet configuration.
- Review a small gallery across garment families, actions and stretch warnings.
- Publish the manifest/version, adapter invocation, normalizer snapshot and
  evaluation protocol together. Keep the held-out test set held out.

This document and its NumPy example do **not** claim those model-level handoff
checks have already passed. Collection is complete; a frozen subset can be used
for initial testing while the consumer adapter is validated.

## 11. Provenance and legacy-data distinction

The relevant collection sources are currently in `.worktrees/collection-scale/`:
`launch/folding_final_export.py`, `collection/sim1/actual_export.py`,
`launch/folding_fps_audit.py`, `launch/folding_scale_policy_v2.py`, and the
recovery/continuation launchers. Use the run's `PLAN.json` source hashes, not an
assumption that a future checkout is identical. The main training contracts are
[`data/unified_dataset.py`](../data/unified_dataset.py) and
[`data/batch.py`](../data/batch.py).

Legacy original NPZ corpus:

```text
/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/cloth_curated_med/
```

Legacy materialized corpus:

```text
/data/group_data/katefgroup/VLA/3DPCWM/unified_v1/cloth/{train,val}/
```

That corpus has different garment membership, assembly history, preprocessing,
gripper sampling and exclusion manifests. Its cloth blocklists should remain
enabled when reproducing old results, but must not substitute for the new
collection's retained indices. The older
[unified dataset spec](unified_dataset_spec.md) is historical context, not the
authoritative preprocessing recipe for these new NPZ exports.
