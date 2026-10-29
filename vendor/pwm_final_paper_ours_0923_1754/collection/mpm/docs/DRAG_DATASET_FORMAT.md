# MPM dragging dataset: format and training guide

This is the training-facing reference for the **gentle-v4 dragging collection**
on `robert/clean`. Checked against the exporter, adapter and actual published
TARs on September 16, 2026. No training or collection settings were changed to
write this guide.

## 1. Which data to read

Shared project prefix:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model
```

The fixed recovered source and separate collection roots are:

| Source | Path relative to the shared prefix | Use |
| --- | --- | --- |
| Recovered completed collection | `datasets/mpm-drag-gentle-v4-recovered-20260915/train` | Fixed set of 334 verified clips |
| Restarted collection | `staging/mpm-drag-gentle-v4-restart-20260915/data/train/train` | Published blocks; grows while collection runs |
| Additional dev-blackwellb collection | `staging/mpm-drag-blackwellb-20260916/data/train/train` | New independent run; use only blocks after publication |

Published blocks contain `block-XXXXXX/diagnostic-export/*.tar`. The repeated `train/train`
in the restart path means **split/evaluation role**; it is not a typo.
The loader recursively discovers TARs below a supplied root.

**Do not point training at the entire `staging/` tree or campaign bundle.** That
would also discover unapproved exports under `export-review/`. Do not use the
original `mpm-drag-gentle-v4-20260915/data` root: it includes an export whose old
review failed. Use the recovered copy instead, without including both copies.
Historical `robert_mpm_drag_setA` and `robert_mpm_v2_drag` shards are a different
contract and are not automatically compatible.

### Publication and reproducibility

New blocks are staged outside the training root, checked through both loader
profiles, and renamed into the published root only after success. Each has
`FILTER.json`, `COMPLETE.json`, `LOADER_REVIEW.json` and nested exporter receipts.
The recovered release has its own top-level `COMPLETE.json` and recovery evidence;
older recovered blocks need not have the newer `LOADER_REVIEW.json` filename.

The nested name `diagnostic-export`, and `accepted_for_training: false` in its
generic exporter metadata, are retained historical vocabulary. They are **not**
the production approval mechanism. Approval comes from the enclosing collection
gate, successful consumer review and publication/recovery receipts. Do not flip
those fields or treat any arbitrary diagnostic export as approved.

Before a training experiment, freeze a reviewed list of published blocks/shards
and their hashes in a separate train/validation release. The loader discovers
shards at construction; a growing directory is not a reproducible dataset
version and is not automatically refreshed. The MPM config currently accepts
**one directory per split**, not a list of independent roots. Combining the
sources above needs a deduplicated release/view with unique block names and
episode IDs; do not concatenate glob results from overlapping roots.

## 2. What one clip means

- One independently generated scene with **1–4 objects**, 1,024 surface particles
  per object: `N = 1024 × object_count`.
- Train assets from the immutable 10,000-shape bank. Current production uses
  homogeneous elastic or plasticine scenes; **no sand**.
- Native grid 256³; box size 1.5625 m; Z is up and raw floor is `z=0.1 m`.
- Current stiffness strata: 10 kPa, 100 kPa, 1 MPa.
- 96 settling transitions happen before the recorded motion. Their evidence is
  retained in raw files/journals but not appended to the training trajectory.
- **1, 2 or 3 events → 49, 81 or 113 saved motion states** (`T`), respectively.
  Each clip therefore has `T-1` outgoing transitions.
- Recorded interval is **0.0417 seconds**, not exactly `1/24`: current native
  clock is 834 steps of 50 microseconds per transition. Read recorded timestamps.
- Fixed object-local topmost 10% particle patch; pull, lift/hold/release and
  single-leg reverse families; continuously sampled event peak speeds
  0.05–0.25 m/s with ramps.
- Version: `mpm-drag-command-sequence-v4`. Mandatory settling, integrity,
  collapse, strict side/ceiling-wall and persistent-fragmentation filtering.
  Floor contact is allowed. Retention is not proof of calibrated physical truth:
  shared-grid contact artifacts and point-based fragmentation limits remain.

## 3. Training TAR schema

Each TAR contains WebDataset records named `batch_00000000.<field>`, etc.
**A record is already a small batch of clips**, not one clip. Let `B` be that
record's batch size (normally 4, possibly 1–3 in a tail). Clips in a record share
`T` and `N`. The exporter groups shapes and writes at most eight records/32 clips
per shard; short shards are normal. TAR count is not clip count.

`.pyd` entries are pickled NumPy arrays; only load trusted project shards.

| Field suffix | Shape | Dtype | Meaning / units |
| --- | --- | --- | --- |
| `particle_pos.pyd` | `[B,T,N,3]` | float32 | Raw world positions, metres |
| `particle_vel.pyd` | `[B,T,N,3]` | float32 | Observed particle velocity, m/s |
| `initial_vel.pyd` | `[B,N,3]` | float32 | Velocity at first recorded state, after settling |
| `material_properties.pyd` | `[B,N,5]` | float32 | `[E, nu, density, mat_type, yield_stress]` |
| `object_ids.pyd` | `[B,N]` | int32 | Zero-based object membership; not material IDs |
| `valid_particles.pyd` | `[B,T,N]` | bool | Validity; current exports have no particle padding |
| `n_objects.pyd` | `[B]` | int32 | Scene object count |
| `grab_velocity.pyd` | `[B,T-1,N,3]` | float32 | Prescribed velocity for each outgoing transition, m/s |
| `is_grabbed.pyd` | `[B,N]` | int32, 0/1 | Persistent selected-patch membership |
| `drag_active.pyd` | `[B,T]` | bool | Transition activity, with a final `false` padding entry |
| `provenance.json` | list of B objects | JSON | Episode identity, commands, timing and selection lineage |

Material units: `E` and `yield_stress` are Pa, density is kg/m³, `nu` is
dimensionless, material vocabulary is 0=elastic, 1=plasticine (2=sand exists in
the broader MPM vocabulary but is forbidden here). The current model batch
consumes E/nu/density/type; **yield stress is stored but not model-conditioned**.

Provenance includes `episode_id`, `raw_sha256`, `asset_ids`, `task`,
`action_schema`, `command_semantics`, `action.events`, `state_indices`,
`state_time_s`, `split`, `evaluation`, `object_cohort`, `split_manifest_sha256`,
`drag_material_policy`, `wall_selection_policy`, `collapse_policy` and
`fragmentation_policy`. Use **episode_id** for global identity: sample keys and
numeric shard basenames repeat between shards/blocks.

## 4. Actions, targets and hold/release

For transition `k`, the conditioning is:

```text
commanded_displacement[k] = grab_velocity[k] × (state_time_s[k+1] - state_time_s[k])
active_patch[k,n] = drag_active[k] AND is_grabbed[n]
prediction_target[k] = observed_position[k+1] - observed_position[k]
```

Commands are pre-P2G velocity overrides on selected material points, **not robot
poses, forces, or guaranteed achieved motion**. Never replace them with future
observed motion: that would leak the prediction target into conditioning.

| Phase | Patch selected? | Temporal active flag | Command velocity |
| --- | --- | --- | --- |
| Drive | Yes | True | Nonzero |
| Hold | Yes | True | Zero |
| Release | Yes | False | Zero |

Therefore zero action does **not** distinguish hold from release. Preserve both
static patch membership and temporal gates through batching, device moves and
autoregression. All deformable particles, **including selected particles**,
remain prediction targets; do not hard-clamp them to the commanded displacement.

## 5. Unified model batch and window alignment

Use `data.mpm_drag.UnifiedMPMDragDataset`, not the passive MPM adapter. The
passive adapter rejects active samples. The drag adapter produces domain `mpm`
and adds the fixed translation **(-0.75, -0.75, -0.10) m** to raw positions.
It does not center each object or scene. Translation does not change commands.

For context length `C`, future length `F`, and window start `s`:

| TrajectoryBatch field | Shape / alignment |
| --- | --- |
| `positions` | `[B,C+F,N,3]`, raw states `s:s+C+F`, shifted to canonical frame |
| `actions` | `[B,F,N,3]`, commanded displacements at transitions `s+C-1:s+C-1+F` |
| `action_mask` | `[B,N]`, persistent selected patch |
| `action_active` | `[B,F,N]`, activity AND selection for future transitions |
| `context_action_active` | `[B,C-1,N]`, activity AND selection at `s:s+C-1` |
| `future_deltas` | `[B,F,N,3]`, observed adjacent-frame displacements from last context state |
| `object_ids`, `E`, `nu`, `rho`, `mat_type` | `[B,N]` |
| `valid_lengths` | `[B]`, number of valid particles |

Physical positions/actions are not pre-normalized in the returned batch. The
batch carries the resolved position/delta normalizers; models/objective consume
them. Use the existing `xy_p99_per_domain` or `xy_p99_global` profile and save
the resolved snapshot with the run. Their **MPM particle scales currently
coincide**; they differ for other domains. Do not refit statistics merely to load
dragging, multiply velocities by 1/24, or normalize twice.

Default training windows are C=4/F=16. Uniform window sampling shares a start
across a stored batch. The low-level adapter also implements movement-weighted
per-row sampling, but the supported unified training factory rejects non-default
movement-sampling options; that is not an enabled training-config knob here.
Evaluation pins the draw using the WebDataset sample key (not globally unique
episode identity), so it is deterministic but not an independent random window
for every episode. A window can contain only hold/release; that is legitimate.
Use an explicit episode/window manifest for controlled evaluation or overfitting.

## 6. Run a CPU loader smoke check now

From the repository root, with the training environment activated. This reads
one batch from the fixed recovered dataset; it does not start training or touch
the collection. The flags make decoding errors fatal and avoid distributed shard
splitting for this single-process check.

```python
from data.mpm_drag import UnifiedMPMDragDataset

root = "/data/group_data/katefgroup-ssd/world_model/particle_world_model/datasets/mpm-drag-gentle-v4-recovered-20260915/train"
ds = UnifiedMPMDragDataset(
    root, normalization="xy_p99_per_domain",
    context_len=4, future_len=16, training=False, shuffle=False,
)
ds.strict_read = True
ds.no_shard_split = True
batch = next(iter(ds))
assert batch.positions.shape[1] == 20
assert batch.actions.shape[1] == 16
assert batch.action_active.shape == batch.actions.shape[:-1]
assert batch.context_action_active.shape[1] == 3
assert batch.domain == "mpm"
print("positions:", tuple(batch.positions.shape))
print("actions:", tuple(batch.actions.shape))
print("selected particles:", batch.action_mask.sum(dim=1).tolist())
print("future active particle-transitions:", batch.action_active.sum().item())
```

This exact example was executed successfully: positions `(4,20,1024,3)`,
actions `(4,16,1024,3)`, and 103 selected particles per clip. The sampled future
window had zero active particle-transitions, which is valid release behavior,
not evidence that the dataset is passive.

For complete verification, `runs/mpm-drag-recovery-20260915/check_published.py`
already read all 334 recovered clips under both profiles and checked exact raw
positions, commands and gates; receipt `PUBLISHED_CONSUMER.json`. It is a
no-clobber audit script, not a repeatable command that overwrites its receipt.

## 7. Training configuration — split preparation still required

The current campaign is **train-only**. There is no qualified new dragging
validation/test release here yet. The standard trainer requires distinct train
and validation roots; do not use passive validation, the same root twice, or
overlapping copied episodes to make it run.

A scene-held-out split from the same train asset bank can support development
validation if its episodes are disjoint. It is **not held-out-object evaluation**.
Held-out-object tests and 5–7-object dragging need separate collection/review.
The path validator checks directory overlap, not episode/asset leakage; audit
those identities when preparing the split.

After preparing fixed, disjoint approved roots, use the existing configs. These
are **templates with placeholder paths**, not launch-ready commands:

```bash
python train.py --config configs/transformer.yaml \
  run_name=mpm_drag_transformer \
  'data.domains=[mpm]' data.batch_size=4 \
  data.context_len=4 data.future_len=16 \
  data.roots.mpm.task=drag \
  data.roots.mpm.train=/ABSOLUTE/APPROVED_DRAG_RELEASE/train \
  data.roots.mpm.val=/ABSOLUTE/APPROVED_DRAG_RELEASE/val \
  normalization.profile=xy_p99_per_domain \
  train.action_dropout=0
```

For UNet, use `configs/unet.yaml` and **also set `model.unet_actions=true`**.
Its base config has that flag false. Configuration validation rejects dragging
without particle-action conditioning. Keep `train.action_dropout=0`; the
pose-only dropout path is not supported for dragging. Transformer defaults
already enable action conditioning (`model.no_action_cond=false`) and temporal
actuation embeddings. Do not pass `model.no_action_cond` to UNet: it is a
Transformer-only option and the UNet config validator rejects it.

`data.batch_size` is samples per process, not stored TAR records. Shape-bucketing
handles object counts/tails. Use a fixed future length when mixing clip lengths.
These are interface examples, not tuned optimization settings or an overfit
reproduction protocol. UNet's base config also retains a 40K stop; review the
training schedule rather than assuming the command creates a new tuned recipe.

Other supported domains can be included in `data.domains` with their own roots.
**Passive and active MPM cannot currently be selected as two independent entries
in the same unified configuration**: there is one `mpm` slot and one `task`.
Do not place passive and drag TARs in one directory to bypass that restriction.

## 8. Raw files are for auditing, not the training entry point

Each completed simulation raw is HDF5 `raw.h5`, with JSON manifest in attribute
`manifest_json`, schema/completion attributes and arrays including `x`, `v`,
`F_trial`, `C`, object/material fields, `selected`, `command_velocity`,
`transition_active`, `active_substeps`, `state_time_s`, settling histories and
native wall-contact evidence. Raw `transition_active` has length `T-1`; only
the TAR compatibility field pads it to T. Raw motion `x` is `[T,N,3]`, not batched.

`RESULT.json` records retention/rejection. A raw file existing, a subprocess
exiting zero, or a complete frame journal does not by itself authorize training.
Rejected raws and evidence are deliberately preserved. Settled checkpoints
preserve additional physical state; do not attempt to resume plastic simulation
from exported positions/velocities alone.

## References

- [Current release and protocol](DRAG_GENTLE_V4_RELEASE.md)
- [Recovery, approval evidence and live-status command](DRAG_RECOVERY_20260915.md)
- [Additional dev-blackwellb collection](DRAG_BLACKWELLB_20260916.md)
- [Contact limitations](DRAG_CONTACT_256.md) and [fragmentation screening](DRAG_FRAGMENTATION_FIX.md)
- Adapter: `data/mpm_drag.py`; root routing: `data/unified_dataset.py`
- Batch contract: `data/batch.py`; config gates: `utils/config.py`
- Serialization: `collection/mpm/validate.py::_batch_payload` and `export_diagnostic`
- Publication: `collection/mpm/drag/launch.py::reviewed_export`
