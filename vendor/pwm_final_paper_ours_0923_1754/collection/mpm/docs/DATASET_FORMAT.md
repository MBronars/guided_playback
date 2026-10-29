# MPM dataset format and training handoff

Verified against an actual fresh-campaign TAR, the exporter, and the current
clean-branch loader on September 16, 2026. This documents **passive MPM first**;
dragging differences are explicit in section 8. It supersedes the historical
MPM format assumptions in `docs/unified_dataset_spec.md`, not other domains.

**September 18 integration update:** the new training release uses indexed
`mpm_fresh` and `mpm_drag` domains under
`/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/`.
The TAR fields and units below remain the underlying data contract; the legacy
`mpm` reader examples are compatibility checks, not the new release's preferred
training entry point. For generated evaluation categories, the exact
`INDEX.json` bridge, collections normalization artifact, and current loader
checks are documented in [Evaluation collection](EVALUATION_COLLECTION.md#current-indexed-domains).
Do not run a random train/validation index split on an already frozen test set.

## 1. What to train on

Use reviewed **WebDataset `.tar` shards** beneath `datasets/train`, not raw HDF5,
the asset bank, or a campaign's top-level directory. Exported positions are in
physical metres; the unified adapter handles the coordinate shift and attaches
normalization. Do not normalize or center files yourself.

The completed passive campaign is:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-operator-campaigns-v1/fresh-10432785-v4
```

All four disjoint roots below are needed for the complete campaign. Paths are
relative to the above directory; counts are the final reviewed-receipt snapshot,
not a full release hash audit or a count of other campaigns.

| Relative training root | Retained clips |
| --- | ---: |
| `data/collection/datasets/train` | 22,087 |
| `recovery-001/salvaged/datasets/train` | 160 |
| `recovery-001/collection/datasets/train` | 16,370 |
| `recovery-002/collection/datasets/train` | 8,671 |
| **Total** | **47,288** |

Each passive clip has **49 saved states**, i.e. **48 transitions**. Total:
**2,317,112 saved frames**, not that many independent clips. Training windows
overlap; they must not be counted as additional independent simulations.

These are 1–4-object **training** scenes. They are not validation/test data.
The shape bank is separate:
`/data/group_data/katefgroup-ssd/world_model/particle_world_model/assets/mpm/fresh-10k-v1`.
Its train/validation/test UID cohorts are defined by `SPLITS.json`. Availability
of held-out shapes does not mean held-out trajectories have been generated.

## 2. Directory and episode identity

```text
datasets/train/
  train-84718/
    MANIFEST.json
    00000.tar
    00001.tar
    ...
  train-84719/
    MANIFEST.json
    00000.tar
    ...
```

`MANIFEST.json` records the retained count, shard hashes, and the corresponding
chunk-completion hash. Full quality/loader receipts live outside the training
root at `collection/chunks/train-<cycle>/COMPLETE.json`. The salvage case has its
review receipts under `recovery-001/recovered/train-<cycle>/` instead.

Inside a TAR, one sample key groups several files, e.g.:
`batch_00000000.particle_pos.pyd`, `batch_00000000.object_ids.pyd`, and
`batch_00000000.provenance.json`. **One sample key is a pre-batch of episodes**,
not one episode. Export defaults are up to four episodes per pre-batch and eight
pre-batches per shard, grouped by compatible trajectory shape. Tails can be smaller.
Numeric shard names and sample keys repeat in different directories.

Use `provenance.json[row].episode_id` as the stable episode identity. A precise
stored location is `(full shard path, sample key, batch row)`. Each provenance
row also includes asset IDs, raw SHA-256, task/split/population, object cohort,
split-manifest hash, state indices/times, floor, and selection-policy provenance.
Do not split rows/windows from the same episode across train and validation.

`.pyd` files contain **pickle protocol-4 NumPy arrays**, not `.npy` files and not
Python extension modules. WebDataset decodes them. **Only unpickle trusted
project data:** pickle can execute code.

## 3. Stored passive sample schema

`B_s` = episodes in the stored pre-batch; `T = 49`; `K` = objects;
`N = 1024 × K` in this campaign, hence 1024/2048/3072/4096 particles. Particle
index/order is stable through the clip. Object IDs are scene-local, not asset UIDs.

| Decoded key | Shape | dtype | Meaning / units |
| --- | --- | --- | --- |
| `particle_pos.pyd` | `[B_s,T,N,3]` | float32 | Native world XYZ, metres |
| `particle_vel.pyd` | `[B_s,T,N,3]` | float32 | Native instantaneous velocity, m/s |
| `initial_vel.pyd` | `[B_s,N,3]` | float32 | Velocity at stored state 0, m/s |
| `material_properties.pyd` | `[B_s,N,5]` | float32 | `[E, nu, density, mat_type, yield_stress]` |
| `object_ids.pyd` | `[B_s,N]` | int32 | Object membership, `0..K-1` |
| `valid_particles.pyd` | `[B_s,T,N]` | bool | All true in these unpadded exports |
| `n_objects.pyd` | `[B_s]` | int32 | Number of objects per episode |
| `grab_velocity.pyd` | `[B_s,N,3]` | float32 | All zero for passive clips |
| `is_grabbed.pyd` | `[B_s,N]` | int32 | All zero for passive clips |
| `drag_active.pyd` | `[B_s,T]` | bool | All false; terminal state has no outgoing transition |
| `provenance.json` | list of `B_s` dictionaries | JSON | Episode identity, timing, source and review metadata |

Material columns:

- `E`: Young's modulus in Pa; this campaign samples 10,000 / 100,000 / 1,000,000 Pa.
- `nu`: dimensionless Poisson ratio, 0.2 in this campaign.
- `density`: kg/m³, 1000 in this campaign.
- `mat_type`: **0 elastic, 1 plasticine, 2 sand**; integer-valued despite the float array.
- `yield_stress`: Pa; 10,000 for plasticine, zero for the other two types here.

The current passive model adapter consumes columns 0–3 and **does not expose
yield stress as a model input**. Its older four-column docstring describes those
consumed channels, not the actual five-column fresh export. Material values are
not pre-log-transformed or standardized on disk.

## 4. Units, coordinates, time and targets

- Native arrays are z-up in the simulator's world frame. The recorded floor is
  `floor_m = 0.1` m. `UnifiedMPMDataset` adds **(-0.75, -0.75, -0.10) m** once.
  `batch.positions` remains in canonical **metres**, not normalized coordinates.
- Exact recorded times are in each provenance row's `state_time_s`. For this
  passive campaign, spacing is **0.0417 s**, duration **2.0016 s**. Do not replace
  it with exactly `1/24`, infer it from GIF playback, or confuse saved spacing
  with the much smaller native integration timestep.
- The target is **adjacent-frame displacement**, not instantaneous velocity:
  `delta[t] = position[t+1] - position[t]`, in metres per saved transition.
  `particle_vel.pyd` is diagnostic data, not the current model's prediction target.
- Default recipes use `C=4` context states and `F=16` future states, a 20-state
  window. The first future delta starts at the last context state. The stored
  49-state trajectory remains intact; the loader chooses windows at read time.
- Uniform training window selection shares one start within a stored pre-batch.
  Optional movement-weighted/gated selection is per episode. Evaluation's default
  window is sample-key-seeded; `first_window=True` explicitly requests the prefix.
  Record the window policy when comparing runs. `future_len=-1` reads all states.

Normalization is selected separately through `normalization.profile`:
`xy_p99_per_domain` or `xy_p99_global`, optionally with an explicit
`normalization.artifact`. Both live in `stats/normalization.json`; no fitting occurs
automatically. Motion bounds are metres **per stored transition**, not m/s;
position bounds are ±0.8 m. The loader attaches normalizers; downstream model code
uses them. Do not normalize `batch.positions` a second time. Frozen MPM scales
currently coincide between the two profiles, but this need not hold after refitting.
See [normalization](../../../stats/README.md) for checkpoint and calibration rules.

## 5. Read one real batch on CPU

Run from the repository root in an environment with the project's PyTorch,
NumPy and WebDataset dependencies. This example reads only one completed chunk;
it neither trains a model nor modifies the dataset.

```python
import torch
from data.unified_dataset import UnifiedMPMDataset

root = (
    "/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/"
    "mpm-operator-campaigns-v1/fresh-10432785-v4/"
    "recovery-002/collection/datasets/train/train-84718"
)
dataset = UnifiedMPMDataset(
    wds_location=root,
    normalization={"profile": "xy_p99_per_domain"},
    context_len=4, future_len=16,
    training=False, shuffle=False,
)
dataset.first_window = True
batch = next(iter(dataset))
assert batch.positions.ndim == 4 and batch.positions.shape[1] == 20
assert batch.actions.shape[1] == 16
assert torch.isfinite(batch.positions).all()
assert not batch.action_mask.any() and not batch.actions.any()
print("positions:", tuple(batch.positions.shape), "canonical metres")
print("actions:", tuple(batch.actions.shape), "metres per transition")
print("object IDs:", batch.object_ids.unique().tolist())
print("material IDs:", batch.mat_type.unique().tolist())
```

The output contract is [TrajectoryBatch](../../../data/batch.py): positions
`[B,C+F,N,3]`, actions `[B,F,N,3]`, action mask `[B,N]`, object IDs `[B,N]`,
valid lengths `[B]`, and E/nu/rho/material ID `[B,N]`. Passive actions are zero;
initial object velocities are not actuators. The training stream can rebatch
stored samples by compatible particle count to the requested `data.batch_size`.
Do not add an extra outer batch dimension with a default PyTorch DataLoader.

## 6. Combine the four roots without copying data

**Current limitation:** MPM's `data.roots` override requires exactly one train
directory and one val directory. It does not accept four train paths. The reader
recursively discovers TARs, so a new index directory containing four directory
symlinks works. Do **not** point it at the whole campaign: that would also discover
qualification, intermediate, recovered duplicates and other diagnostic TARs.

Optional preparation below creates only a new local index; no shared data is
changed. It refuses to overwrite an existing index. For team training, place an
equivalent reviewed index in an agreed shared release location instead of using
a private checkout path. Keep the index membership fixed for a reproducible run.

```python
from pathlib import Path

campaign = Path(
    "/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/"
    "mpm-operator-campaigns-v1/fresh-10432785-v4"
)
sources = {
    "original": "data/collection/datasets/train",
    "salvaged": "recovery-001/salvaged/datasets/train",
    "recovery1": "recovery-001/collection/datasets/train",
    "recovery2": "recovery-002/collection/datasets/train",
}
targets = {name: campaign / relative for name, relative in sources.items()}
assert all(path.is_dir() for path in targets.values())
index = Path("validation/data-links/mpm-fresh-v4/train").resolve()
index.mkdir(parents=True, exist_ok=False)
for name, target in targets.items():
    (index / name).symlink_to(target, target_is_directory=True)
print(index)
```

The index is a reader convenience, **not** a release-approval or integrity audit.
Preserve directory names: flattening the `00000.tar` files causes collisions.

## 7. Train: explicit roots, stats and held-out validation

Choose and verify a held-out validation export first. This passive campaign's
four roots are **all training**, so none can serve as its held-out val root.
For new-shape evaluation, compare provenance `asset_ids` against the frozen UID
splits; a distinct directory alone does not establish an object holdout. 5–7-object
evaluation is a separate OOD population. Historical validation data can be an
explicit baseline, but is not automatically a fresh-bank held-out certificate.

After creating the index above, set `MPM_TRAIN_ROOT` to its printed absolute path
and `MPM_VAL_ROOT` to your reviewed validation directory. The following launch
deliberately refuses to run until those choices are supplied:

```bash
: "${MPM_TRAIN_ROOT:?Set the absolute reviewed training-index path}"
: "${MPM_VAL_ROOT:?Set the absolute reviewed held-out validation path}"
python train.py --config configs/unet.yaml \
  run_name=mpm_fresh_unet \
  'data.domains=[mpm]' \
  "data.roots.mpm.train=$MPM_TRAIN_ROOT" \
  "data.roots.mpm.val=$MPM_VAL_ROOT" \
  data.batch_size=4 data.context_len=4 data.future_len=16 \
  normalization.profile=xy_p99_per_domain \
  train.steps=40000 train.stop_after_step=0
```

Use `configs/transformer.yaml` for the unified transformer. This is a specialist
dataset selection, not a different model architecture. Add other supported
domains to `data.domains` for joint training; unnamed roots retain their registered
corpora. Architecture options, such as UNet bidirectional temporal processing,
must be chosen explicitly if matching a prior run. This command is a new training
recipe, **not** the saved-batch overfit replay protocol.

Overridden domains use **only explicitly named exclusions**, not the historical
registered blocklists. The reviewed fresh shards already exclude rejected raw
clips; add release-specific exclusion files only if an additional review requires
them. Resume preserves roots, exclusion identities and normalization snapshots;
do not repoint a running checkpoint at a different dataset or silently change
symlink membership. Root identity alone does not freeze changing directory contents.
See [supported training](../../../docs/SUPPORTED.md#explicit-dataset-roots).

## 8. Dragging is a separate, explicit contract

Do not mix passive and dragging shards under the passive reader or derive drag
commands from future ground-truth particle motion. Select the drag adapter with
`data.roots.mpm.task=drag`; use its own reviewed train/val roots.

The current sequence adapter accepts versioned provenance and expects:

- `task: drag` plus `action_schema` in each provenance row. Gentle v4 is
  `mpm-drag-command-sequence-v4`; older supported schemas retain their own meaning.
- `grab_velocity.pyd`: **`[B_s,T-1,N,3]`**, m/s, unlike the passive static shape.
- `is_grabbed.pyd`: `[B_s,N]`, the fixed selected patch; `drag_active.pyd`:
  `[B_s,T]`, with the terminal entry false. Zero velocity during an active hold
  does not make the clip passive.
- `T=49/81/113` for 1/2/3 events; exact stored spacing 0.0417 s. Settling frames
  are separate evidence, not extra driven training frames.
- Actions are commanded velocity × recorded transition duration. The adapter
  emits temporal `action_active` and `context_action_active` masks aligned to
  the sampled window. Gentle v4 is sand-free and requires fragmentation-filter
  provenance and strict wall filtering. Its supported speed bound is 0.25 m/s.

Use [dragging recovery](DRAG_RECOVERY_20260915.md) and
[gentle-v4 release](DRAG_GENTLE_V4_RELEASE.md) for those dataset paths and gates.
The passive count in section 1 does not include dragging.

## 9. Acceptance and reproducibility checklist

- Train only from atomically published shards with matching manifests and completed
  quality/control/loader receipts. Excluded raw clips never become training samples.
- `accepted_for_training: false` is deliberately retained in diagnostic/staging
  provenance. Passing local filters and loader checks is **not formal release
  promotion or proof of physical fidelity**. Do not rewrite this flag to manufacture
  approval. Record the staged-data decision explicitly in an experiment handoff.
- Preserve the exact shard membership/hashes, stable episode IDs, asset splits,
  selection policy, source/capsule identity and exclusion lists. Audit all four
  roots when assembling this campaign; the original collection alone is incomplete.
- Record C/F, window policy, normalization artifact/profile/checksum, model config
  and RNG settings. Never fit normalization on validation/test data.
- Report clips, saved states and transitions separately. GIF playback rate and
  number of windows sampled during training are not dataset frame counts.

Code references: [export schema](../validate.py), [passive adapter](../../../data/dataset.py),
[unified adapter/root validation](../../../data/unified_dataset.py),
[drag adapter](../../../data/mpm_drag.py), [batch contract](../../../data/batch.py).
Operational provenance: [runbook](RUNBOOK.md), [passive recovery](RECOVERY_10432785.md),
[asset bank](ASSETS.md).

## 10. Checks performed for this document

The CPU example above ran successfully on the named real chunk: positions
`[4,20,1024,3]`, actions `[4,16,1024,3]`, all finite, no passive actuation.
The four-root index example was executed in a temporary directory: the reader's
recursive discovery found **3,488 TARs with no duplicate resolved target paths**.
The temporary index was removed after the check; no permanent index or dataset
was created. Python examples parse, local links resolve, and the training command
passes shell syntax checking. No training was launched or full corpus rehashed.
