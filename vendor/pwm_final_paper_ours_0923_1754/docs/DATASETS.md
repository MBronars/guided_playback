# Current datasets

Verified **September 17, 2026**, updated the same evening for the merge and the pwm-data-v1 release. This guide maps the datasets used by the particle world-model project: what they contain, where they live, how they become training batches, and what the current evidence supports.

There are **six datasets in the recent collection-training effort**: fresh passive MPM, MPM dragging, folding, RL pushing, Yikai pick-and-place, and dexterous-hand manipulation. The repository also retains four earlier/default domains and the optional Water-3D domain. Similar names do not imply interchangeable data or normalization.

**Implementation status.** The main checkout is `robert/clean`. The five indexed names `mpm_fresh`, `mpm_drag`, `folding`, `yikai`, and `dex` landed there in merge `706e694` on September 17, 2026 (candidate `robert/dataset-readiness-0916` through `679a1d7`, which is the tested source snapshot v8 committed byte for byte; `data/mpm_drag.py` took the equivalent adapter change committed in `2b16f68`). Clean now registers `mpm`, `pickup`, `cloth`, `pushing`, `water`, `rl_push` and the five indexed domains; the indexed ones are opt-in and never selected by default. An indexed domain needs a train and a validation index directory, and the validation directory may be a release's `test` folder: **in pwm-data-v1 the test sets are also the validation sets** (small ones since 2026-09-19 for pushing, rl_push and yikai: 100 / 200 / 200 episodes, seeded subsets of the complete folders, which stay in the release as `test_full` report folders; user decision) (decision of September 17, 2026), so training-time curves and `best.pt` read them and the paper says so.

Some collection-specific format documents predate that integration and still describe training support as missing. Their storage schemas remain useful; the implementation and verification status below refer to the frozen, tested candidate.

## Dataset map

| Domain name | What it contains | Motion supplied to the model | Current role |
|---|---|---|---|
| `mpm_fresh` | New multi-object elastic, plasticine and sand dynamics | None; passive dynamics | Indexed domain on clean (merged 2026-09-17); pwm-data-v1 train folder built |
| `mpm_drag` | New elastic/plasticine scenes with driven material-point patches | Prescribed patch velocity × transition duration, plus hold/release gates | Indexed domain on clean (merged 2026-09-17); pwm-data-v1 train folder built |
| `folding` | New SIM1 garment collection with two articulated grippers | Recorded gripper motion | Indexed domain on clean (merged 2026-09-17); pwm-data-v1 train folder built |
| `rl_push` | RL-policy Panda pushes of reconstructed rigid objects | Recorded gripper motion | Opt-in domain on clean; pwm-data-v1 folders built 2026-09-18 (see [RL_PUSH.md](RL_PUSH.md)) |
| `yikai` | Full pick-and-place scenes, including distractors (the adapter drops the table) | Recorded gripper motion | Indexed domain on clean (merged 2026-09-17; adapter `c29fbb3` 2026-09-18); pwm-data-v1 folders built 2026-09-18 |
| `dex` | Five tactile teacher recordings with XHand1/Sharpa hands | Recorded hand-link/joint motion | Indexed domain on clean (merged 2026-09-17); pwm-data-v1 `dex` holds v1 (train only), `dex_v2` the 2026-09-18 regeneration with a held-out-object test, `dex_v2_filtered` that regeneration minus 625 episodes with simulator artifacts, `dex_v2_filtered2` a stricter filter (minus 2,934), the folder to train on (read with `--folder dex=dex_v2_filtered2`) |
| `mpm` | Earlier passive MPM v2 corpus | None | Existing default domain |
| `pickup` | Earlier successful place/convex/pick/upright collections | Recorded Panda gripper motion | Existing default domain; since 2026-09-19 also the pwm-data-v1 folder `pickup` (user decision: include it with a reasonable amount of placing) |
| `cloth` | Earlier materialized `cloth_curated_med` collection | Recorded gripper motion | Existing default domain |
| `pushing` | Earlier rod-and-block pushing corpus | Recorded rod motion only | Existing default domain |
| `water` | Water-3D liquid in a fixed closed box | None | Existing opt-in domain; separate normalization/material entry |

All these are simulator-derived geometry/trajectory datasets. Camera streams, when present in a source release, are not inputs to the current particle/pose training path.

## Release pwm-data-v1 (2026-09-17)

The training data has one home, with a train folder and a test folder per dataset and **no development split**:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/
```

| Dataset | Train | Test | How it was built |
|---|---|---|---|
| `mpm_fresh` | 47,288 episodes (872 chunks, 3,488 shards) | pending: unseen-shape export (also the validation set) | hard links |
| `mpm_drag` | 2,333 episodes (31 published blocks) | pending: drag evaluation export | hard links |
| `folding` | 4,510 clips, 939 garments | 236 clips, 49 garments (the official test) | hard links |
| `folding_8mm` | the same clips; cloth thinned to an 8 mm farthest-point spacing, at most 10,000 vertices per garment (54% kept) | the same 236 clips (56% kept) | hard links + subset files, built 2026-09-18 |
| `dex` | 8,865 episodes, five teacher shards plus `geometry/` (v1) | none: the held-out-object test is part of `dex_v2` | hard links, copied geometry |
| `dex_v2` | 19,913 episodes (built 2026-09-18): 14 boxes/cylinders, 100 objaverse meshes, screwdriver models 1-3 | 194 episodes on 24 held-out objects | hard links, copied geometry |
| `dex_v2_filtered` | 19,292 episodes: `dex_v2` minus 621 with simulator artifacts (built 2026-09-18) | 190 episodes on the same 24 held-out objects | hard links to `dex_v2`'s files |
| `dex_v2_filtered2` | 16,999 episodes: `dex_v2` minus 2,914, also the milder single-finger rides and all repose XHand1 on boxes (built 2026-09-18) | 174 episodes on the same 24 held-out objects | hard links to `dex_v2`'s files |
| `pushing` | 93,184 episodes, 364 shards | 100 episodes, 25 shards of one stored sample each (since 2026-09-19; the complete 10,148 episodes, 40 shards are `test_full`) | copies |
| `rl_push` | 1,085 objects, 51,555 episodes (204 excluded by the 2026-09-18 audit) | 200 episodes of the 117 held-out objects (since 2026-09-19; the complete 5,598 episodes, 15 excluded, are `test_full`) | copies, re-frozen 2026-09-18 |
| `yikai` | 11,015 episodes, 2,134 scenes (the candidate's train + development) | 200 episodes of 100 scenes (since 2026-09-19, hard links; the complete `wm_v2_eval`, 548 episodes, 104 scenes, is `test_full`) | geometry-only copies (no camera streams), built 2026-09-18 |
| `pickup` | 42,253 episodes on 800 assets: 196 shards (50,092 stored episodes) read with `manifests/pickup_exclusions_train_20260919.json`; every convex, pick and upright shard and 49 of the 63 place shards, so placing is 24.8% (place 10,458, convex 10,210, pick 10,715, upright 10,870) | 200 episodes of held-out assets, 50 per collection, 141 of the 142 held-out assets | copies (train), source rows rewritten bit for bit (test), built 2026-09-19 |

Membership is `<split>/INDEX.json`: `pwm-indexed-episodes-v1` where an indexed loader exists, a shard-level index for pushing. Every retained training-population episode is `train`; the candidate's development episodes are merged back. Test folders hold only data that was never in train. Hard-linked files are the collector's own files under a second name: never modify or re-permission them from the release side, and the indexed loader checks size and mtime on every read. Every source hash was verified before linking. Receipts, the build scripts, an independent audit and loader smoke reads are in `build/` inside the release and in `/data/user_data/robertre/scratchpad/pwm_data_v1_build/RESULTS.md`. `stats/` in the release holds the scales artifact below byte for byte. The collectors' `accepted_for_training: false` flags are left as they are by design; what stands in for them is recorded per dataset in the sections below. The decisions behind the release (seven datasets in scope at first and pickup added on 2026-09-19, no development sets, motion-weighted window starts for the MPM and folding domains, equal joint draws first) were worked out in a to-do list that was never part of this repository. It was closed on 2026-09-19, when every training job had its confirmed setup table, and its full text is kept at `/data/user_data/robertre/scratchpad/backups_0919/DATASET_TODO.20260919.md`.

Reading it: every dataset is `data.roots.<domain> = {train: <release>/<domain>/train, val: <release>/<domain>/test}`. Pushing reads both folders whole on clean; the indexed domains load through `IndexedEpisodeDataset(<split folder>, domain=...)` and `scripts/unified/collection_recipe.py --indices <release root>` picks the `test` folder as validation when a dataset has no `val` folder; `--folder DOMAIN=NAME` reads a domain from a variant folder (`folding=folding_8mm`). rl_push's two folders are read whole with the exclusion lists the release ships in `rl_push/manifests/`; the builder emits that whenever the layout holds `rl_push/train`, and `--rl-push-release` is only for the single-folder 2026-09-14 release. `--folder dex=dex_v2_filtered2` reads dex from its stricter filtered folder, the one to train on (`dex=dex_v2_filtered` and `dex=dex_v2` read the first filter and the unfiltered regeneration). A dataset without a `test` folder yet (mpm_fresh, mpm_drag until their exports land; dex only through its `dex_v2*` folders) cannot be trained on from the release. The scored clips of each test folder are frozen in `data/manifests/eval_cases/` ([EVAL_PROTOCOLS.md](EVAL_PROTOCOLS.md)); launch recipes and pre-flights for the release live under `/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1/`.

## Frozen populations used for the six-domain pilots

These were the pilots' splits; pwm-data-v1 merges train and development for training. They are the **experiment's retained snapshot counts**, not a live count of everything under a growing collection directory. A clip/episode is one trajectory; overlapping windows and multiple sampler draws are not additional independent episodes.

| Domain | Train episodes | Development episodes | Retained official test | Saved states per episode | Effective interval |
|---|---:|---:|---:|---|---|
| `mpm_fresh` | 42,623 | 4,665 | — | 49 | **0.0417 s** |
| `mpm_drag` | 1,851 | 205 | — | 49 / 81 / 113 | **0.0417 s** |
| `folding` | 4,040 | 470 | 236 | 902 | 1/30 s |
| `yikai` | 9,989 | 1,026 | 548 | 240–1,287 across retained splits | 0.05 s |
| `dex` | 7,996 | 869 | — | 20–400 after stride 2 | 0.05 s; native 0.025 s |
| `rl_push` | 32,974 | 3,652 | — | Variable; source median about 36 | 0.05 s |

The RL push counts are after episode exclusions but **before** the C4/F16 minimum-length requirement: 32,522 training episodes can supply that window starting at frame 3. Its retained populations span 1,084 training and 117 validation objects. Do not interpret every retained short episode as a usable training window.

For the five indexed domains, development membership is a SHA-256 split of whole groups, seed 916, with a 10% threshold:

- **Fresh MPM and dragging:** whole episodes, from the same collected training-asset population. These are new-motion holdouts, not unseen-shape tests. Separate asset-bank validation/test cohorts do not establish that trajectories for those cohorts were collected.
- **Folding:** whole garments. Train/development/test contain **842 / 97 / 49 garments**. The source's 4,510 training clips were divided into 4,040 train and 470 development clips; the 236 source test clips remain separate.
- **Yikai:** whole scenes. Train/development/test contain **1,941 / 193 / 104 scenes**. A development scene holdout does not imply disjoint object assets. The source's `wm_v2_eval` population remains the official test, with the quality exclusions described below.
- **Dex:** whole first episodes within the existing five teacher/object populations. Neither teachers nor object models are held out by this development split.
- **RL push:** its existing object manifest defines the split. Almost every held-out object has a near-twin training shape. This measures new pushes on familiar geometry, not clean novel-shape generalization.

Official-test episodes were excluded from normalization fitting, overfit fixtures, and the overnight learned-prediction evaluation. Earlier schema/reconstruction audits did inspect retained test episodes; the test files have not been completely untouched.

## The six collection datasets

### Fresh passive MPM — `mpm_fresh`

**Purpose:** predict unactuated multi-object deformation and contact across elastic, plasticine and sand materials. Each scene has 1–4 objects with 1,024 persistent particles per object, so a cloud has 1,024–4,096 points. The 49 states cover 48 transitions, or 2.0016 seconds.

The completed `fresh-10432785-v4` campaign contains 47,288 retained episodes across four disjoint publications. Relative to:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-operator-campaigns-v1/fresh-10432785-v4
```

| Published root | Retained episodes before our development split |
|---|---:|
| `data/collection/datasets/train` | 22,087 |
| `recovery-001/salvaged/datasets/train` | 160 |
| `recovery-001/collection/datasets/train` | 16,370 |
| `recovery-002/collection/datasets/train` | 8,671 |

The training files are WebDataset TARs. One sample key contains a small **pre-batch of episodes**; use the provenance `episode_id`, or full shard path + sample key + row, to identify an episode. Shard basenames and sample keys repeat across publications. Raw HDF5 and the shape bank are not training shards.

Positions are metres. The adapter applies `(-0.75, -0.75, -0.10)` m once. The saved interval is **0.0417**, not exactly `1/24`. Actions are zero. Material columns store E, Poisson ratio, density, material ID and yield stress; the current model consumes the first four and does not condition on yield stress. Collection values are documented in the [MPM format guide](../collection/mpm/docs/DATASET_FORMAT.md).

**Limits:** the development set shares the training asset population; release retention is a quality screen, not a guarantee of physical fidelity. This is a separate dataset and calibration from legacy `mpm`.

### MPM dragging — `mpm_drag`

**Purpose:** learn the deformation caused by externally driving a material-point patch, including contact, holds and releases. The gentle-v4 collection uses 1–4 objects, 1,024 points/object, elastic or plasticine material, and one to three events. Sand is excluded. Settling occurs before the saved 49/81/113-state trajectory.

The frozen index combines reviewed publications from these roots, relative to `/data/group_data/katefgroup-ssd/world_model/particle_world_model`:

```text
datasets/mpm-drag-gentle-v4-recovered-20260915/train
staging/mpm-drag-gentle-v4-restart-20260915/data/train/train
staging/mpm-drag-blackwellb-20260916/data/train/train
```

Only published blocks were included. Do not glob all of `staging/`, include rejected `export-review` directories, or combine the original failed publication with its recovered copy. Later publications are not silently added to this snapshot.

Its action contract is deliberately different from rigid manipulation:

```text
action[k] = prescribed_velocity[k] × recorded_transition_duration[k]
target[k] = observed_position[k+1] - observed_position[k]
```

The command is a pre-P2G velocity override, not a force or a guarantee of achieved displacement. `action_mask` identifies the persistent selected patch; `action_active` and `context_action_active` distinguish a zero-velocity **hold** from **release**. All deformable particles, including the driven patch, must still be predicted and scored. Do not replace commands with future observed motion or clamp predictions to those commands.

The canonical shift and physical material inputs match fresh MPM. The collection screens settling, walls, collapse and persistent fragmentation, but shared-grid contact artifacts and the limits of point-based screening remain. See the [drag format guide](../collection/mpm/docs/DRAG_DATASET_FORMAT.md).

**Naming trap:** clean already has `UnifiedMPMDragDataset` through an explicit `data.roots.mpm.task=drag` route that emits domain `mpm`. The candidate's indexed **`mpm_drag`** name has its own frozen membership and fitted scale. Those are different configurations, not aliases. Historical `robert_mpm_drag_setA` / `robert_mpm_v2_drag` exports are not automatically compatible with gentle-v4.

### SIM1 garment folding — `folding`

**Purpose:** predict cloth motion under two grippers on a table, across a new 1,000-garment collection. Five action slots per garment cover left-inward, right-diagonal and two-hand-half recipes with variations. Of 5,000 planned slots, 4,746 clips on 988 garments were retained.

The authoritative source membership is:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-noon-20260916-v4/production/train-INDEX.json
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-noon-20260916-v4/production/test-INDEX.json
```

Follow each record's absolute `file` path. Some retained files live in earlier collection directories; concatenating generation indices would duplicate inherited clips. Episode identity is `(garment_id, world/action slot)`.

Each NPZ contains 902 states at 30 Hz, native cloth vertices, and 1,000 points per gripper. Each gripper is split into hand/left-finger/right-finger counts of 200/400/400, with prismatic joints. Total cloud sizes in the frozen indices range from **4,710 to 34,739**; the candidate retains the native points instead of importing legacy cloth's point cap.

The export is already in the canonical frame and already removes the first physical second. **Do not translate again, stride by two, or remove another settling second.** The loader predicts cloth motion and conditions on recorded gripper motion. Cloth material fields use documented fill values where parameters were not exported.

**Limits:** whole-garment holdout is meaningful, but retained clips are scripted collection motions, not certified successful folds. The source retains `accepted_for_training: false`; the subsequent candidate integration and overfits establish bounded technical usability, not a change to that publication label. This collection is distinct from legacy `cloth` / `cloth_curated_med`. See the [folding format guide](SIM1_FOLDING_DATASET.md).

**Thinned variant `folding_8mm` (2026-09-18).** Folding is the slowest domain and its largest scenes (34,739 points) set the memory ceiling of a joint run, so the release carries a second folding folder with one retained cloth-vertex subset per garment: the farthest-point prefix (pytorch3d `sample_farthest_points` from vertex 0 on frame 0 of the garment's first clip) whose radius stays at or above 8 mm, capped at 10,000 vertices, the same vertices for every frame and clip; the 2,000 gripper points are unchanged. Native spacing (median nearest-vertex distance) is 5.2 / 6.7 / 8.1 mm at p10 / p50 / p90; the rule gives 8.8 mm, and 8.8 to 11 mm on the tenth of garments that hit the cap (97 train, 5 test). Train keeps 54% of its cloth vertices, test 56%; scenes run 3,627 to 12,000 points, mean 8,403, and the summed attention cost is about a third of the native folder's. The files are hard links of `folding/`; each garment gains `cloth-subset-8mm.npz`, whose identity the index pins (`subset_*` fields) and the loader checks on every read; `folding_window` applies the subset to positions, ids and masks, and the motion profile used for window weighting. Read it with `data.roots.folding` pointing at `folding_8mm/{train,test}`, or `collection_recipe.py --folder folding=folding_8mm`. Train and test share one density, so folding numbers from this folder are scored on the same subsets; full-resolution scoring is a separate protocol. Census, per-garment farthest-point orders and figures: `scratchpad/folding_downsample_0918/` (page https://claude.ai/artifact/FFkigvqGhfsnEujZLRvf8R).

### RL-policy rigid pushing — `rl_push`

**Purpose:** predict a rigid object's response to achieved Panda gripper motion. Only successful pushes are stored. The source is a low-dimensional pose/joint release; point clouds are reconstructed during loading.

The pilots read the frozen release:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/rl-push-lowdim-20260914
```

Its live upstream source is `/data/group_data/katefgroup/VLA/3DPCWM/rl_push_data_09_14`. The frozen copy and its `RELEASE.json` identity are the training reference; live top-ups do not change the run. Startup verifies the release's file set, sizes and hashes, and checkpoints retain its identity.

A window has 671 Panda gripper points plus 512 object points: **1,183 total**, four rigid bodies, and articulated fingers. Object points are selected from the stored 1,000-point template; their identity remains consistent within the window. Coordinates receive `(-0.45, 0, 0)` m. Windows begin at source frame 3 or later to avoid initial settling.

The pinned exclusions remove 110 audited episodes for excessive sink/off-table motion, unexplained kicks, contact-launch outliers or gripper penetration. Exact rules and thresholds are in [RL_PUSH.md](RL_PUSH.md) and [the manifest generator](../scripts/dev/rl_push_manifests.py). These are screening heuristics, not proofs of impossible dynamics. Per-object mass and friction are not recorded.

**Limits:** 116 of 117 held-out objects have a near-twin training template within point-sampling noise. The split cannot establish novel-shape, unsuccessful-push, new-policy or real-world performance. Short recordings also limit the amount of autoregressive feedback tested.

### Yikai / SkillWeaver pick-and-place — `yikai`

**Purpose:** predict full-scene rigid dynamics during pick-and-place, including the manipulated object, distractors and the table. The source contains geometry, poses, robot state, task metadata and camera streams; the current training adapter reads geometry/pose fields, not RGB, depth or language.

```text
/data/group_data/katefgroup/datasets/sim_pickplace/wm_v2/{gen,fail}/manifest.jsonl
/data/group_data/katefgroup/datasets/sim_pickplace/wm_v2_eval/{gen,fail}/manifest.jsonl
```

Use manifest rows with a present `npz`, not scene-directory globs: some attempts only create stubs. The source has 12,121 training-population trajectories and 599 evaluation trajectories. The pinned motion audit excludes 1,106 and 51 respectively, leaving the 11,015 train/development and 548 test episodes in our snapshot. **Ordinary task failures remain eligible**; audit exclusions concern launch/impulse/teleport/sink flags, not failure to complete the task.

Recordings are 20 Hz with 2,207–4,767 points once the table is dropped (671 gripper points plus 512 per object). **Shared Franka gripper (2026-09-19):** the gripper rows are the pickup corpus's link points (`data/embodiments/panda_gripper_pickup.npz`: hand 241, left finger 212, right finger 218, the set rl_push builds its gripper from) under the recorded link poses, not the recording's own 667-point sampling of the same links (237 hand, 159 + 56 per finger with its tip). Measured on the release: equal link frames and boxes (under 1 mm), recorded fingers at (0, ±q, 58.4 mm) in the hand frame within 0.003 mm, two-sided Chamfer distance between the samplings 6 mm on the hand and 2 mm on the fingers, which is sampling noise; on production windows every object row, passive pose target and joint layout is bit-identical before and after, gripper rotations agree to 1e-6 rad and translations to 0.2 mm (the two point sets have different centroids). `YIKAI_SHARED_GRIPPER` in `data/recording_adapters.py`; `load_native_episode(..., shared_gripper=True)` gives the same rows to the native reader (default off there). Folding's two grippers are SIM1's own template and are not part of this. Since `c29fbb3` (2026-09-18) the adapter drops the table, shifts each episode by (-0.45, 0, -`state/table_top_z_env`) so the table top sits at z = 0 (source table tops span -0.05 to 0.18 m; the height is cross-checked against the index record and reported in the loader's `last_windows`), merges fingertips into their fingers, uses three Panda links plus every scene body, and starts windows at frame 10 because distractors settle during the first ten frames (`FIRST_START` in `data/recording_adapters.py`). It conditions on achieved gripper motion while predicting the passive objects, including distractors. After the shift 446 of 11,015 train and 21 of 548 test episodes (4.0% / 3.8%) still have a point beyond 0.8 m in |x| or |y| (largest 1.41 m); decided 2026-09-18: every episode stays in train and test, and the 462 episodes that still offend from frame 10 on are tagged in [`data/manifests/yikai_extent_tags_20260918.json`](../data/manifests/yikai_extent_tags_20260918.json) (442 train, 20 test; 86 of them "far", beyond 0.9 m) with the inclusive source-frame ranges where a point is beyond 0.8 m, so an evaluation can report flagged windows separately; no loader applies the tags. In 217 episodes only a distractor offends (76% from the first frame, 72% never move, at most 1.01 m); in 174 only the gripper does, for a median tenth of the frames (median 0.88 m, the far cases are failed attempts with 1.3–1.5 m of arm travel); the manipulated object is carried out to at most 0.96 m. Positions are divided by the fixed 0.8 m half-range and never clipped for this domain; the UNet's banded rotary code has a unique range of 1.67 m per axis, which point pairs in the far gripper frames can exceed. The per-episode shift departs from the one-fixed-shift-per-domain rule for this reason: the table height varies per scene and every other dataset has its support surface at z = 0.

Two schema details are essential:

- `state/particle_part_poses_wxyz` really uses **wxyz**. The separate field named `state/object_poses/*/rotation_wxyz` actually uses **xyzw**; the training path avoids relying on that misleading name.
- Raw `obj_ids == 0` combines objects and table. Recover body identity from part membership/names. Table height varies by scene and must not be assumed to be canonical `z=0`; the adapter removes it per episode.

**Limits:** development scenes can share object assets with training; the retained official test is quality-filtered, not the full original test population. Whole-scene averages can be dominated by stationary distractors, so report moving and low-motion subsets. The existing floor auxiliary does not constrain these rigid objects to their tables. See the [source format and audit guide](/data/user_data/robertre/scratchpad/yikai_wm_v2_0915/DATASET_FORMAT.md).

### Dexterous-hand teachers — `dex`

**Purpose:** predict passive manipulated-object motion given the achieved motion of a dexterous hand. Since
2026-09-18 the release has a second dex folder, `dex_v2`: the five tactile-genesis teachers regenerated with new
objects and a test set of held-out objects (user decisions 2026-09-18).

- Train and evaluate on `dex_v2_filtered2`: `dex_v2` minus 2,934 episodes (2,914 train, 20 test), the simulator
  artifacts a full-episode audit flagged, the milder single-finger rides a looser detector setting finds, and every
  repose-XHand1 episode on boxes (user decisions 2026-09-18). Read it with `collection_recipe.py --folder
  dex=dex_v2_filtered2`, or `data.roots.dex = {train: <release>/dex_v2_filtered2/train, val:
  <release>/dex_v2_filtered2/test}`. `manifests/dex_v2_exclusions2.json` there lists every excluded episode and why.
- `pwm-data-v1/dex_v2_filtered` keeps the first filter (minus 625), `pwm-data-v1/dex_v2` the unfiltered regeneration
  and `pwm-data-v1/dex` v1 (train only, the last bullet of Limits), all for provenance.
- The table counts `dex_v2`. In the same row order, `dex_v2_filtered2` has 3,978 / 1,854 (objaverse only) / 3,317 /
  3,941 / 3,909 train and 49 / 34 / 44 / 24 / 23 test episodes (`dex_v2_filtered`: 3,983 / 3,804 / 3,655 / 3,941 /
  3,909 and 49 / 46 / 48 / 24 / 23).

| Teacher | Hand | Task | Train objects | Train episodes | Test objects | Test episodes |
|---|---|---|---|---:|---|---:|
| `in_palm_rotate-xhand1` | XHand1 | Rotate an object in the palm | 14 boxes/cylinders, 100 objaverse meshes | 3,983 | 3 boxes, 20 objaverse meshes | 49 |
| `in_hand_repose-xhand1` | XHand1 | Reorient an object | the same | 3,988 | the same | 49 |
| `in_hand_repose-sharpa` | Sharpa | Reorient an object | the same | 3,986 | the same | 49 |
| `screwdriver-xhand1` | XHand1 | Rotate a screwdriver | models 1-3 | 3,957 | model 0 | 24 |
| `screwdriver-sharpa` | Sharpa | Rotate a screwdriver | models 1-3 | 3,999 | model 0 | 23 |

Source (its README is the dataset card):

```text
/data/group_data/katefgroup-ssd/world_model/robertre_dexhand/tactile_genesis_teachers_v2
```

**Objects.**

- **Boxes:** tactile-genesis `OBJECTS_4CM_16`, 15 boxes of 30-45 mm and 2 cylinders. Variants 7, 8 and 11 are held
  out.
- **Objaverse meshes:** Dexonomy objaverse_5k (CC BY-NC 4.0), in their most stable resting pose, scaled so every box
  edge is 30-45 mm, 20 per scene. Half of each in-hand teacher's train episodes are on them.
- **Screwdrivers:** models 1-3 in train. Model 0 is test only, with a corrected grasp file (the shipped one starts it
  14.8 cm too high, which broke every v1 episode with it).

Genesis builds each mesh's collision shape: one convex hull, or CoACD pieces, up to 426. The objaverse shards were
generated with `max_collision_pairs` 500, because the tasks' own 30 / 48 overflows on many-piece meshes and Eden
resets those envs.

**Episodes.**

- Each shard is one rollout from reset, and only each env's first episode is used, ending at its first termination.
- Stable identity is `<teacher>@<part>/<env>`; test parts are named `heldout_*`, so ids never repeat across splits.
- Native states are 40 Hz, and the adapter takes stride two for 20 Hz. A C4/F16 window needs at least 39 native
  states; the index excludes the 87 train and 3 test episodes that are shorter.
- No episode ended in a solver error, and there is no perturbation. Outcome labels come from the recorded
  terminations.

**Positions.** They are reconstructed from recorded poses and frozen surface samples. A multi-piece object's points
come from the outer surface of its collision pieces; on held episodes they sit a median 1.1 mm from the hand. The
object is a passive free body; hand links form the fixed-root/revolute-child kinematic structure. XHand1 and Sharpa
have different layouts and cannot be mixed inside one batch.

**Actions mean achieved hand motion**, not the raw policy output or verified torques. Randomized mass and friction are
not stored. The mass shift is a point mass at the centre of mass, so thin objaverse meshes, which weigh under a gram
at the default density, keep a small rotational inertia.

**Limits.**

- The test set measures unseen objects for the same five teachers, not unseen hands or tasks.
- Fingers sink into the objects. The teachers' simulation has soft contacts (10 ms constraint time constant) and
  squeezes at the torque limits, so the hand and object collision shapes overlap:
  - boxes: median 1.3 mm per frame, p99 14.6 mm, 23% of frames past 5 mm;
  - objaverse meshes: p99 9-12 mm, 7-12% of frames past 5 mm;
  - v1 boxes: the same as v2's.

  Genesis's own contact depths confirm it (the offline measure tracks them with correlation 0.987). Details in the
  dataset card, item 6; page https://claude.ai/artifact/KMyUSE6sV948UArg3mU7US.
- Some objaverse meshes are skeletal (frames, rods, plates), and a third of the train meshes weigh under 2 g. Hollow
  ones sometimes ride on the thumb tip like a thimble in repose episodes:
  - 51 of 6,091 objaverse episodes for at least 1 s, 17 for at least 2 s, the longest 8 s;
  - none past 1.5 s for boxes and screwdrivers.

  The list is in the dataset's `_checks/`.
- `dex_v2_filtered` drops the episodes where these artifacts are worst (dataset card, section Filtered release):
  - start pop (293): a Sharpa object starts more than 5 mm inside the hand's collision hulls, which are fatter than
    its visual meshes, and the simulator shoots it out in the first 0.1-0.15 s;
  - buried finger (177): a released hand point stays more than 10 mm inside the object for at least 1 s;
  - ejection (106): a screwdriver's speed jumps by more than 1 m/s within one 25 ms step, mostly as it drops;
  - single-finger carry (54): the thimble cases above.

  The soft contact itself stays: in the kept repose-XHand1 box episodes a finger is more than 5 mm inside the box in
  about a third of the frames. The audit found no non-finite values, duplicates, teleports, palm motion or support
  by geometry the points do not show.
- `dex_v2_filtered2` also drops the milder single-finger rides (463: a ride of 0.5 s, or of 1 s with looser
  tolerances) and every repose-XHand1 episode on boxes, which held 81% of the frames with a hand point past 10 mm.
  In what it keeps a hand point is past 5 mm in 3.4% of train frames and past 10 mm in 0.26%.
- The v4 scales artifact's dex entry was fit on v1 (boxes and screwdrivers); v5 carries the `dex_v2` fit, which also
  holds for `dex_v2_filtered` (its own reference fit is within 1.3%); v6 carries the `dex_v2_filtered2` fit, whose
  particle z scale is 4% below v5's.
- v1, the 2026-09-17 release: 8,865 train episodes from `tactile_genesis_teachers_v1` (17 boxes/cylinders, screwdriver
  models 1-3, 20% perturbed envs), no test.

The [Dex format guide](../collection/dex/docs/DATASET_FORMAT.md) documents the raw schema.

## Earlier/default datasets and Water-3D

These remain separate corpora. They were **not** part of the six-domain overnight pilots. Their default roots and exclusions are defined in [data/unified_dataset.py](../data/unified_dataset.py); historical episode counts have not been recounted for this handoff.

Paths below use `G = /data/group_data/katefgroup/VLA/3DPCWM`.

| Domain | Registered data location | Split and important distinction |
|---|---|---|
| `mpm` | `G/robert_mpm_v2/webdataset_full49`; validation `G/robert_mpm_v2_test_seen` | Earlier passive corpus; current registry uses 1/24 s. Training collapse exclusions are separate from evaluation exclusions. Seen/unseen/clean/count-OOD populations are different protocols. Fresh MPM has its own exact clock, release and scale. |
| `pickup` | `G/robert_pickup_data_{place,convex,pick,upright}_success` | Successful rigid manipulation; recorded gripper actions at 20 Hz. Validation is the last 10% of sorted shards per root. The 142-asset holdout list is excluded from both ordinary train and validation, so that validation is not the asset-holdout benchmark. **pwm-data-v1/pickup (2026-09-19)** is the release form of this corpus: no development split, the held-out assets are the test set (200 episodes), placing is cut to one collection in four, and the folder states its own canonical translation (-0.45, 0, -0.11), written into recipes as `data.roots.pickup.shift_m`: the collection's table top is at 0.110 m, so the registered -0.10 m leaves it 10 mm above z = 0 (gripper's lowest point over 2,400 windows 9.9 mm, objects rest at 10 to 11 mm). The registered domain keeps -0.10. IsaacLab replays of four RL policies, 946 tabletop assets, 1,183 points = the shared 671-point Franka gripper + 512 object points. |
| `cloth` | `G/unified_v1/cloth/{train,val}` | Materialized legacy `cloth_curated_med`; prior garment split and cloth blocklists apply. Its materializer already performs the original 60-to-30-Hz stride and coordinate shift. Do not apply either again. This is not the new `folding` population. |
| `pushing` | `G/pushing_data_actions` | Earlier rod/block trajectories; validation is the last 10% of sorted shards. Physical frame interval remains unverified (`DOMAIN_DT=None`). Only rod motion is supplied: the source's block `absolute_targets` contain future ground truth and must not become actions. |

The old [unified dataset specification](unified_dataset_spec.md) is historical: its C2 windows, blanket action statement and some schema assumptions predate current C4 training and commanded dragging. Prefer the current code and domain-specific format documents for those contracts.

### Water-3D — `water`

The materialized release has **1,000 train / 100 valid / 100 test trajectories**, verified from its three manifests:

```text
/data/group_data/katefgroup/datasets/deepmind_lts/Water-3D/materialized_v1/{train,valid,test}
```

The source stores 801 states at 200 Hz. The adapter takes every eighth state for **25 Hz** and selects 4,096 particles per window. It rotates the source from y-up to z-up, then applies a fixed translation. There are no actuators, rigid bodies or explicit wall particles.

Water is opt-in. It needs material ID 6 (`model.num_mat_types=7`), a suitable UNet material-point budget, and explicit compatible normalization such as `stats/normalization_water3d.json`. The six-domain collections artifact does not contain a per-domain water entry.

**Limits:** the box is fixed and walls are implicit; point subsampling omits neighbors. The documented experiments are fixed-batch overfits, including historical runs that predate the dedicated material ID, not a full-corpus benchmark. See [WATER3D.md](WATER3D.md) for their exact qualifications.

## Shared training contract

All adapters produce [TrajectoryBatch](../data/batch.py). The current collection recipes use **C=4 context states and F=16 future states**, domain-pure batches, and a common production UNet or Transformer. The domain is selected explicitly; it does not select a separate specialist architecture.

- `positions`: `[B, C+F, N, 3]`, canonical **metres**. Do not pre-normalize or independently recenter every frame.
- `future_deltas`: adjacent-frame displacements. The first target is `position[C] - position[C-1]`; later targets difference consecutive recorded future states.
- `actions`: `[B, F, N, 3]`, metres per saved transition. Rigid actuators use achieved motion. MPM dragging uses commanded motion plus temporal activity gates. These cases must not be conflated.
- Rigid sidecars carry link templates, poses and articulation. Free-body pose deltas use translation and rotation-vector channels; prismatic/revolute children carry their joint-coordinate changes.
- Object IDs are scene-local grouping labels, not globally comparable asset classes or material identities. Persistent point identity and joint topology must survive the window.
- MPM supplies measured physical channels. Missing cloth/rigid/gripper properties use explicit fill values; those fills are not measurements of unrecorded simulator randomization.

The five indexed adapters sample episodes uniformly, then choose windows. By default a batch draws windows of one episode so point layout and articulated topology agree; since 2026-09-17 the collection recipe draws mpm_fresh and mpm_drag batches from B different scenes of equal object count (`data.batch_across_episodes`) and weights training window starts by motion for mpm_fresh, mpm_drag and folding (`data.window_motion`: p(start) ∝ (mean future step of the predicted particles + floor)^alpha, alpha 1, floors 2 / 0.25 / 0.02 mm; measured on the release, the share of draws on quiet windows drops from 47 / 44 / 77% to 22 / 22 / 42%). Yikai, dex and rl_push stay uniform; validation windows never depend on either setting. Equal domain weights mean equal probabilities of drawing domain batches, not equal particles, frames, gradient contribution or wall time. The existing RL push adapter retains its own object/file grouping.

A common F16 does not span a common physical duration: fresh MPM uses 0.6672 s from anchor to the sixteenth future state, folding about 0.5333 s, and the 20-Hz rigid datasets 0.8 s. There is currently no explicit physical-dt model input. Changing stride changes both the dynamics and the displacement statistics.

## Normalization used by the collection pilots

Only the current artifact is kept in `stats/`; the superseded versions named below are in git history (last present at `58ade7c`)
and, from v2 on, in the release's `stats/` folder (`stats/README.md`).

**Current artifact for runs on pwm-data-v1: `stats/normalization_collections_20260919_v7.json`** (artifact version
`collections-xy-p99-2026-09-19-v7`, sha256 `0b7c0e6c10bf...`, the same file as the release's `stats/` copy): v6 with pushing on ONE scale for
all axes (user decision 2026-09-19; the paragraph **v7** below). `collections-xy-p99-2026-09-18-v6` (sha256 `a5bea36d3740...`) is v5 with the dex entry refit on
`dex_v2_filtered2/train`, the dex folder that trains (user decisions 2026-09-18; the numbers are below the table).
`collections-xy-p99-2026-09-18-v5` (sha256 `c3476b4ffe0c...`) is v4 with the dex entry refit on `dex_v2/train`.
`collections-xy-p99-2026-09-18-v4` (sha256 `ab4c77b996e2...`) is v3 with the folding entry refit on the
thinned folder `folding_8mm` that trains (512 episodes, the same episodes and windows as the native fit; particle xy 1.80 / z 1.72 mm
against 1.60 / 1.35 on the native vertices, pose unchanged, because thinning keeps a higher share of moving vertices). v3
(`collections-xy-p99-2026-09-18-v3`, sha256 `ab5fd089989d...`) was assembled 2026-09-18 by `scripts/unified/verify_collections.assemble_artifact` on the v2
base and differs from v4 only in that entry. Its entries are the reference fits on the release's train
folders (`scratchpad/pwm_data_v1_build/scales_confirm/{reference,yikai_reference}`): 1,024 episodes for mpm_fresh and dex,
all 2,333 for mpm_drag, 512 for folding, yikai's 1,024-episode fit without the table, and rl_push from the 2026-09-18 fit record
`normalization_rl_push_20260918.json` (in git history; its entries are v6's rl_push entries); the four legacy entries and pushing
are unchanged from v2.

| Domain | Particle XY / Z, mm per transition | Pose XY / Z, mm | Rotation XY / Z, rad | Against v2 |
|---|---:|---:|---:|---|
| `mpm_fresh` | 22.37 / 76.66 | — | — | xy 0.97x, z 1.11x |
| `mpm_drag` | 7.31 / 6.15 | — | — | xy 0.95x, z 0.83x |
| `folding` | v4: 1.80 / 1.72 on `folding_8mm` (v3: 1.60 / 1.35 native) | 2.7 / 2.8 | 0.0135 / 0.0003 | thinned vs native 1.12x / 1.28x; native vs v2 within 3%, pose xy 1.20x |
| `dex` | 13.26 / 12.11 | 13.0 / 13.1 | 0.318 / 0.186 | within 2% |
| `yikai` | 12.31 / 12.78 | 13.3 / 13.9 | 0.0707 / 0.0993 | xy 0.97x, z 1.07x, rotation 1.11x (table dropped) |
| `rl_push` | 46.01 / 20.35 | 44.77 / 20.43 | 0.1686 / 0.2198 | within 1.3% |
| `pushing` | 3.88 / 0.16 | 3.88 / 0.17 | 0.0026 / 0.0543 | unchanged through v6; v7: 3.88 / 3.88, 3.88 / 3.88, 0.0543 / 0.0543 |

The 2026-09-18 overfit fixtures were drawn under v2 and stay there (a fixture pins its normalization); every training
launch on the release uses the current artifact, v7.

**Dex was regenerated after v4 (2026-09-18), so v5 refits it.** v4's dex entry (the table's row) is the reference fit on
the v1 `dex` folder. v5's entry is the same recipe on `dex_v2/train` (1,024 episodes; receipts
`scratchpad/pwm_data_v1_build/scales_confirm/dex_v2_reference`, assembled by `36_assemble_v5_dex.py`):

- particle xy 12.78 / z 11.05 mm (0.96x / 0.91x of v4);
- pose 12.80 / 11.82 mm (0.98x / 0.90x);
- rotation 0.295 / 0.177 rad (0.93x / 0.95x).

Every other entry of v5 equals v4's.

**v5 also serves `dex_v2_filtered`.** The same recipe on `dex_v2_filtered/train` (receipts
`pwm-data-v1/build/dex_v2_filtered_0918/scales/`) gives particle 12.63 / 10.90 mm, pose 12.85 / 11.72 mm and rotation
0.295 / 0.175 rad: 0.99x / 0.99x, 1.00x / 0.99x and 1.00x / 0.99x of v5's entry, inside the release's 3% tolerance,
so no v6 was cut for it.

**v6 refits dex on `dex_v2_filtered2`.** Dropping every repose-XHand1 episode on boxes and the milder single-finger
rides moves the dex particle z scale out of tolerance. The same recipe on `dex_v2_filtered2/train` (1,024 episodes,
seed 916; receipts `pwm-data-v1/build/dex_v2_filtered2_0918/scales/`, assembled by `38_assemble_v6_dex_filtered2.py`):

- particle xy 12.81 / z 10.60 mm (1.003x / 0.960x of v5);
- pose 13.08 / 11.54 mm (1.022x / 0.976x);
- rotation 0.302 / 0.178 rad (1.022x / 1.007x).

Every other entry of v6 equals v5's.

**v7 puts pushing on one scale for all axes (2026-09-19, user decision; no new fit).** Pushing is a planar task. Its v6
vertical translation and tilt scales were the p99 of motion that is almost always zero (0.165 mm and 0.17 mm per frame,
0.0026 rad per frame), about 20 times finer than the in-plane scales. With clipping off, ordinary block wobble became a
target of 3 to 30 scales and a rare topple a target above 100, so the loss spiked and the trainer's spike gate rejected the
step: about 225 of 5,001 steps in a two-GPU x0 run (job 10498541). v7 sets particle `[xy, xy]` and pose
`[translation xy, translation xy, rotation z, rotation z]` (3.88 mm, 0.0543 rad per frame). The train folder is read
without 533 episodes in which the simulation failed (tilt above 5 degrees, centroid more than 5 mm out of plane, or an
in-plane jump above 10 mm in one frame; `pushing/manifests/pushing_exclusions_train_20260919.json` in the release, written
into a recipe by `collection_recipe.py`). With both, the largest normalized pushing target falls from 111 to 5.7 and the
same run rejects 2 of 5,001 steps (job 10499806; none with the training noise-level mixture, job 10500148). The two are fast in-plane pushes
whose loss passed ten times a running loss that had fallen to 0.002, not outliers. The evidence note inside the v7 file says 0: it was
written from the learning-rate lag, which resolves about 3 steps; the trainer's saved rejected batches give the exact count. Every other entry of v7 equals v6's. Receipts: the release's
`build/scales_v7_0919/` and `build/pushing_exclusions_0919/`.

The tested artifact is **`normalization_collections_20260916_v2.json`** (superseded, in git history), artifact version `collections-xy-p99-2026-09-16-v2`, using `xy_p99_per_domain`. The [frozen artifact](/data/user_data/robertre/scratchpad/overnight_collections_0917/src/stats/normalization_collections_20260916_v2.json) is authoritative for these runs; the earlier filename without `_v2` reproduces older engineering checks.

| Domain | Particle XY half-range, mm/transition | Particle Z half-range, mm/transition |
|---|---:|---:|
| `mpm_fresh` | 23.0168 | 68.9195 |
| `mpm_drag` | 7.6867 | 7.4413 |
| `folding` | 1.5517 | 1.3811 |
| `yikai` | 12.7502 (fit with the table; without it, 2026-09-18: four 128-episode seeds 11.4–13.0, 1,024-episode reference 12.31) | 11.9477 (without the table: 12.1–13.2, reference 12.78) |
| `dex` | 13.0626 | 12.0862 |
| `rl_push` | 46.3916 | 20.0816 |

The five new fits use 128 deterministically selected **training** episodes and four starts per episode, with a 65,536-row priority reservoir. XY axes share a bound. Pose translation/rotation scales are separate entries; free-link and joint-coordinate estimates are combined componentwise. RL push retains its separately fitted frozen-release entry. Existing legacy entries and the historical global profile are unchanged.

Confirmed on the merged pwm-data-v1 train sets on September 17 with the same recipe (four seeds and a larger reference fit; receipts in `/data/user_data/robertre/scratchpad/pwm_data_v1_build/RESULTS.md`): every xy scale within 5% of the artifact and z within 3% on folding, dex and pushing; the two MPM z scales sit inside the 128-episode recipe's own noise but off the population value (fresh 68.9 vs 76.7 mm on 1,024 episodes, drag 7.44 vs 6.15 mm on all 2,333), so they are to be refit on many more episodes when the next artifact version is cut. The artifact was not changed. Yikai was refit on 2026-09-18 with the table dropped (four 128-episode seeds: particle xy 11.4–13.0 / z 12.1–13.2 mm against the artifact's table-included 12.75 / 11.95; pose 12.2–13.7 / 13.0–14.0 mm and 0.063–0.074 / 0.093–0.104 rad against 13.38 / 13.19 mm and 0.0638 / 0.0896 rad); the 1,024-episode reference fit, xy 12.31 / z 12.78 mm (pose 13.32 / 13.95 mm, 0.0707 / 0.0993 rad), defines the entry of the next artifact version.

These are sample p99 scales, **not hard physical bounds**. Valid targets can exceed normalized magnitude one. The six-domain recipes set **both `train.clip_deltas=false` and `train.clip_pose_deltas=false`**. Spatial encoding uses the shared 0.8 m scale; the candidate avoids clamping new folding/Yikai/Dex pose centres that legitimately exceed the old range.

Normalizers are attached to physical-unit batches and applied by the consuming model/objective. Checkpoints save the resolved numerical snapshot and its checksum; evaluation/resume must use that identity. Changing the dataset release, exclusions, stride or normalization is a new experiment. See the [normalization guide](../stats/README.md).

## Reproducible entry points and preservation

For the completed pilots, use this experiment root:

```text
/data/user_data/robertre/scratchpad/overnight_collections_0917
```

| Artifact | Purpose |
|---|---|
| `src/` | Exact tested training/loader implementation (snapshot v8); since `706e694` the merged `robert/clean` contains the same source apart from `data/mpm_drag.py` |
| `manifests/<domain>/{train,val,test}/INDEX.json` | Explicit episode membership for the five indexed domains; test exists only where supplied |
| `manifests/<domain>/SUMMARY.json` | Population counts, grouping, source receipts and exclusion provenance |
| `configs/unet_20k.yaml`, `configs/transformer_20k.yaml` | Exact six-domain pilot recipes, including roots and normalization snapshots |
| `evaluation/LEDGER.json` | Fixed development cases and their evaluation protocols |
| `GALLERIES.md` | Embedded rollout/overfit GIFs, comparisons and loss curves |

The source archive is `/data/user_data/robertre/scratchpad/dataset_readiness_0916/receipts/source_snapshot_v8.tar.gz`. The candidate's `scripts/unified/collection_recipe.py` builds explicit recipes, and `launch/train_collections.sh --check` checks a configured launch without allocating GPUs. Those are candidate tools, not a claim that new domain names work in the current main checkout.

The indices freeze membership without copying the full corpora. Referenced source files and auxiliary geometry must remain available. In particular, Dex indices reference geometry under `scratchpad/dataset_readiness_0916/manifests/dex_geometry`, and folding references inherited collection directories. **Do not delete those as temporary duplicates.** Runtime size/time checks and recorded hashes detect specified changes; they do not make the filesystem immutable or prove every byte of every corpus was rehashed.

For a new release, snapshot reviewed publications, deduplicate stable episode identities, split whole episodes/garments/scenes as appropriate, fit training-only statistics into a new artifact, and repeat reconstruction/overfit/rollout checks. Do not silently point an existing run at a growing staging root.

## What has actually been verified

The candidate passed the bounded loader, normalization, geometry, capacity and training checks documented in the [readiness report](/data/user_data/robertre/scratchpad/dataset_readiness_0916/DATASET_READINESS_FINAL.md). These include one terminal-window reconstruction check for all **74,518 retained episodes across the five indexed domains**, and full-cloud production-model checks. They do not cover every possible window.

Both full-size architectures completed 20K iterations of a 200K schedule across all six domains. All recorded losses/gradient norms were finite; the existing spike guard rejected 11 UNet and 9 Transformer updates. All context augmentation was off and both target clips were off. These runs do **not** test light-noise benefits.

Each model was evaluated on 247 development cases with three sampler draws each, including 128-frame and selected full-recording rollouts. Earlier six-domain overfits were supplemented by 12 additional overnight fits. For rigid/mixed scenes, primary position error excludes supplied actuators/fixtures; all commanded deformable particles remain scored. Moving and low-motion diagnostics expose dilution by stationary geometry.

**Model quality remains uneven.** RL push shows a useful pilot signal; Transformer dragging and UNet folding are promising on the measured cases. Large-displacement MPM errors remain substantial, and long Dex/Yikai rollouts can lose to a stationary baseline. Neither successful overfitting nor passing reconstruction checks proves generalization or task success. There is one training seed per architecture.

The expanded candidate suite recorded 1,908 passes and 14 skips, alongside one exactly reproduced pre-existing pickup-clipping failure. The separate checkpoint-window baseline remained 42 known failures and 23 passes. Exact data-iterator order is also not restored on resume. These limits are documented, not claims of new dataset regressions or bug-free software.

Start with the [dataset galleries and loss curves](/data/user_data/robertre/scratchpad/overnight_collections_0917/GALLERIES.md), [results](/data/user_data/robertre/scratchpad/overnight_collections_0917/RESULTS.md), and [review/next steps](/data/user_data/robertre/scratchpad/overnight_collections_0917/REVIEW_AND_NEXT_STEPS.md). All counts and readiness statements above are tied to the identified snapshot; later collections or branch merges require an explicit update to this guide.
