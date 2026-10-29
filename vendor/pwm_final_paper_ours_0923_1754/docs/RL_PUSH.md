# RL push: the opt-in `rl_push` domain

A Franka Panda gripper, driven by an RL policy, pushes one rigid object toward a goal on a table
(Isaac Sim 4.5, 20 Hz). The release is low-dim: poses and joint positions, no point clouds. The loader
(`data/rl_push.py`) builds points from the poses at load time, in exactly the pick-and-place layout.

- Release: `/data/group_data/katefgroup/VLA/3DPCWM/rl_push_data_09_14` (format `rl_push_lowdim_v1`; its
  `README.md` documents every array). One `.npz` per object under `objects/<root>/<category>/<asset>.npz`.
- Objects: 1,202 SAM 3D-reconstructed assets in 54 categories (Kitchen_Table 634, libero 473,
  Bathroom_Counter 95). Only successful pushes are saved. One file (`libero/bbq_sauce_bottle/asset_0129`, the only
  bbq_sauce_bottle) has no episodes left, so 1,201 objects in 53 categories carry pushes.
- The release is still growing. On 2026-09-14 the owner moved every episode whose finger joint opened by
  more than 1 mm out of `objects/` (19,245 of 54,446) and started top-up runs with the fingers locked; the
  snapshot read at 23:10 held 36,736 episodes (1,517,691 frames, 21.1 h). Re-run the audit, the manifests and
  the stats fit when the owner says the release is final.
- **Re-frozen 2026-09-18 as `pwm-data-v1/rl_push`** (see the section at the end): the owner's source stopped
  changing on 2026-09-15 11:43 at 57,153 episodes; the audit, the outlier list and the scales were redone on all of
  them, and the release has a train folder (1,085 objects) and a test folder (the 117 validation objects).
- The 2026-09-14 training run read a frozen copy of the earlier snapshot, so later top-ups could not change it:
  `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/rl-push-lowdim-20260914`
  (`README.md`, `reconstruct.py`, `manifest.csv`, `objects/`, `robot/`, `runs/`). Its `RELEASE.json` lists every
  file's sha256, equal to the source's at copy time; no object file changed after 2026-09-14 22:51:42.
  The launcher pins its manifest digest and verifies the exact file set, sizes and hashes.
  Training startup repeats content verification and checkpoints bind the manifest digest.
  Files are read-only but directories are writable: these checks establish identity at
  startup, not filesystem immutability afterward. Older checkpoints without this digest
  resume with a warning after current contents are verified; their historical identity
  cannot be retroactively established. A different release needs an explicit
  `RL_PUSH_RELEASE_SHA256` when using the launcher.

## Points (the pick-and-place layout)

| rows | source | count |
|---|---|---|
| 0 to 240 | `panda_hand`, pickup's link points posed by the recorded hand pose | 241 |
| 241 to 452 | `panda_leftfinger` at (0, +q_left, 58.4 mm) in the hand frame, no rotation of its own | 212 |
| 453 to 670 | `panda_rightfinger` at (0, -q_right, 58.4 mm) | 218 |
| 671 to 1,182 | object, from `object_cloud_outer_surface` posed by the object pose | 512 |

The gripper link points are the pickup corpus's own, which yikai shares since 2026-09-19 (`data/embodiments/panda_gripper_pickup.npz`, made and
checked by `scripts/dev/pickup_gripper_points.py`: identical in all four pickup roots, pickup particles equal link
pose @ these points to 1e-6 m, and pickup's fingers follow the same joint convention to 3e-6 m). Object points are
a fresh random 512 of the 1,000 stored per training window, and one subset per object (seeded by its asset key)
for validation and evaluation.

Object id 1 = gripper, 0 = object; `MAT_GRIPPER` / `MAT_RIGID`; physics fills E 1e9, nu 0.3, rho 750 (no masses are
recorded). Actions are the achieved gripper point motion. The rigid sidecar (four bodies: object, hand, left and
right finger), the jointed fingers (`model.articulated_bodies`) and the pose channels come from the same functions
as pickup. Stored hand quaternions are normalized on read (some are not unit length at an episode's first frame).

## Decisions

| decision | value | where |
|---|---|---|
| domain | `rl_push`, opt-in (never selected by default) | `data/unified_dataset.py` `OPTIONAL_DOMAINS` |
| frame | robot base frame, shifted by (-0.45, 0, 0) m like pickup; the table top is already z = 0 | `CANONICAL_TRANSLATION` |
| frame time | 0.05 s | `DOMAIN_DT` |
| window | C4 + F16 (`DOMAIN_FUTURE_LEN` 16), starts at frame 3 or later | `data/rl_push.py` `RL_PUSH_FIRST_START` |
| validation | 117 held-out objects: 10% of every (root, category) group with at least 5 objects, seed 0 | `data/manifests/rl_push_val_assets.json` |
| outliers | 110 episodes of 85 objects (rules below), keyed by collection provenance | `data/manifests/rl_push_exclusions.json` |
| scales | own entry fitted on the training split: particle xy 46.4 mm, z 20.1 mm; pose translation xy 45.2 mm, z 20.2 mm, rotation xy 0.167 rad, z 0.220 rad | `stats/normalization_rl_push.json` |

Frames 0 to 2 are skipped because objects spawn about 5 mm above the table and settle, and a few get a sideways
kick at frame 2 with the gripper out of reach. Both manifests apply to both splits and are hashed into every
checkpoint (`exclusion_files`, `utils.checkpoint.corpus_identity`). Exclusions are keyed by asset and
`source_run`, `source_shard`, `source_episode_index`, `source_env`, which stay valid when the owner merges top-ups
into an object file; episodes added later are only covered after the audit runs again.

The outlier list drops 110 of the 36,736 audited episodes (118 rule matches). The object sinks more than 5 mm into
the table (24), leaves it by more than 20 mm (22), starts moving before the gripper can reach it (13), gains more
than 0.10 m/s of horizontal speed from one frame to the next after the settle frames while no contact is possible
(8), gets such a kick during the settle frames at event frame 3 or later, where windows may start (8; event frame f
compares the transitions f-2 -> f-1 and f-1 -> f), or leaves a contact more than twice as fast as the fastest
gripper point (38); or the gripper goes more than 5 mm into the object (5). The contact-launch
ratio is an outlier heuristic, not a general proof of impossible dynamics: spin, contact geometry,
and unrecorded mass/friction limit that interpretation. Its upstream candidate filter requires
object speed >0.30 m/s, gain >=0.10 m/s, event frame >=4, and visual-point gap <=20 mm.

The rules live in `scripts/dev/rl_push_manifests.py`. Stage 2 already filters unexplained
events at 0.10 m/s and omits individual settle-event gains. `--gain_mps` must equal that
detector cutoff; both higher and lower retuning requests are refused until stage 2 is
regenerated. Other thresholds operate on the retained receipt metrics. New manifests
record and hash the stage-2 threshold summary and detector reference script as well as
the three data receipts. This records those references at manifest creation; it does not
prove how an earlier receipt was generated. The committed historical manifest hashes
only the data receipts, which supports regenerating its exclusion list but does not pin
the upstream detector parameters or code. It is retained unchanged for existing runs.
Object files without episodes (one so far,
`libero/bbq_sauce_bottle/asset_0129`, emptied by the owner's finger filter) are skipped by the loader and by the
validation picker.

Kept but near a threshold, for the record: 335 episodes sink 2 to 5 mm (90 beyond 3 mm; two thirds are small
saucepans and teapots), 228 have the gripper 2 to 5 mm inside the object (60 beyond 3 mm; two thirds are teapots),
32 leave the table by 10 to 20 mm, and 132 leave a contact 1.5 to 2 times as fast as the gripper.

The pushes are fast. The fitted particle xy scale is 46.4 mm per frame, against 17.2 mm in pickup's frozen entry; on
one 256-window training batch (seed 0, every particle's future steps), 36% of the steps exceed pickup's xy scale and
1.0% exceed the `rl_push` one. Train with the `rl_push` entry and unclipped targets: the default clamps normalized
targets to [-1, 1], which would clip every step beyond the p99 scale. The training split holds 1,084 objects and
32,974 episodes (103 excluded; 32,522 long enough for a C4 + F16 window from frame 3) and the validation split 117
objects and 3,652 episodes (7 excluded). The scales were refitted on the frozen copy after the outlier list grew from
67 to 110 episodes (artifact `rl-push-xy-p99-2026-09-15-frozen`); no value moved by more than 0.11%.

## Commands

```bash
# the specialist on the frozen release, two GPUs (after the pre-flight in CLAUDE.md)
srun --jobid=<id> --overlap --ntasks=1 --cpus-per-task=32 --gres=gpu:2 --cpu-bind=none \
    bash launch/legacy/train_rl_push_unet.sbatch bidir

# the same recipe as one process on the registered (live) release
python train.py --config configs/unet.yaml 'data.domains=[rl_push]' run_name=rl_push_unet \
    normalization.artifact=stats/normalization_rl_push.json model.articulated_bodies=true \
    train.clip_deltas=false train.clip_pose_deltas=false

# evaluation: one window per validation episode, or rollouts over the rest of each episode from frame 3
PYTHONPATH=. python scripts/unified/eval_unified.py --checkpoint runs/<run>/checkpoints/final.pt \
    --domains rl_push --rollout --rollout_frames 36 --clips all --seed 0 --out eval.json

# regenerate when the release changes (new files; pass --replace to overwrite deliberately)
PYTHONPATH=. python scripts/dev/rl_push_manifests.py val --out data/manifests/rl_push_val_assets.json
PYTHONPATH=. python scripts/dev/rl_push_manifests.py exclusions --stage1 <stage1_episodes.npz> \
    --stage2 <stage2_rows.json> --stage2_summary <stage2.json> --stage2_script <check_contacts.py> \
    --launches <launches.json> --out <new_exclusions.json>
PYTHONPATH=. python scripts/dev/rl_push_stats.py --root <release> --out stats/normalization_rl_push.json \
    --artifact_version <version>
```

`data.roots.rl_push.train` points at another release folder. Without a `val` folder, name the validation object
list among its exclusions, since an overridden domain applies only the files it names.

Validation reads objects interleaved across roots, then categories, so a `--clips N` prefix spans every root and
category it can. Each batch holds windows of one object file and one length; receipts count clips per object file
(`shards_read`). A rollout request longer than the episodes yields the rest of every episode, so horizons differ per
clip (`horizon_frames_min` and `_max`). Episodes are short (median 36 frames): from frame 3 with four context
frames, validation rollouts reach 18, 28 and 57 frames at the 10th, 50th and 90th percentiles.

Validation objects are new instances, not new shapes. After the best yaw, 116 of the 117 have a training object
whose 1,000-point template matches within point-sampling noise (median Chamfer 2.4% of the object's size), and
training objects are as close to one another. Validation therefore measures new pushes on seen shapes; a novel-shape
test needs a category or shape-cluster holdout with a cross-split geometry audit. `best.pt`
is also selected on this split, so it is not an unbiased final test. This split also does
not establish performance on unsuccessful pushes, new collection policies or real data.

## Audit

The acceptance audit behind these decisions was run with the scripts in
`/data/user_data/robertre/scratchpad/rl_push_0914/` (`RESULTS.md` there): integrity, forward kinematics, table plane,
speeds, tilt and a contact check of the gripper against the object's collision pieces on every episode
(`check_lowdim.py`, `check_contacts.py`), the launch check (`check_launches.py`: object centre speed against the
fastest gripper point at every contact that speeds the object up) and the validation shape check
(`val_leakage.py`). Known limits of the release: only successful pushes; four early object files store `source_run`
as one value (handled); per-object masses and friction are not recorded.

## The 2026-09-18 re-freeze: `pwm-data-v1/rl_push`

The owner's source has not changed since 2026-09-15 11:43 (57,153 episodes, 2,519,134 frames, 35.0 h; 1,062 of
1,202 objects at their 50-episode quota; the one object that had no episodes on 09-14 now has pushes). The three
acceptance checks were rerun on every episode (`/data/user_data/robertre/scratchpad/rl_push_0918/`: `stage1.json`,
`stage2_all.json` with `--random` covering all episodes, `launches.json`). Rates match the 09-14 audit: airborne
> 20 mm 0.075% (was 0.060%), sinking > 5 mm 0.084% (0.065%), contact launches > 2x 0.13% (0.10%), unexplained
settle kicks 0.5% (0.36%), no new failure kind, all 1,202 files load.

- **Outliers:** `data/manifests/rl_push_exclusions_20260918.json`, 219 of 57,153 episodes (0.38%): contact launch
  73, sinking 48, airborne 43, motion before contact is possible 23, unexplained gain 15, settle kick at frame 3 or
  later 18, gripper inside the object 9 (229 rule matches). The same rules and thresholds as the 09-15 list; the
  registered `rl_push_exclusions.json` (110 episodes, audit of 36,736) is unchanged so the 09-14 release, its
  launcher and the finished specialist keep their corpus identity.
- **Split:** the validation object list is unchanged (117 objects, seed 0) and defines the **test** objects. Their complete folder is
  `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/test_full` (117 objects, 5,598 episodes, 15 excluded, 5,534 usable from frame 3). Since 2026-09-19 the folder recipes read, `rl_push/test`, is a 200-episode seeded subset of it (user decision; one or two episodes per object, object files rewritten with the chosen episodes; an evaluation or validation batch holds one object's windows, so recipes read `log.val_batches: 117` and `log.sampler_batches: 60`). Train is
  `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/train` (1,085 objects, 51,555 episodes, 204 excluded, 50,815 usable; 1,134,427 window starts).
  Both folders are read whole through `data.roots.rl_push = {train, val: <test>, exclusions: {train: [new list],
  val: [new list]}}`; the test set doubles as the validation set (decision 2026-09-17).
  `scripts/unified/collection_recipe.py --indices <pwm-data-v1>` writes that block from the layout (every
  `rl_push_exclusions*.json` under `rl_push/manifests/`, no object list); `--rl-push-release` stays for the
  single-folder 09-14 release. Every file's sha256 was
  taken at the source before and after the copy and in the release (`RELEASE.json`, `INDEX.json` per split);
  `source/` holds the owner's README, `reconstruct.py`, `manifest.csv`, `robot/` and `runs/`; `manifests/` the
  two lists. Validation objects are still new pushes on seen shapes (the geometry did not change).
  At startup `utils/rl_push_release.verify_rl_push_release` checks each split folder the run reads: its
  `RELEASE.json` must be the one the dataset folder's `RELEASE.json` records, the two must agree on every object
  file, and the folder must hold exactly the listed files (the objects and `INDEX.json`) with the recorded sizes and
  sha256. Checkpoints bind both split manifests' digests, so a resume refuses a changed train or test folder
  (fixed 2026-09-18: the check used to know the single-folder 09-14 release only, and the first smoke jobs on this
  layout stopped there).
- **Scales:** the rl_push entries of the release's current scales artifact (`stats/normalization_collections_20260919_v7.json`; unchanged since v6). They
  come from the fit record `normalization_rl_push_20260918.json` (in git history since 2026-09-19; `rl-push-xy-p99-2026-09-18`, fitted on the train folder
  with the new list by `scripts/dev/rl_push_stats.py --manifests`): particle xy 46.01 / z 20.35 mm per frame, pose
  44.77 / 20.43 mm and 0.1686 / 0.2198 rad, within 1.5% of the 09-15 entry (46.39 / 20.08; 45.21 / 20.24; 0.1665 /
  0.2203). The entry still has to be carried into the collections artifact for a joint run.
