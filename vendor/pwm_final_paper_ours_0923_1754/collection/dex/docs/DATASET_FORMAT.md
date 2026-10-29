# Dex-hand dataset format and training handoff

Verified on September 16, 2026 against the five v1 shards, the code in `collection/dex`, and `robert/clean` at
`0f1620f`. Sections 1–6 describe the data as stored and how it maps onto the jointed-body contract.

**Updates of 2026-09-18:**

- **Training support.** It has landed as the indexed `dex` domain (merge `706e694`); see
  [docs/DATASETS.md](../../../docs/DATASETS.md). Section 7 describes clean before that merge.
- **v2 is released separately.** `tactile_genesis_teachers_v2` (next to v1) is the source of `pwm-data-v1/dex_v2`
  (read with `--folder dex=dex_v2`); `pwm-data-v1/dex` keeps v1.
  - Train: boxes, 100 objaverse meshes and screwdriver models 1-3. Test: held-out objects only.
  - Shards are named `<teacher>@<part>` in `train/` and `test/`.
  - No perturbation. Terminations are recorded (`done_reason`), and the objaverse shards use a larger collision budget
    (section 8, item 7).
  - Its README is the dataset card.
  - `pwm-data-v1/dex_v2_filtered2` (read with `--folder dex=dex_v2_filtered2`) is `dex_v2` minus 2,934 episodes:
    simulator artifacts, milder single-finger rides and all repose XHand1 on boxes; train on it. `dex_v2_filtered`
    (minus 625) was the first filter. The dataset card's section Filtered release has the criteria and counts.
- **Sections 1-2 and 8 describe v1.** The schema in section 3 covers both versions.

## 1. What to train on

```text
/data/group_data/katefgroup-ssd/world_model/robertre_dexhand/tactile_genesis_teachers_v1
```

Five shards, one per tactile-genesis teacher policy, each one rollout of 2,048 simulated envs from reset. Every env
contributes **one episode**: its frames up to its first termination. 1,638 envs per shard ran the policy as is and 410
had noise added to their actions (`episode_perturbed`, section 8).

| Shard | Task | Hand | Step limit | Episodes ≥ one 20-frame window | Frames at 40 Hz |
| --- | --- | --- | ---: | ---: | ---: |
| `in_palm_rotate-xhand1` | turn an object in the palm | xhand1, 12 joints | 400 | 2,041 | 776,879 |
| `in_hand_repose-xhand1` | reorient an object | xhand1 | 800 | 2,033 | 1,592,136 |
| `in_hand_repose-sharpa` | reorient an object | sharpa, 22 joints | 800 | 2,047 | 1,585,795 |
| `screwdriver-xhand1` | spin a screwdriver | xhand1 | 800 | 1,517 | 1,099,331 |
| `screwdriver-sharpa` | spin a screwdriver | sharpa | 800 | 1,520 | 967,543 |
| **Total** | | | | **9,158** | **6,021,684** |

10,240 episodes, 6.02 M control steps (41.8 h of robot time), 4.7 GB. A 20-frame window at 20 Hz spans 39 control
steps, so shorter episodes cannot fill one. Windows overlap when sampled; do not count them as independent episodes.

This is a **staging dataset, not an approved release**. There is **no held-out export**: every episode comes from the
same five rollouts, so a validation split has to be cut from these shards (section 7). Screwdriver object model 0 is
unusable (section 8); leave it out.

## 2. Files and episode identity

```text
tactile_genesis_teachers_v1/
  README.md                        dataset card
  <teacher>.npz                    the shard (np.savez: members stored uncompressed)
  <teacher>.json                   manifest: settings, checkout commit, timing; a shard without one is incomplete
  <teacher>.log                    generator log
  <teacher>_outcomes.npz / .json   per-episode outcome labels (collection/dex/outcomes.py)
  _code/                           the spike versions of the generator, launcher, labeler and reader that made v1
  _eden_logs/                      simulator framework logs
```

An episode is `(teacher, env)`, with `episode_env[e] == e`; use `f"{teacher}/{env:04d}"` as its stable ID. Its frames
are `0 .. episode_length[e] - 1` of column `e`. Later frames in the same column belong to the env after its automatic
reset and are not part of any episode.

`episode_shape_id[e]` names the env's object model: 17 in-hand models (15 boxes of 30–45 mm and 2 cylinders, the
`OBJECTS_4CM_16` list in tactile-genesis; models 15 and 16 are the cylinders) and 4 screwdriver models.
**Envs are assigned to object models in contiguous blocks**: 120 or 121 envs per in-hand model and exactly 512 per
screwdriver model (envs 0–511 are screwdriver model 0). Taking "the first N envs" therefore covers one or two object
models. Splitting by object model means splitting by env block.

## 3. Stored schema

`T` = step limit (400 or 800), `E` = 2,048, `L` = recorded links (13 xhand1, 23 sharpa), `D` = joints (12 or 22),
`A` = action size (5, 12, 22, 8, 13 in table order above).

| Key | Shape | dtype | Meaning / units |
| --- | --- | --- | --- |
| `link_pos` | `[T,E,L,3]` | float32 | World position of every recorded hand link, metres |
| `link_quat` | `[T,E,L,4]` | float32 | World orientation, **wxyz**, unit length (within 2e-7) |
| `obj_pos` | `[T,E,3]` | float32 | World position of the object's frame origin (not its centroid), metres |
| `obj_quat` | `[T,E,4]` | float32 | Object orientation, wxyz |
| `dofs` | `[T,E,D]` | float32 | Hand joint angles, radians, in the order of section 4 |
| `action` | `[T,E,A]` | float32 | Raw policy output, noise included, **unclipped** |
| `reward` | `[T,E]` | float32 | Reward of step `t` |
| `done` | `[T,E]` | bool | True on the step whose result ended the episode |
| `time_out` | `[T,E]` | bool | Never true in v1; the current generator sets it from the task's time-out term |
| `done_reason` | `[T,E]` | uint8 | Current generator only: which termination fired at step `t`, an index into the manifest's `done_reasons` (0 none; `solver_error` = Eden's reset of an env the Genesis rigid solver flags) |
| `episode_env`, `episode_length`, `episode_shape_id` | `[E]` | int64 | Episode table |
| `episode_terminated_early`, `episode_perturbed` | `[E]` | bool | Episode table |
| `episode_end_reason` | `[E]` | uint8 | Current generator only: `done_reason` at the episode's last step |
| `episode_return` | `[E]` | float32 | Sum of `reward` over the episode |
| `link_names` | `[L]` | str | Recorded link order (below) |
| `dof_names` | `[0]` | float64 | **Empty in v1**; the joint order is in section 4 |
| `control_dt` | `[]` | float64 | 0.025 s |
| `objgeom_verts_i` | `[V_i,3]` | float32 | Vertices of object piece `i`, already in the **object frame**, metres |
| `objgeom_faces_i` | `[F_i,3]` | int64 | Triangles of piece `i` |
| `objgeom_envs_i` | `[n_i]` | int64 | Envs whose object includes piece `i` |
| `objgeom_pos_i`, `objgeom_quat_i` | `[3]`, `[4]` | float32 | Piece offsets. **Do not apply**: the vertices already include them |

An env's object is the union of the pieces whose `objgeom_envs_i` contains it. In-hand objects are one piece each
(boxes 8 vertices, cylinders 66; 17 pieces). Screwdrivers are 2–3 convex pieces each (10 pieces over 4 models). Applying the
piece offsets again floats a screwdriver about 5 cm off the fingers. Object points for training are sampled from the
outer surface of that union: a multi-piece object's faces between pieces lie inside it and are skipped
(`scripts/unified/dex_geometry.outer_surface_points`, used by the index builder since 2026-09-18). The v1 geometry files
were built from the whole union; for screwdriver models 0–3 that puts 0.7 / 5.7 / 0.3 / 0.1% of the sampled surface
on inner faces.

`<teacher>_outcomes.npz`: `outcome` `[E]` int8 (0 ran to the step limit, 1 dropped, 2 ended on a success streak,
3 other early termination, 4 solver error), `end_object_z` `[E]` float32 metres, and for the screwdriver shards
`tilt_end_deg` `[E]` float32. v1 labels are inferred from the object state; shards that record `episode_end_reason`
are labelled from it. The v1 manifests record the tactile-genesis commit; shards made by the current generator also
record generator, config and checkpoint hashes, the termination names (`done_reasons`, `end_reasons`) and, when used,
the grasp file (`grasps_file`), object list (`objects_file`) or kept object variants (`object_variants_kept`).

Recorded link order (Genesis's order, root first; fixed URDF links are already welded into their parents):

```text
xhand1: right_hand_link, right_hand_thumb_bend_link, right_hand_index_bend_link, right_hand_mid_link1,
        right_hand_ring_link1, right_hand_pinky_link1, right_hand_thumb_rota_link1, right_hand_index_rota_link1,
        right_hand_mid_link2, right_hand_ring_link2, right_hand_pinky_link2, right_hand_thumb_rota_link2,
        right_hand_index_rota_link2
sharpa: right_hand_C_MC, right_thumb_CMC_VL, right_index_MCP_VL, right_middle_MCP_VL, right_ring_MCP_VL,
        right_pinky_MC, right_thumb_MC, right_index_PP, right_middle_PP, right_ring_PP, right_pinky_MCP_VL,
        right_thumb_MCP_VL, right_index_MP, right_middle_MP, right_ring_MP, right_pinky_PP, right_thumb_PP,
        right_index_DP, right_middle_DP, right_ring_DP, right_pinky_MP, right_thumb_DP, right_pinky_DP
```

## 4. Units, frames and time

- **One world frame for every env.** Positions are metres, z up. The hand is fixed-base: its root link has the same
  pose in every frame of every env (maximum deviation 0.0 mm), so nothing needs per-env recentring. The hand's
  placement differs by task:

  | Shard | Root link position (m) | Root link orientation (wxyz) |
  | --- | --- | --- |
  | `in_palm_rotate-xhand1`, `in_hand_repose-xhand1` | (0.070, 0.000, 0.475) | (0.7071, 0, −0.7071, 0) |
  | `in_hand_repose-sharpa` | (0.070, −0.015, 0.480) | (0.7071, 0, −0.7071, 0) |
  | `screwdriver-xhand1` | (0.015, −0.055, 0.250) | (0.4056, −0.5792, 0.5792, 0.4056) |
  | `screwdriver-sharpa` | (−0.055, −0.020, 0.290) | (0.5, 0, 0.866, 0) |

- **Object start.** In-hand objects start with their origin at z = 0.52 m, spread over 36 mm in x and y. A screwdriver's
  frame origin sits near z = 0 while its geometry is centred about 0.13 m up, so use the rebuilt points, not
  `obj_pos`, for anything positional.
- **Time.** One control step is 0.025 s (40 Hz); 20 Hz means every other step. At step `t` the generator recorded
  the state, then applied `action[t]`; `reward[t]` and `done[t]` are that step's result. The state after an
  episode's last action is not recorded.
- **Targets.** Nothing on disk is a delta. The contract's per-transition displacements and pose deltas are computed
  from these poses at the chosen stride.
- **Joint order.** `dofs[..., j]` follows the hand's `dofs_name` list in tactile-genesis
  (`dexterous-hands/src/entities/robots/{xhand1,sharpa_hand}.py`). For xhand1 the recorded angles lie inside the URDF
  limits 99.9% of the time under that order, against 93.0% under the URDF's declaration order. For sharpa the two
  orders are the same. The link order above is a different order; map joints to links through the URDF.

  ```text
  xhand1: thumb_bend_joint, index_bend_joint, mid_joint1, ring_joint1, pinky_joint1, thumb_rota_joint1,
          index_joint1, mid_joint2, ring_joint2, pinky_joint2, thumb_rota_joint2, index_joint2   (each "right_hand_" + name)
  sharpa: thumb_CMC_FE, thumb_CMC_AA, thumb_MCP_FE, thumb_MCP_AA, thumb_IP, index_MCP_FE, index_MCP_AA, index_PIP,
          index_DIP, middle_MCP_FE, middle_MCP_AA, middle_PIP, middle_DIP, ring_MCP_FE, ring_MCP_AA, ring_PIP,
          ring_DIP, pinky_CMC, pinky_MCP_FE, pinky_MCP_AA, pinky_PIP, pinky_DIP                     (each "right_" + name)
  ```

- **Actions.** The two repose teachers command all joints; in-palm rotation (5) and the screwdriver teachers (8, 13)
  command a subset, and in-palm rotation holds the other joints at their reset angles. A commanded joint's PD target
  is `raw × 0.5 × (upper − lower)` plus a default pose of 0. **Which column drives which joint is not recorded and has
  not been verified.**
- **Motion per 20 Hz step**, 99th percentile, from 256 clean envs spread over every object model (screwdriver
  model 0 left out). Object rotation is the angle between consecutive orientations.

  | Shard | Object xy (mm) | Object z (mm) | Object rotation (°) | Finger link xy / z (mm) | Joint angle (°) |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | `in_palm_rotate-xhand1` | 8.6 | 7.3 | 28.1 | 11.7 / 10.1 | 16.2 |
  | `in_hand_repose-xhand1` | 8.7 | 6.1 | 24.3 | 9.6 / 12.5 | 19.5 |
  | `in_hand_repose-sharpa` | 12.2 | 8.1 | 36.4 | 12.6 / 15.5 | 19.0 |
  | `screwdriver-xhand1` | 4.3 | 0.8 | 11.3 | 6.9 / 2.4 | 10.0 |
  | `screwdriver-sharpa` | 1.5 | 2.3 | 19.4 | 9.5 / 3.1 | 8.8 |

## 5. Read one episode on CPU

The `collection.dex` reader needs only NumPy. Run from the repository root with any Python that has it:

```python
import json
import numpy as np
from collection.dex.points import episode_points, hand_bodies, object_points

root = "/data/group_data/katefgroup-ssd/world_model/robertre_dexhand/tactile_genesis_teachers_v1"
shard = "in_palm_rotate-xhand1"
z = np.load(f"{root}/{shard}.npz")
urdf = json.load(open(f"{root}/{shard}.json"))["robot_file"]
env = 703
hand = hand_bodies([str(n) for n in z["link_names"]], urdf)   # canonical points per recorded link
obj, pieces = object_points(z, env)                            # 512 points in the object frame
X = episode_points(z, env, hand, obj)                          # [length, N, 3] world metres, object first
outcome = np.load(f"{root}/{shard}_outcomes.npz")["outcome"][env]
print(X.shape, len(obj), sum(len(b) for b in hand), "pieces", pieces, "outcome", int(outcome))
```

It prints `(384, 2158, 3) 512 1646 pieces [5] outcome 2`: a 384-step episode that ended on a success streak, with
512 object and 1,646 hand points. sharpa hands get 1,682 points. The command-line check adds contact distances:

```bash
python -m collection.dex.points --shard <root>/in_palm_rotate-xhand1.npz --env 703
# env 703: 384 frames, 2158 points, geoms [5], object extent [42.0, 42.0, 42.0] mm
# object-to-hand gap: min 0.1 mm, median over frames of per-frame min 1.0 mm; finite True
```

The manifests name the URDF by its absolute path in the v1 checkout; pass `--urdf` for another checkout.
`np.load(npz)[key]` reads the whole member on every access, so slice each array once. A loader can instead
memory-map a member, because `np.savez` stored them uncompressed:

```python
import struct, zipfile
import numpy as np

def npz_member(path, key):
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(key + ".npy")
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        n_name, n_extra = struct.unpack("<HH", fh.read(30)[26:30])
        fh.seek(info.header_offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(fh)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(fh)
        offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, order="F" if fortran else "C", offset=offset)
```

## 6. From poses to jointed bodies

Bodies at link granularity, as in the one batch built so far:

- body 0: the object, a free passive body; its points come from its pieces;
- the hand root: a free body, actuated but static;
- each revolute link: a **revolute child** of its parent body, actuated, one joint channel. Fixed URDF links are
  folded into their revolute parents (`collection/dex/geometry.py:merge_fixed`), which reproduces exactly the links
  Genesis records.

Hand points are sampled by mesh area. For each window the joint description is taken at its first frame `t0`: the
axis is `R_parent(t0) · R_rpy(joint origin) · axis_urdf` in world coordinates, and the pivot is the parent link's
world joint origin minus the parent body's centroid, both at `t0`.

Under the current contract the actuated bodies' input is their **achieved** motion, i.e. the recorded joint angles.
Section 8 explains why that may be the wrong input for this data. One batch shares one joint layout
(`utils/pose_native.py:layout_from` takes one layout row), so **xhand1 and sharpa windows cannot share a batch**.

Checked: on an export of `robert/clean` at `0f1620f`, the spike fixture builder (section 7) turned a v1-format
xhand1 shard into a 4-window jointed batch (4 context and 16 future frames at 20 Hz, N = 2,158, 14 bodies). Rebuilding
the particles from the joint description missed by 0.0002 mm. The builder hardcodes the xhand1 URDF, so **sharpa's
joint round trip is unchecked**. Its poses do check out: rebuilt sharpa points sit about 1 mm from the object.

## 7. Training: what exists and what a loader must decide

On `robert/clean` at `0f1620f`:

- **Exists.** Jointed bodies (`model.articulated_bodies`, joint types in `utils/pose_native.py`), the fixed-batch
  overfit harness, and a working template for an opt-in rigid domain that builds points from poses at load time:
  `rl_push` (`data/rl_push.py`, [docs/RL_PUSH.md](../../../docs/RL_PUSH.md)).
- **Missing.** There is no `dex` domain: `data.domains=[dex]` fails validation
  (`data/unified_dataset.py:selected_domains`). A domain needs:
  - an adapter;
  - entries in `OPTIONAL_DOMAINS`, `CANONICAL_TRANSLATION`, `DOMAIN_DT` and `DOMAIN_FUTURE_LEN`
    (`data/unified_dataset.py`), and in `OPTIONAL_DOMAINS` / `OPTIONAL_RIGID_DOMAINS` (`data/normalization.py`);
  - a material-ID entry for the UNet (`utils/config.py`, `material_ids`) and a config check that requires its
    normalization artifact;
  - its own normalization artifact under `stats/`;
  - exclusion and validation manifests, tests, and this domain's decision record.

  Any real training launch then needs the mandatory pre-flight.

The only runnable path today is the overfit harness on a saved batch. It ran on September 11 on `robert/yikai-native`
(8-seed final 0.94 ± 0.07 mm on 4 xhand1 windows, a fixed-batch fit, not a benchmark). The harness flags below exist on
`robert/clean`, but the harness has not been re-run there.

```bash
# a jointed batch (spike builder: hardcodes the xhand1 URDF, loads whole arrays)
/data/user_data/robertre/conda_envs/pwm_blackwell/bin/python \
  /data/group_data/katefgroup/robertre_dexhand/spike/build_fixture.py \
  --rollout <xhand1 shard>.npz --name <fixture> --batch_dir <dir> --repo <checkout> --articulated
# fit it; the batch is labelled pickup and borrows pickup's normalization, so clipping is off
PYTHONPATH=. WANDB_MODE=disabled python scripts/unified/overfit_prod.py \
  --config configs/unet.yaml --domain pickup --batch_name <fixture> --batch_dir <dir> \
  --require_saved_batches --steps 10000 --eval_rng legacy --tag <tag> --out_root <dir> \
  --override model.articulated_bodies=true train.clip_deltas=false train.clip_pose_deltas=false
```

Decisions a loader has to make, with what the data says:

1. **Actuator input.** Recorded joint angles work with today's contract. Joint torques would make plans executable
   in the simulator and carry grip force (section 8), but v1 did not record them, so they need a v2 collection.
2. **Rate and window.** The data is 40 Hz. The overfit used 20 Hz with 4 context and 16 future frames
   (pickup uses 38 future frames). Halving the rate doubles per-step motion.
3. **Normalization.** Translations fit pickup's pose scales (15.4 mm xy, 28.2 mm z per step), but object rotation
   does not. Its 99th percentile per 20 Hz step is 11–36°, against pickup's per-component rotation scales of 7.5° (xy)
   and 10.8° (z). A dex artifact must be fitted on training episodes only.
4. **Canonical shift.** Each task places the hand differently (section 4). One fixed shift per domain has to keep
   every position within the ±0.8 m position bound.
5. **Validation split.** Hold out whole object models (env blocks) or hold out episodes. Never put windows of one
   episode on both sides. Only three screwdriver models are usable.
6. **Exclusions.** Screwdriver model 0; episodes shorter than one window; perturbed `screwdriver-sharpa` episodes
   (all 410 fail).
7. **Batching and point budgets.** Batch per hand (section 6). Point counts differ: 1,646 xhand1 and 1,682 sharpa
   hand points by default, plus the object.
8. **Physical inputs.** The object's mass and friction were randomized per env and not recorded, so the model sees
   rigid fill values for them, as in the other rigid domains.

## 8. Known issues that affect training

1. **Screwdriver model 0 (`screwdriver0_fat.obj`, `episode_shape_id == 0`, envs 0–511) is unusable.** At reset its
   geometry sits about 15 cm higher than the other three models (centroid z ≈ 0.28 m against 0.13 m). The cause is
   upstream: tactile-genesis's grasp files (`src/assets/grasps/screwdrivers_{xhand1,sharpa}_grasps_64.pt`, 64 saved
   starting grasps per model) store this model's object pose 14.8 cm higher than the other three while the hand pose
   is the same, and the recorded start heights match the file exactly. Upstream is unchanged at the pinned commit. **Every clean
   episode with it fails**:
   - xhand1: all 408 fail within 2 s, median length 2 steps;
   - sharpa: all 410 fail, 97% within 2 s, median length 9 steps.

   With models 1–3, 7–23% of clean episodes fail per model, almost never within 2 s. Model 0 accounts for 408 of the
   593 clean xhand1 failures and 410 of the 547 clean sharpa ones. A grasp file with model 0's object height lowered
   by that offset (0.1476 m xhand1, 0.1481 m sharpa; generator option `--grasps`) makes model 0 behave like the others
   (0 of 32 and 1 of 32 early failures in the 2026-09-18 pilot); the index builder drops model 0 only from shards made
   with the shipped grasp file.
2. **The in-hand teachers act at the joints' torque limits.** Their raw actions are tens to hundreds in size, so
   PD targets land far past the joint limits and the simulator clips the force. Recorded joint angles do not show
   that force, and torques were not recorded. A plan written in joint angles cannot be replayed faithfully.
3. **The perturbation is not calibrated.** Noise of std 0.3 changes nothing for the in-hand teachers, so treat their
   perturbed episodes as clean. It breaks every perturbed `screwdriver-sharpa` episode.
4. **v1 outcome labels are inferred.** For in-palm rotation, dropped vs success streak is a height split; v1 does
   not record which termination fired, and its `time_out` is never set (use `episode_length < T`). Eden also ends
   envs that the Genesis rigid solver flags (contact-buffer overflow, NaN), which the height rules cannot tell apart.
   Checked on 2026-09-18: all 582 v1 in-palm success streaks turned the 1,250–1,350° a 15-goal streak needs, and 3 of
   2,048 repose-sharpa "drops" ended with the object still up (likely solver resets). Shards from the current
   generator record the termination (`episode_end_reason`) and are labelled from it.
5. **Hidden and soft physics.** Object mass and friction are randomized per env and not recorded: a mass shift of
   0–100 g (in-hand) or 0–200 g (screwdriver), friction ×0.5–1.5 (in-palm rotation), ×0.3–2.0 (repose) or ×0.2–1.2
   (screwdriver). The contact time constant is 10 ms and the no-slip pass is off.
6. **Missing names.** `dof_names` is empty and the action-column mapping is unrecorded (section 4).
7. **Mesh objects (v2).** Genesis builds a mesh object's collision shape itself: one hull, or CoACD pieces (Eden
   defaults: error threshold 15%, CoACD threshold 0.05). About half of v2's objaverse meshes become more than 20
   pieces, up to 426.
   - **Collision budget.** At the in-hand tasks' own budget (`max_collision_pairs` 30 repose, 48 in-palm) such
     objects overflow Genesis's contact buffers, and Eden resets the env: a `solver_error` end, 7.1% of the first
     objaverse train episodes. v2's objaverse shards were regenerated at `--max_collision_pairs 500`
     (`pilots/contact_budget_20260918`), and the index builder drops any episode that still ends that way.
   - **Mass.** Thin-walled meshes weigh under a gram at the default density. The randomized mass shift is a point
     mass at the centre of mass, so their rotational inertia stays small.
8. **Fingers sink into objects (v1 and v2).** Contacts are soft: a 10 ms constraint time constant at a 5 ms step,
   with the teachers squeezing at the torque limit. The hand and object collision shapes therefore overlap:
   - boxes: median 1.3-1.6 mm per frame, p99 about 15 mm, a quarter of frames past 5 mm, up to 2.4 cm;
   - objaverse meshes: less.

   A rollout that recorded Genesis's own contact depths agrees with the offline measure (correlation 0.987). Receipts
   are in `tactile_genesis_teachers_v2/_checks/`.

## 9. Acceptance and reproducibility checklist

- Use a shard only with its manifest, and outcome labels only from `_outcomes.*`.
- Record, for any run:
  - the shard paths and the teacher/env selection;
  - the stride and the context/future lengths;
  - the exclusions (model 0 explicitly) and the split definition;
  - the normalization artifact and its checksum;
  - the tactile-genesis commit from the manifest.
- v1 was generated with `deterministic=False`. Regenerate into a new directory: the generator refuses to overwrite a
  shard, and a rerun is not bit-identical.
- Report episodes, 40 Hz frames and windows separately. Viewer playback rates and training-window counts are not
  dataset sizes.

Code: [generator](../generate.py), [reader](../points.py), [geometry](../geometry.py), [labels](../outcomes.py),
[collection README](../README.md), [batch contract](../../../data/batch.py),
[unified adapters](../../../data/unified_dataset.py), [joint layout](../../../utils/pose_native.py).

## 10. Checks performed for this document

- **Schema.** Array names, shapes and dtypes were read from the zip headers of all five shards. Every member is
  stored uncompressed, and the memory-mapped `obj_pos` equals `np.load` in all five.
- **Frames and episodes.** Root-link poses are identical across all frames and envs (0.0 mm). Quaternion norms are
  within 2e-7 of 1. Every early-ended episode's length is its first `done` plus one. `time_out` is never set.
  Every object piece covers one contiguous env block.
- **Motion statistics.** Computed from 256 clean envs per shard spread over all object models.
- **Screwdriver model 0.** The per-model outcomes and starting heights come from the outcome files and the recorded
  poses.
- **Examples.** The section 5 example and command printed the output shown.
- **Fixture route.** The builder ran against a `git archive` export of `0f1620f` (section 6). The overfit harness
  was not run.
- **Nothing launched or changed.** No training was launched and no dataset file was modified.

Receipts are in `/data/group_data/katefgroup/robertre_dexhand/spike/`: `dataset_format_facts.{py,json}`,
`screwdriver_variants.{py,json}`, `guidance_readiness.{py,json}`.
