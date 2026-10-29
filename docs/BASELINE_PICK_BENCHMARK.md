# Pick-to-location benchmark: spec for baselines (PointWorld, GNS)

Written 2026-09-24. Run your planner on exactly these 25 problems (5 objects × 5 targets). You get the same starting
state and the same targets, and you are scored the same way as our two planners. The spec files are machine-readable,
and every number below was generated from them. Use the files rather than copying numbers from this page.

| file | contents |
|---|---|
| /data/user_data/mbronars/packages/guided_playback/sweeps/pick_benchmark_5x5.json | the benchmark spec: dataset, target rule, offsets, absolute targets, our planner settings, our run folders |
| /data/user_data/mbronars/packages/guided_playback/sweeps/pick_benchmark_5x5_scenes.json | per object: dataset shard/key/element, mesh, density, friction, frame-0 object centroid, frame-0 hand pose, finger opening, all 5 targets (model and raw frames) |
| /data/user_data/mbronars/packages/guided_playback/sweeps/pick_benchmark_5x5_scenes.npz | per object `<asset>__<field>`: `points_model` [1183, 3] frame-0 particles, `roles` (0 hand, 1 finger, 2 object), `particle_body`, `body_poses0_model` [K, 4, 4], `hand_pos0_raw`, `hand_quat0_wxyz`, `finger_q0`, `obj_pts0_raw` [512, 3], `targets_model` [5, 3], `targets_raw` [5, 3] |
| /data/user_data/mbronars/packages/guided_playback/sweeps/export_benchmark_scenes.py | the script that wrote the two scene files (it asserts every centroid against the dataset and the Isaac pool) |

## 1. Coordinate frames

- **Raw frame:** the dataset's world frame, i.e. the `particle_pos` / `link_pos` values in the tar samples, in metres.
- **Model frame** = raw + (−0.45, 0, −0.10) m. This is the pickup canonical translation (`CANONICAL_TRANSLATION["pickup"]`).
  Our world model and all our targets are in this frame.
- The scene files give both frames. Keep the frame you use explicit in your outputs.

## 2. The five episodes (initial placements)

All five come from **/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success**, shard gpu0/00044.tar
(val split). Each episode is identified by the sample `__key__` and the element index `element` within that sample's batch.

| asset | shard / key / element | object centroid, frame 0 (model, m) | hand position, frame 0 (model, m) | hand quaternion, frame 0 (wxyz) |
|---|---|---|---|---|
| milk_carton_asset_0037 | gpu0/00044.tar / batch_00002820 / 1 | −0.1785, 0.2096, 0.0730 | −0.2460, 0.1837, 0.3908 | 0.0705, 0.8276, −0.5501, −0.0863 |
| peanut_butter_jar_asset_0003 | gpu0/00044.tar / batch_00002817 / 3 | 0.1333, −0.1942, 0.0641 | 0.0786, −0.1212, 0.2950 | 0.2946, 0.9503, 0.0573, 0.0822 |
| soda_can_asset_0012 | gpu0/00044.tar / batch_00002822 / 2 | −0.0509, 0.0992, 0.0666 | 0.0812, 0.1831, 0.2890 | 0.2235, 0.9110, −0.0839, −0.3362 |
| pasta_box_asset_0058 | gpu0/00044.tar / batch_00002866 / 1 | −0.2380, −0.1150, 0.0541 | −0.1630, −0.1252, 0.3468 | −0.1764, 0.9137, −0.3128, −0.1901 |
| water_bottle_asset_0087 | gpu0/00044.tar / batch_00002822 / 0 | −0.0752, 0.2268, 0.0686 | 0.0884, 0.2671, 0.1954 | −0.0003, 0.7243, −0.0588, −0.6870 |

- **Hand:** the hand pose is the `panda_hand` link (link index 8 of `link_pos` / `link_quat`). The fingers start fully open,
  with `finger_q` = (0.04, 0.04) m.
- **Particles:** 1183 frame-0 particles per scene. The hand has 241, the two fingers 430 together, and the object 512.
  The object points are the dataset's own particles (`obj_ids == 0`).
- **Meshes, density, friction** (the values the Isaac evaluation uses):

| asset | mesh | asset_id | density kg/m³ | friction | size mm |
|---|---|---|---|---|---|
| milk_carton_asset_0037 | /data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/milk_carton/asset_0037 | 6574 | 806.4 | 1.0 | 56 × 47 × 132 |
| peanut_butter_jar_asset_0003 | /data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/peanut_butter_jar/asset_0003 | 7070 | 1402.4 | 1.0 | 57 × 60 × 93 |
| soda_can_asset_0012 | /data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/soda_can/asset_0012 | 7859 | 1118.9 | 1.0 | 57 × 56 × 102 |
| pasta_box_asset_0058 | /data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/pasta_box/asset_0058 | 7025 | 805.7 | 1.0 | 128 × 147 × 86 |
| water_bottle_asset_0087 | /data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/water_bottle/asset_0087 | 8788 | 1472.9 | 1.0 | 53 × 52 × 122 |

**Only frame 0 of the recorded episode is given.** Our planners read the recorded state at frame 0 and nothing after it.
Do not read recorded frames 1–39: they are the demonstration. If your model needs a history of several frames, repeat
frame 0 (the scene is at rest), and say that you did so.

## 3. Targets

- **Rule:** target = frame-0 object centroid + (dx, dy, +0.15 m). (dx, dy) is uniform over a 0.15 m-radius disc,
  drawn with `numpy.random.default_rng(0)`: five draws in order of r = 0.15·sqrt(U), angle = 2π·U.
- **The same 5 offsets are used for every object:**
  - t0 (−0.015, 0.119, 0.15)
  - t1 (0.030, 0.003, 0.15)
  - t2 (0.115, −0.070, 0.15)
  - t3 (−0.015, −0.116, 0.15)
  - t4 (0.102, −0.044, 0.15)

Absolute targets, model frame, m. The raw frame is (x + 0.45, y, z + 0.10), and it is also in the files.

| asset | t0 | t1 | t2 | t3 | t4 |
|---|---|---|---|---|---|
| milk_carton_asset_0037 | (−0.1935, 0.3286, 0.2230) | (−0.1485, 0.2126, 0.2230) | (−0.0635, 0.1396, 0.2230) | (−0.1935, 0.0936, 0.2230) | (−0.0765, 0.1656, 0.2230) |
| peanut_butter_jar_asset_0003 | (0.1183, −0.0752, 0.2141) | (0.1633, −0.1912, 0.2141) | (0.2483, −0.2642, 0.2141) | (0.1183, −0.3102, 0.2141) | (0.2353, −0.2382, 0.2141) |
| soda_can_asset_0012 | (−0.0659, 0.2182, 0.2165) | (−0.0209, 0.1022, 0.2165) | (0.0641, 0.0292, 0.2165) | (−0.0659, −0.0168, 0.2165) | (0.0511, 0.0552, 0.2165) |
| pasta_box_asset_0058 | (−0.2530, 0.0040, 0.2041) | (−0.2080, −0.1120, 0.2041) | (−0.1230, −0.1850, 0.2041) | (−0.2530, −0.2310, 0.2041) | (−0.1360, −0.1590, 0.2041) |
| water_bottle_asset_0087 | (−0.0902, 0.3458, 0.2186) | (−0.0452, 0.2298, 0.2186) | (0.0398, 0.1568, 0.2186) | (−0.0902, 0.1108, 0.2186) | (0.0268, 0.1828, 0.2186) |

## 4. Horizon, objective, seeds

- **Horizon:** 60 transitions (61 frames including frame 0) at 20 fps, i.e. 3.0 s.
- **Objective the planner optimizes:** the distance from the object's centroid at the last frame (frame 60) to the target.
  There is no extra shaping. Our planners use only this term.
- **Plan selection:** if your planner keeps several candidates, pick the final plan by the planner's own predicted
  objective. Never pick by the simulator result. The simulator is only used to score the plan you commit to.
- **Seeds:** use seed 0 for your own randomness (noise, action sampling), one plan per problem, and record the seed.
  For reference, our samplers use `torch.Generator('cuda').manual_seed(0)` for the initial noise. They also use
  `torch.manual_seed` for the model's random point sampling; the no-guidance planner uses seed `1000003·seed + 1009·replan + chunk`
  for each chunk of 4 candidates.

## 5. What to hand back per problem

The hand trajectory we replay in Isaac, as one npz per problem named `<method>_t<id>_<asset>.npz`:

| key | shape | meaning |
|---|---|---|
| `hand_pos` | [61, 3] | panda_hand link position per frame, **model frame**, m (frame 0 = the recorded frame-0 pose) |
| `hand_quat` | [61, 4] | panda_hand orientation per frame, wxyz |
| `finger_q` | [61, 2] | left / right finger opening, m (0.04 = open, 0 = closed) |
| `pred_obj_centroid` | [61, 3] | your model's predicted object centroid per frame, model frame (this is what the planner optimized) |
| `plan_seconds` | () | warm planning time for this problem: planner compute only, no simulator, model already loaded/compiled |
| `seed` | () | the seed used |

If your planner predicts full point clouds, you may instead return our rollout contract. That is `predicted` [1, 61, 1183, 3]
in the model frame, in the same point order as `points_model` (roles from `roles`). We then fit the hand pose per frame
ourselves.

## 6. How it is scored (we run this, identically for every method)

The Isaac floating-hand replay is /data/user_data/mbronars/packages/pickup_replay_isaac/scripts/replay_isaac.py,
driven by /data/user_data/mbronars/packages/guided_playback/gp/replay/isaac_float.py. For each plan:
- **Hand:** a floating Franka hand tracks the plan's hand pose at 20 fps (state-tracking controller, gravity off on the hand).
- **Gripper:** binary. It is commanded closed on frames where the mean finger opening is below 0.035 m (80 N), and open otherwise.
- **Object:** the convex hull of its frame-0 particles, with the density and friction from the table in section 2,
  on a kinematic table at the lowest frame-0 particle.
- **Reported numbers:**
  - Isaac distance to the target: the simulated object centroid at frame 60 vs the target (**the headline number**).
  - The planner's predicted distance to the target.
  - Isaac peak lift.
  - Planning time, as mean ± std over the 25 problems.

The summary rows are median, count within 50 mm and count within 60 mm.

Reference numbers on these 25 problems (Isaac distance to the target at frame 60):

| planner | median | ≤ 50 mm | ≤ 60 mm | plan time |
|---|---|---|---|---|
| ours, guided (10 iterations of Adam on the initial noise, population 4) | 38 mm | 18/25 | 21/25 | 126.5 ± 0.8 s |
| ours, no guidance (MPPI argmin, full 19-row windows, K = 20, commit 5) | 121 mm | 2/25 | 4/25 | 138.1 ± 0.6 s |
| ours, no guidance, earlier variant (truncated tail windows, K = 32, commit 5) | 135 mm | 2/25 | 4/25 | 190.4 ± 0.4 s |

Per-problem tables are in /data/user_data/mbronars/packages/guided_playback/sweeps/planner_eval.md and
/data/user_data/mbronars/packages/guided_playback/sweeps/mppi_full19_eval.md (and mppi_eval.md for the truncated variant). Our time is warm generation on one L40S,
bf16, compiled. Report your hardware with your times.
