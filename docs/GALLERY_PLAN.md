# Gallery plan: many objects, many lifts, many obstacles

Written 2026-09-25 for the pickup-only checkpoint
`/data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt`,
guidance package `/data/user_data/mbronars/packages/guided_playback`, page https://claude.ai/artifact/KUQyckCP4x3t9ghkiXZTxh.

## 1. What the existing 11 creative cases measure (all from `runs/pickup/creative_*/replay_isaac/`)

Peak object lift in the floating-hand Isaac replay, mm, four members per case:

| object | unguided draws that lift (>100 mm) | guided members that lift |
|---|---|---|
| milk carton (4 cases) | 16 / 16 | 13 / 16 |
| peanut butter jar (2) | 7 / 8 | 1 / 8 |
| water bottle (2) | 8 / 8 | 1 / 8 |
| soda can (1) | 2 / 4 | 1 / 4 |
| pasta box (2) | 8 / 8 | 3 / 8 |

Measured differences between guided members that lift and those that do not (generation, at the frame the fingers close):

| | pre-grasp object motion in the generation (median, mm) | closure frame (median) |
|---|---|---|
| guided, lifts (19 members) | 1.2 carton, 9.3 jar, 13.3 bottle, 4.4 can, 2.9 pasta box | 19-25 |
| guided, fails (25 members) | 17.6 carton, 44.7 jar, 27.4 bottle, 7.5 can, 21.7 pasta box | 23-32 |
| unguided, lifts (41) | 0.8-4.3 | 14-20 |

The failing guided members move the object in the generation before the fingers close (up to 98 mm for the jar over
low books, 207 mm for one member); the lifting ones and the unguided draws do not (<= 13 mm).

Other facts that limit the current gallery:

- Every case used `seed 0`, so the four unguided draws per object are the same in every case (hand tracks agree to
  <= 2.5 mm over the first 25 frames between cases). Only four grasps per object have ever been tried.
- The chain replays the member with the lowest final objective. That member was the lifting one only for the carton.
  Lifting guided members exist for the pasta box (3), the jar (1), the bottle (1) and the can (1) but were never rendered.
- For a 20 cm carry, a 15-20 cm-deep obstacle plus the 40 mm buffer contains the object's start position; the box term is
  45 mm at iteration 0 in those cases (jar and bottle over low books, pasta box over the caddy).
- The sideways cabinet case: three of four guided members do not lift (6, 7, 4 mm); the rendered member (m2) lifts 247 mm.

## 2. Changes before running anything wide (about 1.5 h of code, all in this package)

1. **Grasp-phase freeze** (`gp/adapters/clean_pickup.py`, new knob `freeze_frames`): the optimized noise `P0` gets a
   `free` mask that is 0 for frames `< freeze_frames`. The optimizer loop already applies `state.free`
   (`gp/guide.py`), so the grasp comes from the unguided draw and guidance shapes only the carry. Default
   `freeze_frames = 24` (the unguided closure frames are 14-21).
2. **Pre-grasp stillness term** (`gp/objectives.py`, knobs `still_weight`, `still_frames`): mean object-centroid
   displacement from frame 0 over frames `< still_frames`, added to the objective. Kills the pre-grasp motion when the
   freeze alone is not enough (the model can still move the object through the later noise rows).
3. **Selection by replay, not by objective** (`sweeps/obst_case_chain.sh`): population 8 = two compiled batches
   (seeds 0 and 1), floating-hand replay of all 8, pick the member that lifts (> 100 mm) with the smallest generated
   distance, render the commanded arm for that member only. Costs about 5 min planning + 2 min replay per case.
4. **Geometry rules** (`bin/obstacle_box.py` / case specs): buffer 25 mm; carries >= 30 cm when the obstacle is deeper
   than 10 cm in the carry direction; obstacle placed so `start + object half-width + buffer < obstacle near face`.
5. **Page builder** for the gallery (`bin/build_gallery.py`): one row per case with the arm video, the start|end
   keyframes and the numbers; with/without-obstacle twins side by side.

Iterations stay at 40 for the main arms. One extra arm of the stage-1 test runs 80 iterations so the effect of more
iterations is measured, not assumed.

## 3. Stages (each stage stops for review)

**Stage 1, one case (about 25 min after the code lands):** peanut butter jar, air target +30 cm forward, +15 cm up,
no obstacle, changes 1-3 above, plus the 80-iteration arm. Success = at least one of eight members lifts in the
floating hand and the commanded arm ends within 60 mm of the target.

**Stage 2, the other objects on the same task (about 1.5 h):** bottle, can, pasta box, carton; same task, same settings.
This decides which objects go into the matrix and which need a different grasp seed budget.

**Stage 3, the gallery matrix.** Objects: carton, jar, bottle, can, pasta box (plus any of cereal box, cracker box,
mug from the val shards if stage 2 says the grasps hold). Moves per object:

| move | target | obstacle twin |
|---|---|---|
| A  air lift | +30 cm forward, +15 cm up | none |
| B  left on the ground | -25 cm y | none |
| C  right on the ground | +25 cm y | none |
| D  forward over a cardboard box | +35 cm forward, ground | cereal box 0002 lying on its long edge = 29 cm long, 3 cm deep, 22 cm tall, midway; and the same target without it |
| E  onto the cabinet, upright term | cabinet top (+20 cm z) at +30 cm forward | wooden cabinet 0170 (23 x 19 x 20 cm); twin = same target in the air, no cabinet |
| F  onto the cabinet to the left, upright term | cabinet top at -30 cm y | cabinet; twin = same target in the air |
| G  over a low row of books | +30 cm forward, +10 cm up | three 2-book stacks (7 cm high), 40 cm long; twin without |

That is 5 objects x 10 plans = 50 plans. At population 8 that is about 5 min of planning each and about 10 min of
replay + render each: about 12 h on one GPU, about 6 h wall on the two idle GPUs of the o9-24 allocation (a second
pickup-only server on GPU 1 replaces the idle two-domain server). Run as one queue script per GPU; nothing is
submitted to Slurm.

Barrier assets measured from their point clouds (cm, x y z):
cereal box 0002 22 x 3 x 29 (the "cardboard box"), cracker box 0001 11 x 6 x 15, food storage container 0002
10 x 10 x 13, handleless basket 0003 24 x 25 x 15, large mixing bowl 0001 25 x 25 x 11 (a "place into" target),
wooden cabinet 0170 23 x 19 x 20, wooden cabinet open 0001 26 x 36 x 22, libero basket 0002 22 x 24 x 17.

## 4. Gallery page

One page, grouped by object, newest first inside each group. Each cell: commanded-arm video (asset render), start|end
keyframes with the centroid trace, and the numbers: generated distance, Isaac distance, peak lift, obstacle
clearance, tilt at contact (place cases). Obstacle cases sit next to their no-obstacle twin. Failures go into a
separate section at the end with the same cells. Table at the top: per object x move, how many of the 8 members
lifted in the floating hand.

## 5. Stage 1 result (2026-09-25) and the reach rule

Jar, +30 cm x +15 cm z, freeze 24 + still 5, 8 members: 7/8 lift in the floating hand (259-368 mm; before the change 1/8),
generation 2-9 mm from the target. Commanded arm: 0/7 hold. Cause measured, not guessed: the peanut butter jar starts
0.61 m from the robot base, the target is at 0.90 m, the commanded hand reaches 0.82-0.86 m radius at 0.4-0.5 m height,
past the Franka's reach; the arm sits on a joint limit (margin -0.000 rad, hand rotation error 130-153 deg) in 5/7
members, and the two that lift (280 and 348 mm) lose the jar mid-carry at frames 40 and 50 while the jar pivots in the
fingers (tilt 17 deg at frame 31, 39 deg at frame 39).

**Reach rule for every gallery case:** target radius from the base <= 0.70 m and the object start <= 0.60 m; carries
from far objects go sideways or toward the robot. Start radii (m): pasta box 0.24, cereal box 0.24, food can 0.28,
cracker box 0.30, milk carton 0.34, jam jar 0.34, soda can 0.41, mayonnaise jar 0.41, pickle jar 0.42, water bottle 0.44,
spice jar 0.45, peanut butter jar 0.61.

**Reach rule, lower bound (pasta box, 2026-09-25):** an object 0.24 m from the base makes the elbow (joint 4) fold to its
-3.07 rad limit at the grasp in all 8 members; the arm then loses the box. Start radius must be >= 0.33 m. Objects in
band: carton 0.34, jam jar 0.34, juice carton 0.35, tea box 0.35, soda can 0.41, mayo jar 0.41, mustard bottle 0.41,
pickle jar 0.42, bottle 0.44, spice jar 0.45, tomato 0.45, lemon 0.50, tuna can 0.51 (peanut butter jar 0.61: sideways only).
Selection is now by the commanded ARM (gallery_repick.sh): every floating-hand lifter is arm-replayed, the member the arm
holds with the smallest final distance is rendered. Two pickup servers: GPU 0 :8768 (stage 2/3), GPU 1 :8769 (stage 2b).
