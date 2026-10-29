# Dex-hand teacher rollouts

Rigid-state rollouts of the five teacher policies shipped with
[tactile-genesis](https://github.com/neuroagents-lab/tactile-genesis) (Eden on its vendored Genesis 1.1.2): two hands
(xhand1 with 12 revolute joints, sharpa with 22) on three tasks (turn an object in the palm, reorient it, turn a
screwdriver). The in-hand tasks share 17 object models of about 4 cm (15 boxes, 2 cylinders); the screwdriver tasks use
4 screwdrivers. A shard stores poses, not points: every robot link and the object at each 40 Hz control step, plus each
env's object geometry. Point clouds are rebuilt from the robot URDF meshes and those poses.

The tactile-genesis checkout and its environment are external, and training never imports them.

- **v2** (2026-09-18) is the source of `pwm-data-v1/dex_v2`: boxes, 100 objaverse meshes and screwdriver models 1-3 in
  train, and a test set of held-out objects only. It lives at
  `/data/group_data/katefgroup-ssd/world_model/robertre_dexhand/tactile_genesis_teachers_v2`, and its README is the
  dataset card. Train on `pwm-data-v1/dex_v2_filtered2`, which drops 2,934 episodes: simulator artifacts (start pops,
  buried fingers, screwdriver ejections, objects riding on a finger, including milder rides) and every repose-XHand1
  episode on boxes, where most of the fingers sunk into objects were (the card's section Filtered release).
- **v1** (10,240 episodes, 2026-09-13) is at `.../tactile_genesis_teachers_v1`, released as `pwm-data-v1/dex`.

For the stored format, units, reading an episode and the training status, start with
[Dataset format and training handoff](docs/DATASET_FORMAT.md).

```text
collection/dex/
  generate.py        # one shard: roll out a teacher, record link and object poses and object geometry
  outcomes.py        # per-episode outcome labels
  points.py          # rebuild an episode's point cloud from a shard
  geometry.py        # URDF meshes, surface sampling, fixed-link folding
  launch/
    setup_tactile_genesis.sh  # pinned checkout, LFS teacher checkpoints, Python environment
    tactile_genesis_uv.lock   # the package versions v1 ran with
    run_teachers.sh           # the five v1 shards on two GPUs
  docs/
    DATASET_FORMAT.md         # schema, units, reading an episode, training status and decisions
```

## Interfaces

Run from the repository root. `PY` is the checkout's interpreter, `<checkout>/dexterous-hands/.venv/bin/python`.

| Entry point | Purpose |
| --- | --- |
| `bash collection/dex/launch/setup_tactile_genesis.sh <parent>` | pinned checkout (commit 3023bcd), LFS teacher checkpoints, Python environment from `tactile_genesis_uv.lock` |
| `$PY -m collection.dex.generate` | one shard: one teacher, one rollout of `--num_envs` envs, on a GPU |
| `bash collection/dex/launch/run_teachers.sh <checkout> <out_dir> [gpu_a gpu_b]` | the five shards with the v1 settings, two GPUs in parallel |
| `$PY -m collection.dex.outcomes <out_dir>` | per-episode outcome labels next to each finished shard |
| `$PY -m collection.dex.points --shard <shard>.npz --env <e> [--png f.png]` | rebuild one episode's point cloud and check object-to-hand contact |

`outcomes` and `points` need only NumPy (matplotlib for `--png`), so any Python environment works for them.

Beyond the v1 settings, `generate` takes these options:

- `--grasps <file.pt>`: a screwdriver grasp file in place of the task's own.
- `--objects <list.json>`: mesh files replacing the in-hand object list and its goal ghost.
- `--object_variants 0,...`: keep some of the task's own objects. For the screwdriver, keep a prefix, or pass
  `--grasps` with a file holding exactly the kept models, because the grasp file maps models by position.
- `--part <name>`: names the shard `<teacher>@<part>`.
- `--max_collision_pairs <n>`: replaces the task's rigid-solver collision budget. The in-hand tasks set 30 (repose) or
  48 (in-palm), which is enough for single-hull boxes. Meshes that Genesis splits into many convex pieces overflow
  that budget, and the env ends as a solver error; v2's objaverse shards use 500.

## Reproducing v1

```bash
bash collection/dex/launch/setup_tactile_genesis.sh <parent>   # needs git-lfs and uv
srun --jobid=<job with 2 GPUs> --overlap --ntasks=1 --gres=gpu:2 \
    bash collection/dex/launch/run_teachers.sh <parent>/tactile-genesis <new out_dir>
<parent>/tactile-genesis/dexterous-hands/.venv/bin/python -m collection.dex.outcomes <new out_dir>
```

v1 used 2,048 envs per teacher, each task's full episode length (400 steps for in-palm rotation, 800 otherwise), noise
of std 0.3 on the raw action of 20% of envs, and seeds 0-4 in the order of `run_teachers.sh`. It took about 6 minutes
on two RTX PRO 6000: about 70 s to build each environment, then roughly 35K env-steps/s. The generator initializes Eden
with `deterministic=False`, so a rerun is not guaranteed to match v1 bit for bit.

## Reproducing v2

v2's inputs and launchers live in its dataset folder (`tactile_genesis_teachers_v2`):

1. `pool/prep_objects.py`: the 170-mesh objaverse pool.
2. `split_objects.py`: the object split and grasp files.
3. `run_generate_fixed.sh <gpu>`: the box and screwdriver shards.
4. `run_regen_objaverse_budget.sh <gpu>` (one call per worker, any number of GPUs): the objaverse shards at
   `--max_collision_pairs 500`.
5. `python -m collection.dex.outcomes` on `train/` and `test/`.

`check_v2.py` checks the result. Seeds and timings are in its README.

## Shard format

One `<teacher>.npz` per teacher holds `[T steps, E envs, ...]` arrays of link and object poses, joint angles, raw
actions and rewards, which termination fired (`done_reason`, not in v1), plus the episode table and the object pieces. `<teacher>.json` is the manifest, written last, and
`<teacher>_outcomes.npz/.json` hold the outcome labels. The full schema, units and frame conventions are in
[docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md).

## Known issues

1. The perturbation is not calibrated. The in-hand teachers' raw actions are unclipped and in the tens to hundreds, so
   noise of std 0.3 changes nothing for them, while it breaks every perturbed screwdriver-sharpa episode.
2. The in-hand teachers act at the joints' torque limits: joint targets (raw x 0.5 x joint range) land far past the
   joint limits and Genesis clips the force. Recorded joint angles do not carry that force, and torques are not recorded.
3. Screwdriver model 0 (`episode_shape_id == 0`, envs 0-511 in both screwdriver shards) starts about 15 cm higher
   than the other three models, and every clean episode with it fails, most within a few steps. The cause is
   tactile-genesis's own grasp file, which stores this model's starting object pose 14.8 cm too high. It accounts for
   most screwdriver failures; leave it out of v1. A corrected grasp file (`--grasps`) fixes it for new shards.
4. v1 labels dropped vs success streak (in-palm rotation) by a height split, because v1 does not record which
   termination fired; the current generator records it (`done_reason`, including Eden's solver-error resets).
5. v1's `time_out` is never set; use `episode_length < T` for early endings. The current generator sets it.
6. Physics is soft: 10 ms constraint time constant, no-slip pass off. Object mass and friction are randomized per env
   and not recorded: a mass shift of 0-100 g (in-hand) or 0-200 g (screwdriver), friction x0.5-1.5 (in-palm rotation),
   x0.3-2.0 (repose) or x0.2-1.2 (screwdriver). Genesis adds the mass shift as a point mass at the centre of mass,
   leaving the inertia at the object's own. At the default density (rho 200) thin-walled meshes weigh under a gram
   (v2 objaverse: 0.15-16 g, 36 of 100 train meshes under 2 g, against 6.9-18.2 g for the boxes), so they keep a
   small rotational inertia.
7. Scene composition changes the sharpa teacher. On the same 83 objaverse objects, repose-sharpa dropped 39% of
   episodes in a 140-shape scene against 15% in 20-shape scenes, while the xhand1 teachers barely changed. A 170-shape
   scene crashed at build (CUDA assert in Genesis's contact-force sensor). v2 uses 20 shapes per scene. The cause was
   not found.

Details and what they mean for training: [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md), section 8.

## Not here

The overfit-fixture exporter (it builds a training batch, so it belongs with the data adapters), the rollout viewer and
one-off analyses stay in `/data/group_data/katefgroup/robertre_dexhand/spike/`. The five teachers run on the vendored
Eden; the `s1_play_rotate` task in `s1_play_related_overlay.zip` needs a newer Eden and is not covered.
