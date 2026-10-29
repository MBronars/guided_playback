# PGND baseline — standalone inference

Run a trained PGND checkpoint on `mpm_drag` episodes from this repository, with **torch only**:
no warp, no dgl, no PGND checkout. The pwm211 environment runs it as it is.

```bash
python baselines/pgnd/rollout.py --out /tmp/pgnd_rollouts --n 6
```

By default this loads the mpm_drag run's **latest** checkpoint and rolls it out on that release's
test split. Pick another checkpoint with `--step best | final | 40000`, another run with
`--task three | dex`, or give `--ckpt` a path.

Each episode writes an npz with `pred` and `gt` points in world metres, the gripper track, and
`err_mm`, the per-frame mean position error. The run prints the same numbers.

## What is in here

| File | |
|---|---|
| `network.py` | PGND's network, vendored unchanged from the PGND repository (MIT). The checkpoint's `state_dict` refers to these module names, so do not rename anything. |
| `sim_torch.py` | PGND's simulation step rewritten in torch: wall friction and the gripper sphere on the grid, then quadratic B-spline interpolation back to the points. The training code runs this as warp kernels. |
| `rollout.py` | Scene placement, the gripper track, and the rollout loop. `rollout_points()` takes points and a gripper track and returns predicted points, for data other than `mpm_drag`. |

## How the baseline was trained

`mpm_drag` has no robot: one patch of 103 material points is driven by a commanded velocity.
The PGND baseline represents that patch as PGND's own gripper sphere, centred on the patch and
moving with the patch's **recorded** motion, closed while the drag is active. `rollout.py` builds
the same sphere from the episode, so inference matches training. Note that this gives the model
the patch's achieved motion, where PWM models are conditioned on the command instead.

Runs live in `/data/group_data/katefgroup/runs_gns_mbronars/pgnd_<task>/`, beside the GNS
baselines: `train/ckpt/{latest,best,final}.pt` plus `model-<step>.pt` every 10,000 steps, with the
run's config at `train/hydra.yaml`. A frozen copy of the step-10,000 checkpoint also sits in
`/data/group_data/katefgroup/VLA/3DPCWM/checkpoints/pgnd_mpm_drag/`.
Training itself lives outside this repository, in the PGND checkout (`packages/pgnd`,
`experiments/scripts/train_drag.sh`), which needs warp.

## Checked against the training code

Same checkpoint, same episodes, torch here versus warp in the PGND checkout:

- one step from a real episode state, gripper open and closed: agreement to 2e-6 m;
- whole-episode rollouts (77–109 steps): the two runs stay within 0.1–1.6 mm of each other, on
  errors of 20–200 mm, and report the same mean error to the printed precision.

Reproduce with `experiments/scripts/rollout_drag.py` in the PGND checkout (warp) and this script
on the same episodes.
