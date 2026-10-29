# Distributed planning evaluation (work pool + central Isaac replay)

How the pick noise-vs-action experiment (Oct 2026) was planned on many preemptible GPUs and scored in Isaac.
Everything here is in `guided_playback`; results pages: `pwm_final_paper/results/noise_vs_action_status`.

## Idea

- One **unit** = one plan: one planner setting on one problem (about 1 s to 4 min of planning).
- Many short-lived **workers** (preemptible L40S jobs) load the model once and claim units until none are left.
  A preempted worker loses at most the plan in flight; nothing is ever done twice.
- Workers never run a simulator. One long-lived **replayer** stacks finished plans and replays them in Isaac in
  batches (one Isaac launch per batch), so Isaac boots rarely and never on preempt nodes.
- Planning time recorded per plan = full-loop request time (batch construction, noise, every rollout / gradient step /
  CEM round, scoring), excluding model load, compilation and writing results.

## Pool layout (`gp/pool.py`)

```
<pool>/units/<id>.json       {"id", "shape", "tag", "problem", "req": guided_run knobs}
<pool>/claims/<id>/hb        claim (atomic mkdir) + heartbeat "<worker> <unix time>", refreshed every 30 s
<pool>/plans/<id>/           the plan (rollout.npz, result.json)
<pool>/done/<id>.json        imagined distance, request_seconds, picked member, plan folder, GPU   (atomic write)
<pool>/replayed/<id>.json    Isaac result for that plan                                            (atomic write)
<pool>/failed/<id>.json      traceback; failed units are not retried automatically
<pool>/stale/                claims taken over after their heartbeat went quiet
<pool>/bad_workers/          workers that stopped themselves after a non-finite rollout
```

- **Claiming:** `os.mkdir(claims/<id>)` is atomic, so exactly one worker wins. A claim whose heartbeat is older than
  the lease (180 s, `GP_POOL_LEASE`) and has no `done/` file is stale: the next worker renames it into `stale/` (only one
  rename can succeed) and claims the unit again.
- **Results** are written to a temporary file and renamed, so a unit is either done or not.
- **Non-finite guard:** if any rollout score is NaN or inf, the worker writes `bad_workers/<worker>.json`, gives the unit
  back and exits (a few processes went non-finite mid-run on 23 Oct; cause not found).
- **Shapes:** each unit names its batch shape (e.g. `B4_nograd`, `B4_grad`); workers prefer shapes they have already warmed.

Commands:
```
python -m gp.pool work   <pool> --shapes all [--cache <compile cache>] [--max_minutes M] [--linger M]
python -m gp.pool replay <pool> --gpu_srun "<srun prefix>" [--every 120] [--once]
python -m gp.pool status <pool>
```
`--linger M`: keep waiting M minutes for new units after the last one (closed loop, where rounds arrive one at a time).

## Jobs (`sweeps/noise_vs_action/`)

| File | What |
|---|---|
| `env.sh` | shared environment: code tree (`GP_PAPER_REPO`), pick scenes, speed options (`PWM_MODEL_OVERRIDES`), Isaac physics (`GP_ISAAC_EXTRA`, 1/480 s), compile cache path |
| `make_units.py <pool> <settings.json>` | settings x the 25 pick problems -> units |
| `settings_main.json`, `settings_mppi.json` | the planner ladders (DNO, CEM over noise / actions, best-of-K, MPPI over noise) |
| `worker.sbatch` | one worker: `preempt`, `--qos=preempt_qos`, `gpu:L40S:1`, 4 h, `--requeue`; submit as an array: `sbatch --array=0-15 --export=ALL,POOL=<pool> worker.sbatch` |
| `worker_cl.sbatch` | the same with `--linger 30` (closed loop) |
| `replayer.sbatch` | the replay loop on `katef-cpu` (`--qos=katef_cpu_qos`, 3 days); Isaac launches go into the running `claude-host` GPU job via cross-job `srun --overlap` (the job's own `SLURM_*` variables must be unset first) |
| `build_compile_cache.py` | warms every planning shape in one process and saves `torch.compiler.save_cache_artifacts()`; workers load it at start-up (about 79 s instead of 255 s to be ready). GPU-specific: build on L40S |
| `cl_drivers.sbatch`, `cl_pw.sbatch` | closed-loop drivers (below) |

Rules used: every plan whose time goes on an axis runs on one L40S; workers are L40S only. The local `claude-host` GPU is
for building, smoke tests and the Isaac replays (Isaac results were bit-identical on L40S and A6000).

## Planners (`gp/cem.py`, hooked into `gp/session.py` by the `planner` knob)

- `planner: cem` / `mppi` / `bestofk`: sampling over the same churn noise DNO optimizes; proposals projected to their
  expected length; one Gaussian per plan; elites max(2, K/8); refit mean rolled out and scored; the best scored plan is executed.
- `planner: cem_actions`: CEM over the hand's pose deltas with the model in action-conditioned mode
  (`paper_core.Streamer.given_actions`).
- No `planner` knob: DNO guidance as before (`gp/guide.py`).

## PointWorld

`reports/pw_mpc/pool_worker_pw.py` is a PointWorld worker with the same claims, heartbeats and done records, writing plans
in the same run-folder format, so the same replayer replays them. Launch: `worker_pw.sbatch`.

## Closed loop

`bin/closed_loop_pick_pool.py --pool <pool> --tag <setting>`: per round, one unit per problem carrying the executed
history (last 3 executed transitions as clean context, `gp/adapters/paper_prefix.py`); when all 25 are planned, append each
problem's next 5 frames, replay all committed frames from the start in one Isaac launch, read back the executed clouds,
build the next history (`gp/closed_loop_hist.py`). 12 rounds; the last Isaac replay is the score. Concurrent drivers
share the Isaac GPU through `flock`. PointWorld: `bin/pick_pw_cem_closed_loop.py` (plans its 25 problems in series).

## Where the runs are

`/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action/`: `pool_main` (our planners), `pool_pw` (PointWorld),
`pool_cl` (closed loop, `closed_loop/<tag>/`). Development and smoke pools: `../noise_vs_action_dev/`.

## Pile: PointWorld + CEM (24 Oct)

- Checkpoint `/data/group_data/katefgroup/runs_pw_mbronars/pointworld_mould_medium/latest.pt` (99M "medium" PTv3, domain
  `pickup_mould`, 16-frame horizon, stats `.../mould_stream/norm_stats`). The vendored `pwm_final_paper/baselines/pointworld/ptv3/ptv3_arch.yaml`
  needed the `medium` entry from the PointWorld repo.
- Planner `/data/user_data/mbronars/reports/pw_mpc/pile_cem.py`: one open-loop 150-frame CEM plan over paddle `[dx, dy, dz, dyaw]`,
  same CEM rules as the pick reference; start states = frame 0 of `runs/mould/paper42_epNNN`; cost = last-frame dough xy centroid to target.
- Worker `/data/user_data/mbronars/reports/pw_mpc/pool_worker_pw_pile.py plan|replay <pool>`: plan = timed L40S; replay = Genesis
  with +-70 cm bounds (`examples/replay_with_big_bounds.py`), own claims under `<pool>/rp/`, writes `replayed/<id>.json` with progress.
- Units / jobs: `sweeps/pile_pw/{make_units.py, plan_worker.sbatch, replay_worker.sbatch}`; pool `noise_vs_action/pool_pw_pile`.
- Page: `pwm_final_paper/results/pile_pw_status` (`update.sh`).
