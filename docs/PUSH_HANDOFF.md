# Pushing experiments: handoff

For the agent taking over the pushing guidance experiments. Written 2026-09-24. Everything you need is in the
`guided_playback` package: /data/user_data/mbronars/packages/guided_playback. Read its README.md first.
Then follow this doc, which describes the protocol the pickup benchmark used. Pushing should use the same protocol.

## 1. How to work with this user (read this first)

These rules come from repeated, explicit corrections. Breaking them is the fastest way to lose the user's trust.

- **Visuals first.** Publish a page within minutes of the first result. Republish it as runs land. A number with no
  3D viewer to look at is not a result to this user.
- **One case first.** For any new idea, run ONE episode or target first, publish it, and wait for the user.
  Never fan out over every asset, target or setting before the one case has been looked at. Never stack queues.
- **Facts, not analysis.** Report measurements: distances in mm, times in s, counts. No mechanism stories, no
  speculation, no proposals unless the user asks. If you have not checked why something happened, say so.
- **Absolute paths** for every file and folder you mention.
- **Never delete or overwrite results.** A new config goes to a new run name. `rm -rf` of a run or render
  folder is forbidden.
- **Do not touch other people's processes or jobs.** Never kill a process you did not start. Never scancel.
  When another person is using a GPU, use only the free one.
- **No new git worktrees or branches.** Build inside guided_playback.
- **Post updates** as each run finishes. Use short lines with the measured numbers.
- **Launch when told.** When the user says run, run. Do not add extra smoke tests.

## 2. Two push setups exist: confirm with the user which one

| | block pushing ("pushing") | rl_push |
|---|---|---|
| checkpoint | /data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt | three-domain paper checkpoint, e.g. /data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_three_ctxgen_100m_uncond_unet_bidir/checkpoints/step_160000.pt |
| adapter | `clean_push` (/data/user_data/mbronars/packages/guided_playback/gp/adapters/clean_push.py) | `paper`, `--domain rl_push` (/data/user_data/mbronars/packages/guided_playback/gp/adapters/paper.py) |
| scene | 128-point cube + 128-point rod, 04_18 data, centred per episode | Franka hand pushing objects (rl_push data) |
| simulator | SAPIEN PushBoundary, physics only (/data/user_data/mbronars/packages/guided_playback/gp/replay/sapien_push.py) | Isaac floating hand (/data/user_data/mbronars/packages/guided_playback/gp/replay/isaac_float.py) |
| episodes | test windows in loader order; repeats = population | select json / loader indices |

The existing block-push result is /data/user_data/mbronars/packages/guided_playback/runs/pushing/push0418_push_10cm:
- Settings: target +10 cm in x, 60 transitions, S=8, churn, t_freeze 0.8, unitgrad, shell on, lr 0.2 → 0.02, 10 iterations,
  4 test windows.
- Generated distance to goal 29 / 20 / 27 / 3 mm. SAPIEN distance to goal 57 / 75 / 87 / 3 mm.
- Timing: warm iterations took 6–19 s. The first grad iteration took 742 s and the final no-grad pass 574 s,
  because both are compiles. Serve the checkpoint warm; do not re-launch it per run.

## 3. The benchmark protocol (copy this from pickup)

The pickup benchmark is the template. Its files:
- /data/user_data/mbronars/packages/guided_playback/sweeps/pick_benchmark_5x5.json is the spec baselines reuse.
  It holds the dataset, episode selection, start centroids, offsets, absolute targets, seeds and settings.
- /data/user_data/mbronars/packages/guided_playback/sweeps/cyl5x5_queue.sh is the guided queue, with per-run timing files.
- /data/user_data/mbronars/packages/guided_playback/sweeps/planner_eval.py is the planner-view table.
- /data/user_data/mbronars/packages/guided_playback/sweeps/mppi5x5_queue.sh and
  /data/user_data/mbronars/packages/guided_playback/sweeps/mppi_eval.py are the no-guidance baseline.

The protocol:
1. **Task.** Fixed targets drawn once with a seeded RNG and saved to a file. For pickup that was
   /data/user_data/mbronars/packages/guided_playback/sweeps/cyl_targets.txt: start + (dx, dy, +15 cm), with (dx, dy)
   uniform in a 15 cm disc (numpy default_rng(0), r = 0.15·sqrt(U), angle = 2πU). The same targets are used for
   every episode. For pushing, z stays 0 (`task push` zeroes it). **The target radius and distance for pushing are
   the user's call. Ask.**
2. **Guided planner.** Population 4 (the batch, all the same episode, seed 0). 10 iterations of Adam on the initial
   noise, shell projection on. For pickup the settings were lr 0.1, 60 transitions, S=10.
3. **Planner view.** The plan is the member with the lowest GENERATED distance to the goal. Report that member's
   simulator result only. Do not report the best member in sim: the planner cannot know sim results in advance.
4. **Timing.** The warm generation time is the sum of `iteration_seconds` in result.json. Report it as mean ± std
   over the runs. Replay time is reported separately.
5. **No-guidance baseline.** This is the "MPPI argmin" planner (see section 5) with K fresh samples per replan and a
   commit of 5 frames, every model call on the full 19-row window (`full_window: true`; the window slides back over
   committed rows near the end instead of shrinking). K is set so a plan takes ~150 s: pickup K = 20 -> 138 s.
6. **Save everything.** Each run folder holds rollout.npz (guided and unguided trajectories for every member) and
   result.json (settings, seeds, per-iteration times and distances). Simulator runs go under replay_<sim>/, which holds
   replay.npz and analysis.json. The page and the numbers must be rebuildable from disk.

## 4. Commands

```
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
# server: one checkpoint, compiled, kept running; compile the no-grad path only with --warmup_iterations 0
CUDA_VISIBLE_DEVICES=<gpu> nohup $P -u bin/serve.py --adapter clean_push --ckpt <ckpt> --port <port> --warmup \
    --warmup_transitions 60 --warmup_episodes 0,0,0,0 > logs/serve_push.log 2>&1 &
# one guided run (JSON = the same knobs as bin/guide.py)
$P bin/client.py http://localhost:<port>/guide '{"domain":"pushing","task":"push","episodes":"0,0,0,0","transitions":60,"steps":8,
    "offset":"0.1,0,0","optimizer":"adam","lr":0.1,"iterations":10,"shell":true,"seed":0,"name":"<new name>","desc":"..."}'
# no-guidance planner (section 5)
$P bin/client.py http://localhost:<port>/plan '{... same scene knobs ..., "K":32, "commit":5, "name":"<new name>"}'
# simulator replay (the backend is picked from result.json)
CUDA_VISIBLE_DEVICES=<gpu> $P -u bin/replay.py --run runs/pushing/<name>
# page: symlink the runs into a folder, build, then publish site_x/index.html with every site_x/data/*.js as files
GP_PAGE_ONLY_BEST=1 GP_PAGE_GUIDED_SIM_ONLY=1 GP_PAGE_LABEL=method $P bin/build_page.py --runs site_x_runs --out site_x --title "..." --subtitle "..."
```

Page knobs in /data/user_data/mbronars/packages/guided_playback/gp/viewer/build.py:
- `GP_PAGE_ONLY_BEST=1` shows only the planner's chosen member.
- `GP_PAGE_GUIDED_SIM_ONLY=1` drops the unguided sim layer.
- `GP_PAGE_LABEL=method|cyl|asset|target` sets the run grouping. Keep the run list short and clearly labelled.
  The user has complained about too many unclear options in the left panel.

The member tabs mark the best-in-generation member ◆ blue and the best-in-sim member ● amber. A member that is best in
both is marked ✓ green. Artifacts are capped at 64 MB per version, and each case costs about 1–3 MB.

## 5. The no-guidance planner (MPPI argmin)

The code is /data/user_data/mbronars/packages/guided_playback/gp/mppi_argmin.py, served at POST `/plan`. The server
reloads the module on every request, so edits need no restart. Each replan does the following:
- Draw K fresh noises for the remaining transitions.
- Sample K futures that continue from the committed frames. `_chain_from_prefix` is pwm_final_clean `streaming_chain`
  with the committed transitions as clean level-0 rows.
- Keep the candidate whose final object centroid is closest to the target. There is no averaging.
- Commit its next 5 frames and repeat.

Candidates run in chunks of the batch size (4), so the compiled graphs are reused. With no frames committed, the output
matches streaming_chain exactly, provided the global RNG is seeded the same way (the UNet's FPS draws come from it).
Test: /data/user_data/mbronars/packages/guided_playback/scratch/test_prefix_chain.py.

It has only been run with the `clean_pickup` adapter. Two things to fix for pushing:
- It reads `ad.bf16`, which `clean_push` does not define.
- It assumes a deterministic chain with no churn.

Rerun the parity test on the push checkpoint before trusting it.

Pickup reference (pickup-only checkpoint, 5 assets × 5 targets): guided, Isaac median 38 mm, 21/25 within 60 mm,
126.5 ± 0.8 s. MPPI argmin, full windows, K = 20: Isaac median 121 mm, 4/25 within 60 mm, 138.1 ± 0.6 s
(/data/user_data/mbronars/packages/guided_playback/sweeps/mppi_full19_eval.md). For paper-adapter checkpoints (mould,
rl_push) the same planner is /data/user_data/mbronars/packages/guided_playback/gp/mppi_argmin_paper.py (the server
dispatches /plan by adapter); its sampler already uses full windows.

## 6. GPUs and gotchas

- Node babel-o9-24, job 10545577, 2 × L40S. GPU 0 runs the pickup server (port 8768). Do not stop it.
  GPU 1 is doing Isaac replays for the pickup MPPI queue until that finishes. **Ask the user which GPU you may use**
  before starting a push server.
- Compiled servers (dynamic=False) recompile on a new batch size. Keep B = 4. The compile cache on this node is warm,
  so a pickup restart took 36 s.
- One server holds one checkpoint and handles one request at a time. A request cannot be cancelled once it is sent.
- `pgrep -f` matches your own shell. Kill only by explicit PID, and only processes you started.
- `bin/client.py` sends POST only to `/guide`, `/generate` and `/plan`.
- Keep page sources and renders on /data. /tmp is node-local and the shell can move nodes.
- The run name must be new for each config. Never reuse a run folder.
