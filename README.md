# guided_playback — guided streaming rollouts + simulator playback, all domains

Standalone package (imports `pwm_final_paper` and `pwm_final_clean` read-only). One guidance core, one page.

```
gp/adapters/paper.py        pwm_final_paper checkpoints (pickup, rl_push, mould): stream staircase + full churn (paper_core = pickplace_server/stream_core.py)
gp/adapters/clean_push.py   pwm_final_clean push0418 (block pushing): streaming_chain staircase + churn bank (clean_core = churn_stream_lib.py)
gp/objectives.py            centroid-to-target + wall box clearance + floor
gp/scene_edit.py            close fingers, hand start mid/natural (pickup)
gp/guide.py                 optimizer loop: adam | unitgrad, t_freeze, optional shell re-projection, best (member, iteration)
gp/session.py               one run from a namespace of knobs (CLI and server share it)
gp/replay/{isaac_float,genesis_mould,sapien_push}.py   rollout.npz -> replay_<sim>/{replay.npz, analysis.json}
gp/viewer/build.py          the page: domain tabs -> runs (newest first) -> member tabs; layers guided / unguided / simulated
bin/guide.py  bin/serve.py  bin/client.py  bin/replay.py  bin/build_page.py
runs/<domain>/<name>/       rollout.npz + result.json (+ replay_<sim>/)
```

## Commands (env pwm211, one GPU)

```
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; cd /data/user_data/mbronars/packages/guided_playback
# pick & place / pushing on the three-domain checkpoint
$P bin/guide.py --adapter paper --ckpt /data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_three_ctxgen_100m_uncond_unet_bidir/checkpoints/step_160000.pt \
   --domain pickup --task push --episodes 10,82,1,26 --transitions 60 --steps 10 --churn --t_freeze 0 --fix_fps --joint_limits 13.5,53.5 \
   --close_fingers --hand_start natural --offset 0.15,0,0 --iterations 10 --lr 0.2 --lr_final 0.02 --name s160k_push_natural --desc "..."
# moulding (step_60000 = latest 20k multiple)
$P bin/guide.py --adapter paper --ckpt /data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt \
   --domain mould --task move --episodes 0,1,2,3 --transitions 150 --steps 10 --churn --t_freeze 0.8 --fix_fps --optimizer unitgrad --shell --offset 0.1,0,0 --iterations 10 --lr 0.05 --lr_final 0.005 --name ...
# block pushing (push0418 cont200k_stream final.pt)
$P bin/guide.py --adapter clean_push --ckpt /data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt \
   --task push --episodes 0,0,0,0 --transitions 60 --steps 8 --churn --t_freeze 0.8 --optimizer unitgrad --shell --offset 0.1,0,0 --iterations 10 --lr 0.2 --lr_final 0.02 --name ...
# replay in the domain's simulator (isaac | genesis | sapien picked from result.json), then the page
$P bin/replay.py --run runs/pickup/s160k_push_natural
$P bin/build_page.py            # -> site/index.html (+ site/data/*.js), publish as an artifact with the data files
# server (same knobs as JSON): bin/serve.py --adapter paper --ckpt ... --port 8766 --warmup ; bin/client.py http://<node>:8766/guide '{...}'
```

## Contracts
- `rollout.npz`: predicted / unguided [B, T, N, 3] (m, model frame), roles [B, N] (0 actuator, 1 finger, 2 object, -1 pad), object_mask, target [B, 3], offsets, boxes [nb, 6] (centre, size), trace_mm, episodes, step, particle_body, pose_action_mask, joint_type, body_type, object_ids (0 object, 1 actuator), body_poses0 [B, K, 4, 4], gt, gt_frames.
- `replay_<sim>/replay.npz`: sim [V, B, T, N, 3] at the model resolution (rigid = frame-0 generated particles under the simulated pose; MPM = simulated particles), sim_roles; `analysis.json` rows: gen_to_target_mm, sim_to_target_mm, sim_vs_gen_final_mm, gen_moved_mm, sim_moved_mm, gen_lift_mm, sim_lift_mm, track_mm_mean/max.

## Simulators
- Isaac (pickup, rl_push): `pickup_replay_isaac/scripts/replay_isaac.py`, env isaaclab, floating Franka hand, 20 fps; boot takes minutes at node load > 15.
- Genesis (mould): `packages/Genesis/examples/replay_planner_output.py` run from packages/Genesis in the genesis env (local Genesis 0.4.6); MPM bounds +-0.10 m xy.
- SAPIEN (block pushing): `PointWorld/replay_actions.py --use_set_pose` in the planning_wrapper env; cube placed at the boundary centre with the recorded yaw, env-default block physics; physics only (no Vulkan rendering).

## Rendered frames (style 2) and the Franka arm (style 3)

```
R=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python      # pyrender over EGL
PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 $R bin/render.py --run runs/<domain>/<name>          # gripper only
PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 $R bin/render.py --run runs/<domain>/<name> --arm    # gripper on the Franka (IK)
```
One-row gif per member: left = generated rollout, right = simulator replay, same camera, the pickup kitchen table.
Block pushing and moulding are drawn magnified (`--scale`, default 2 and 3); the robot stays life-size and holds the rod / paddle.

## Samplers

- `--sampler stream` (default): the diffusion-forcing staircase (window_states g=0, keep 3), frame 0 given.
- `--sampler chunked`: 19-row windows, 3 clean context rows + 16 future rows denoised together; the first context = the
  episode's recorded first 3 transitions; next window's context = the last 3 generated rows.
- `--churn`: full churn (eta = 1) with a pre-decided draw bank, optimized above `--t_freeze`; without it only the initial
  pose noise is optimized (deterministic sampler).

## Sweeps

`bin/sweep.py --arms sweeps/<file>.json --servers http://host:8766,http://host:8767 --gpus 0,1` runs every arm on the
servers in parallel, replays each run in its simulator, and writes `<file>.summary.{json,md}` (bin/summarize.py).

Pickup results so far (step_180000 three-domain checkpoint, episode 82 = sideways carton, population 4, Isaac replay):
the working recipe is streaming 60 transitions, NO churn, unitgrad, shell off, lr 0.2 -> 0.02 cosine, 10 iterations
(~150 s on a warm server). Lift 15 cm: 4/4 held, sim-to-goal 18-81 mm (top-ranked member 18 mm; 3 mm with seed 1);
place +20 cm on the table: 3/4 within 50 mm; lift over a 10 cm wall: 2 crossings in ~40 members. Details in
`sweeps/sw1..sw4_*.summary.md`.
