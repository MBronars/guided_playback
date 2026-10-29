# Mould centroid guidance on the paper's cloud_centroid scenarios (seed 42)

Run 2026-09-23 from `/data/user_data/mbronars/packages/guided_playback`. Everything below is reproducible from the
paths given; no number in this file was computed by hand.

## Model

| | |
|---|---|
| checkpoint | `/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt` |
| training config | same directory, `config.yaml` |
| parameters | 104.3 M |
| architecture | point-cloud UNet, widths 192/384/768, strides 4,4, enc depths 1,2,4, dec depths 1,2, head dim 64, pointnet pool/unpool, `unet_dual_ffn`, **bidirectional** temporal attention |
| RoPE | `unet_geo_r_m = 0.133155`, `unet_rope_f_min = 0.15` |
| conditioning | **unconditional**: `action_dropout = 1.0`, `unet_actions = false`; generated context (`context_len = 4`, `unet_deformable_context = true`) |
| initialised from | `/data/group_data/katefgroup/runs_pwm_mbronars/init_snapshots/mould_ctxgen_actcond_step24000.pt` (the action-conditioned ctxgen snapshot) |
| step used | 60 000 |
| runtime | compiled regional + L0, bf16, tf32 matmul |

## Scenarios

`/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42/ep_000.npz` … `ep_049.npz`,
the same 50 the previous paper run `cloud_centroid_diff_paper_seed42` used. Each pins `batch_idx` (the dataset
episode) and `target_offset_xy` (magnitude 7.5–15 cm, random in-plane angle), sampled once with
`np.random.default_rng(42)`. They were **not** regenerated or re-tuned for this run.

Per scenario the target is `centroid(dough, frame 0) + (offset_x, offset_y, 0)`. The frame-0 distance to target
is therefore exactly `|offset|`, which is what makes "distance closed" well defined.

## Guidance settings

Population of **3 members of the same episode and the same target** — the batch is the population, not a set of
tasks. 150 transitions, streaming staircase with `keep = 3` committed rows, `S = 6` sampler steps, 62 model calls
per rollout.

* optimizer `unitgrad`, **lr 0.1 constant** (no `lr_final`, so no decay), **6 iterations**
* step: `x <- shell(x - lr * sqrt(D) * g/|g|)` — [`gp/guide.py:52`](../gp/guide.py), per-draw normalised gradient
* **shell re-projection on**: after every step each draw is rescaled to the norm it was sampled with
  ([`gp/guide.py:54-58`](../gp/guide.py))
* `t_freeze = 0.8`: only bank draws at noise level ≥ 0.8 are optimised (1–3 of the 7 draws per row)
* **full churn, eta = 1, on both pose and particle rows** (`particle_churn: true`) — every re-formed row is
  `x0 + l*(E - x0)` with a fresh draw from the bank, not a deterministic Euler carry
* `fix_fps`: per-member generators for `sample_farthest_points`, so members differ only through their draws
* objective = `||centroid(dough, last frame) - target||`; no box or floor terms for this task
  ([`gp/objectives.py`](../gp/objectives.py))

There is **no separate guidance weight**. The strength is entirely the step size: 6 steps of 10 % of each draw's
own norm, on the top noise levels, with the norm restored after each step.

## Procedure

1. Warm server: `bin/serve.py --adapter paper --ckpt <above> --port 8790 --warmup --warmup_domain mould`, GPU 1.
   It had already compiled the B=3 / N=150 graphs before the first scenario, so **every timing below is warm**.
2. `bin/paper_eval.py --url http://babel-y9-12:8790/guide --seed 42 --n 50 --members 3 --steps 6 --iterations 6
   --lr 0.1 --t_freeze 0.8 --replay_gpu 0`
3. Per scenario: guide the population of 3, pick `argmin(best_mm)` — the member whose **imagined** objective is
   best, which is what can be done at deployment without a simulator — and replay only that member in Genesis.
   Replays run on GPU 0 in the background while the next scenario is guided.
4. Genesis replay ([`gp/replay/genesis_mould.py`](../gp/replay/genesis_mould.py)): Genesis 0.4.6 MPM, the generated
   paddle rows Kabsch-fit to a paddle pose per frame, frame-0 generated dough particles injected, paddle PD-tracked
   with **25 sim steps per model frame**, simulated dough read back at every model frame.

## Metrics ([`gp/replay/common.py:45`](../gp/replay/common.py))

All on the **dough centroid**, in mm.

* `sim_to_target_mm` — simulated dough centroid at the last frame vs the target
* **distance closed** = `target_mm - sim_to_target_mm` (the headline; `bin/paper_report.py` computes it)
* `sim_moved_mm` / `gen_moved_mm` — last-frame vs frame-0 centroid, simulated / generated
* `sim_vs_gen_final_mm` — how far the simulator ended from what the model imagined

## Results, all 50 scenarios

```
DISTANCE CLOSED (sim)      mean 26.8 mm   median 26.0   min 11.3   max 41.9   negative 0/50
fraction of target closed  mean 24.9 %    median 25.9 %
  p10 15.8   p25 21.3   p75 33.2   p90 38.3 mm
  >=20 mm closed: 40/50      >=30 mm closed: 15/50
distance closed (imagined) mean 35.1 mm   (unguided: 3.0 mm)
target distance            mean 109.6 mm
sim centroid to target     mean 82.8 mm   median 81.6   min 44.3   max 125.4
sim dough motion           mean 2.86 cm   std 0.80   (generated 3.66 +- 1.21 cm)
sim vs generated gap       mean  9.0 mm   max 46.8
guide seconds (warm)       mean 113.8 s   median 111.9   min 106.1   max 176.5   over 160 s: 1/50
replay seconds             mean 152.0 s
```

Worst three by distance closed: scenarios 24 (11.3 mm), 49 (12.3), 31 (13.4). Best three: 18 (41.9), 2 (41.5),
45 (39.8). The single 176.5 s scenario (index 2) was guided while another session's server shared GPU 1.

## Artefacts

* summary with per-scenario numbers and timings:
  `/data/user_data/mbronars/packages/guided_playback/runs/_paper_eval/paper42_summary.json`
* aggregate script: `/data/user_data/mbronars/packages/guided_playback/bin/paper_report.py`
* live log: `/data/user_data/mbronars/packages/guided_playback/paper_eval.log`
* per-scenario clouds, guided rollout and Genesis replay:
  `/data/user_data/mbronars/packages/guided_playback/runs/mould/paper42_ep000` … `paper42_ep049`
  (`result.json` records every setting; `rollout.npz` the clouds; `replay_genesis/analysis.json` the sim numbers)

## One correctness note

Before 2026-09-23 this package carried the **particle** rows deterministically under full churn and only re-formed
the pose rows — correct for rigid scenes, wrong for dough. With that bug the same guidance produced
`sim_vs_gen_final_mm` of 83–98 mm; after the `churn_part` fix in
[`gp/adapters/paper_core.py`](../gp/adapters/paper_core.py) and [`gp/adapters/paper.py`](../gp/adapters/paper.py)
it is 0.8–16.7 mm on the same episodes. Every number in this file is post-fix.
