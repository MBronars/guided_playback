# Letters I, C, L, R on the new mould checkpoint: scope

Written 2026-09-25. Nothing has been run. This is what has to be built, what the sweep looks like, and what it costs.

## 1. What the old letter results were

- Task: shape the 1024-particle dough blob of test clip 0 of `deformable_mould_new_6k_wds/test` into a letter footprint
  with the paddle, 150 frames (some arms 200), one plan per (letter, seed).
- Target: `pwm_efficient/letter_template.py::fit_letter_field` fits an EMNIST prototype (scale, translation; rotation
  fixed per letter at `default_rng(SeedSequence([42, ord(L)])).uniform(-40, 40)`) to the frame-0 blob by IoU, letter area
  = blob area x `letter_density_scale 0.75`. Output `letter_field.npz`: `coverage [768, 2]` interior points,
  `contour [240, 3]`, `chosen_rot_deg`, fit info.
- Objective (planner): symmetric mean-squared chamfer in the plane between dough particles and the coverage points
  (cloud->template + template->cloud), averaged over every frame of the plan.
- Metric (paper): the same chamfer on the LAST frame of the Genesis MPM replay, reported as RMS mm
  (`sqrt(sim_chamfer_m2) * 1000`), `pwm_efficient/sweep/replay_and_score.py`.
- Numbers: alphabet mean 6.95 +- 2.71 mm (cfg47, 26 letters, ~1340 s/letter); best config `d_no20_pop5_gw2` 6.80 +- 2.45 mm
  on R+L (~247 s); the paper's four showcase letters were the best seed of 50 per letter: C 6.40, O 2.90, R 3.31, L 4.58 mm
  (`paper_final/best_letters`). The old planner replanned every 5 frames (30 replans of a 5-frame commit) and its knobs
  were nopt / pop / gw / t_stop / rows / substeps. Those sweeps are not being repeated.
- Verified today: the new loader's mould episode 0 (`runs/mould/paper42_ep000/rollout.npz` gt frame 0) is the same clip
  the old letter runs started from (1024 dough points, nearest-neighbour distance 0.0 mm, same paddle pose).

## 2. What exists in this package for mould

- `paper` adapter, checkpoint `pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt` (steps 80k and
  100k also exist; 60k is the one every mould number so far used).
- Whole-horizon initial-noise optimization: stream staircase keep 3, S sampler steps, full churn on pose and particle
  rows, unitgrad or Adam, shell re-projection, `t_freeze`, population = batch. Objective today = centroid-to-target only.
- Genesis replay `gp/replay/genesis_mould.py` (25 substeps per frame, +-0.10 m bounds), analysis rows in mm.
- Measured on the centroid task, population 3, 150 transitions, S=6: 17 s per iteration, 8.3 GiB peak; S=10 population 4:
  36 s per iteration, 10.7 GiB. Genesis replay 88-150 s per member.
- Prior knob sweep on the centroid task (`runs/mould/mould60k_*`, episode 0, 10 cm target): lr 0.1 flat beat 0.05 and
  the decaying schedule; t_freeze 0.5 optimizes the imagined objective far better (25-49 mm vs 66-89) but the sim result
  did not follow (85-98 mm), so t_freeze 0.8 was kept; shell on beat shell off; 8 iterations beat 6.

## 3. What has to be built (all inside guided_playback, about half a day)

1. `gp/letter.py`: import `fit_letter_field` and the chamfer from `pwm_efficient/letter_template.py` (numpy, scipy,
   torch only; the letter bank is `pwm_efficient/stats/letter_bank.npz`). `LetterObjective(field, dough_mask)`:
   chamfer of the dough rows to the coverage points. Knobs: `letter_frames` (last k frames averaged; k = 1 is the paper
   metric, the old planner used all frames), `coverage_weight` (1.0), `floor_weight`. Save `letter_field.npz` into the
   run folder with the old keys so the old scoring and rendering scripts read it unchanged.
2. `gp/session.py`: `--task letter --letter R --letter_seed 42 --density_scale 0.75 --coverage_frac 0.75`; the target
   passed to `rollout.npz` becomes NaN (no centroid goal) and a new `letter_contour [240, 3]` key is written for the viewer.
3. `gp/guide.py`: `--save_iterates`: keep every iteration's rollout for every member and write `iterates.npz`
   (`positions [I+1, B, T, N, 3]` float16, `objective [I+1, B]`, `picked_iteration [B]`). At 150 x 1280 x 3 float16 that
   is 1.2 MB per member-iteration, about 140 MB for 3 members x 40 iterations. This is the "full plan" trace: iteration 0
   is the unguided draw, the last is the committed plan, so it is the analogue of the old per-iteration lookaheads and
   feeds the same refinement GIF (`Genesis/examples/render_lookahead_gif.py` reads `iter_*.npz`; a 20-line exporter
   writes that layout from `iterates.npz`).
4. `gp/replay/genesis_mould.py`: when `letter_field.npz` is present, add `sim_chamfer_mm` and `gen_chamfer_mm` (RMS, last
   frame, model resolution) to the analysis rows. The sim particles at the last model frame are the last substep, so
   this equals the old `sim_chamfer.json` number.
5. `bin/letter_eval.py`: the driver (one letter, one config, N seeds; picks the member by imagined chamfer, replays the
   pick and, when asked, every member) and `bin/letter_report.py` (per letter: imagined and sim RMS mm per seed, picked
   vs best-in-sim, plan seconds). Same shape as `bin/paper_eval.py` / `paper_report.py`.
6. Viewer: draw `letter_contour` as the target on the page (`gp/viewer/build.py`), so the first result can be looked at.

Not building: receding-horizon guided replanning. If whole-horizon plans cannot form a letter, `gp/mppi_argmin_paper.py`
already commits 5-frame prefixes and could wrap the guided optimizer; that is a second step after the first result.

## 4. Sweep plan (the new knobs only)

Fixed for every arm: episode 0, letter rotation seed 42, density scale 0.75, churn on, shell on, unitgrad, fix_fps, keep 3,
population = 3 members of the same letter (the batch), seed 0 unless the arm is a seed repeat.

Stage 0, one case (one GPU, about 20 min including compile): letter R, 150 transitions, S=6, lr 0.1 flat, 12 iterations,
t_freeze 0.8, `letter_frames` = all frames (the old objective). Publish the page. Stop and look.

Stage 1, knobs, letters R and C only, one config per arm (about 25 arms):

| knob | values | why |
|---|---|---|
| iterations | 12, 24, 48 | "let it plan longer": 17 s each, so 48 iterations is 14 min per run |
| lr | 0.05, 0.1, 0.2 (flat) and 0.2 -> 0.02 | the centroid sweep favoured 0.1 flat; a letter needs more shape change |
| t_freeze | 0.8, 0.65, 0.5 | 0.5 overfit the imagined centroid; a shape objective may behave differently |
| S | 6, 10 | 10 doubles the cost, gave cleaner dough in the earlier pickup/mould runs |
| transitions | 150, 200 | the old best R and L came from 200-frame plans |
| letter_frames | all, last 15, last 1 | which frames the chamfer is averaged over |

Expected cost: about 25 arms x 2 letters x ~8 min average = 7 GPU-hours of guidance plus 50 x 2 min of Genesis replay
(runs on a second GPU, or the same one after). Fits one L40S overnight or two L40S in an afternoon.

Stage 2, the ICLR set: the best stage-1 arm on I, C, L, R x 10 seeds (seed = the initial draw; the old showcase used 50
seeds per letter, but each of those plans was a 4-minute receding-horizon run, ours are whole-horizon). Replay every
member of every seed in Genesis so we report both the deployable number (member picked by imagined chamfer) and the
best-in-sim number, per letter. Cost: 4 letters x 10 seeds x ~14 min = 9.5 GPU-hours guidance, 120 replays x 2 min =
4 GPU-hours replay. Iterates are saved for every run (about 140 MB each, 5.6 GB total) and the non-top-3 per letter
can be pruned afterwards.

Stage 3, figures: refinement GIF (iteration 0 -> final, each replayed in Genesis) for the best seed per letter, and the
arm render (`bin/render.py --arm`).

## 5. GPU needs

- One L40S (or RTX PRO 6000) holding a warm compiled server for the mould checkpoint (population 3 uses 8.3 GiB, S=10
  population 4 uses 10.7 GiB, so a 46 GB card has room to run the Genesis replays alongside).
- A second GPU makes the replays free; without it the replay queue runs after the guidance queue on the same card.
- Today: interactive job 10545577 on babel-o9-24 holds 2 L40S (GPU 0 the pickup server at 9.5 GB, GPU 1 at 25 GB, in use).
  The preempt partition shows L40S and RTX PRO 6000 nodes in mixed state. Which card to use is the user's call.
- Environments: `pwm211` for guidance, `genesis` for replay, `foundationpose` for renders; all present.

## 6. Open questions for the user

1. Horizon: keep 150 frames (old default) or allow 200 (where the old best R and L came from)? The sweep includes both.
2. Population: 3 members (paper42 protocol) or 4-5? Memory allows 5 at S=6 on an L40S; time per iteration grows with B.
3. Report the picked-by-imagined member, the best-in-sim member, or both? The plan above computes both.
4. Is letter I acceptable with the EMNIST bank (a bar, easy to hit with a poor fit)? The fit picks the prototype by IoU
   with the round blob, so the I will be a wide bar unless density scale is lowered for it.

## 7. Stage 0 result (2026-09-25, GPU 1 of job 10560734, babel-l9-16)

Built: `gp/letter.py` (objective + metric), `sweeps/letters/fit_fields.py` (pctrl env) -> `sweeps/letters/fields/letter_{I,C,L,R,O}_seed42.npz`
(high-res bank, rotation rule of the canonical alphabet eval; coverage/contour match `alphabet_eval_cfg47/letter_X/letter_field.npz`
to 0.001 mm; my metric reproduces the old best-R paper number 3.31 mm from its replay.pkl), `--task letter` in `gp/session.py` /
`bin/guide.py`, `--save_iterates` in `gp/guide.py` (iterates.npz: positions [I+1, B, T, N, 3] fp16, objective [I+1, B]),
`sim_chamfer_mm` / `gen_chamfer_mm` rows in the Genesis replay, letter contour + chamfer series on the page, `GP_RELOAD=1` in
`bin/serve.py` (edits to the objective/loop reload per request, the compiled model stays).

Server: `logs/serve_letters_8795.log`, http://babel-l9-16:8795, mould step 60k, compiled B=3 / N=150. 9.5 s per iteration
(S=6, 3 members), 12.5 GiB peak.

Run `runs/mould/letters_R_stage0_s6_lr0.1_it12_tf0.8_all` (2 min guidance, 6 Genesis replays 11 min), page
https://claude.ai/artifact/CUKZhQDy4DUzSGS5wQAJ7R:

| member | unguided sim chamfer | guided sim chamfer | imagined (last frame) | sim vs gen | picked iteration |
|---|---|---|---|---|---|
| 0 | 7.16 mm | **5.70 mm** | 5.51 | 1.5 mm | 8 |
| 1 | 11.49 | 7.24 | 7.04 | 0.4 | 7 |
| 2 | 13.66 | 6.91 | 6.79 | 0.9 | 12 (planner's pick: lowest all-frame objective) |

Frame-0 blob vs the letter R footprint = 7.1 mm (the letter is fitted to the blob). Old numbers on this letter: cfg47 5.93 mm,
best of 50 seeds 3.31 mm. The all-frame objective is noisy across iterations at lr 0.1 (member 1 jumped to 24.9 mm at
iteration 11); the best iterate is kept, so the plan is fine, but a lower or decaying lr and more iterations are the obvious
next arms. The Genesis-vs-generated gap is 0.4-1.5 mm, so the imagined chamfer is a reliable selector on this checkpoint.

## 8. Stage 1 result (28 arms, 2026-09-25 05:07-08:35, one RTX PRO 6000; table `sweeps/letters/stage1_summary.md`)

Genesis last-frame chamfer, mm; "pick" = the member with the lowest optimized objective (what a deployment can choose), "best" = the
best member in simulation. Unguided best draw: R 7.71, C 14.41, L 13.83, I 17.33. Old paper: cfg47 R 5.93 / C 8.83 / L 15.21 / I 9.17;
best-of-50 R 3.31 / C 6.40 / L 4.58.

| letter | arm (24 it, lr 0.1, S6, tf0.8, all frames unless noted) | pick | best | gap |
|---|---|---|---|---|
| R | last 15 frames | **5.03** | 5.03 | 0.9 |
| R | last frame only | 5.06 | 5.06 | 0.9 |
| R | S=10 | 5.44 | 5.44 | 0.8 |
| R | 200 frames | 5.66 | 5.66 | 0.6 |
| R | 96 it | 6.37 | 6.24 | 1.9 |
| R | 48 it | 6.43 | 6.38 | 0.9 |
| R | 48 it, 0.2->0.02 | 6.49 | 6.49 | 2.0 |
| R | Adam 0.1 | 6.65 | 5.34 | 1.2 |
| R | lr 0.2 | 6.74 | 5.03 | 1.4 |
| R | lr 0.05 | 6.97 | 6.28 | 1.8 |
| R | t_freeze 0.5 | 7.16 | 7.16 | (no optimization: objective stayed at the unguided value) |
| R | 48 it, 0.1->0.01 | 7.86 | 6.12 | 1.8 |
| R | baseline | 8.18 | 6.60 | 1.7 |
| R | t_freeze 0.65 | 9.74 | 6.74 | 8.6 |
| R | seed 1 | 11.12 | 7.22 | 1.8 |
| R | 96 it, 0.2->0.02 | 14.81 | 5.67 | 11.5 |
| C | baseline | 8.14 | **5.14** | 1.1 |
| C | 48 it, 0.2->0.02 | 16.31 | 15.76 | 12.8 |
| C | 48 it | 18.69 | 14.14 | 13.8 |
| C | t_freeze 0.5 | 22.25 | 12.15 | 12.4 |
| I | baseline | **8.62** | 8.62 | 6.2 |
| I | 48 it, 0.2->0.02 | 10.18 | 9.41 | 7.0 |
| I | 48 it | 11.45 | 8.37 | 5.6 |
| I | t_freeze 0.5 | 13.37 | 12.06 | 26.9 |
| L | baseline | 10.42 | 9.51 | 4.0 |
| L | t_freeze 0.5 | 14.72 | 14.72 | 3.1 |
| L | 48 it, 0.2->0.02 | 15.67 | **5.31** | 9.9 |
| L | 48 it | 15.84 | 11.38 | 10.5 |

Facts:
- Scoring the chamfer on the last 15 frames (or the last frame) gave the best planner-picked R (5.03 / 5.06 mm) and picks that were also
  best in simulation, at the same 24-iteration budget. All-frame scoring picks members by their whole-trajectory average, and the pick was
  the best member in only 3 of 16 R arms.
- Long all-frame optimization (48-96 iterations) on C, L, I and once on R produced plans with imagined chamfer 3-6 mm that replayed at
  10-19 mm: the model imagined the dough moving 13-18 mm while Genesis moved it 0.3-8 mm (one member lifted the paddle 29 mm). Paddle
  tracking was exact (0.1 mm) and paddle speed unchanged (max 4.1 mm/frame), so the failure is hallucinated dough motion without contact.
  The added `phantom` term (dough-centroid displacement gated by the nearest paddle-dough distance beyond 20 mm) scores those plans at
  6-12 mm and every good plan or unguided draw at 0.3-0.5 mm.
- t_freeze 0.5 and 0.65 hurt on every letter (little or no optimization at 0.5). Adam, lr 0.05/0.2, and the decays did not beat flat lr 0.1.
- S=10 (5.44) and 200 frames (5.66) both helped R with reliable picks; not yet combined with late-frame scoring.
- 9.5 s per iteration (S=6), 12.5 GiB; a Genesis replay of one member 90-150 s.

Stage 2 (queued automatically, `sweeps/letters/stage2.json`, 20 arms): late-frame scoring on all four letters at 24 and 48 iterations, the
phantom penalty (weight 1 and 3) at 48 iterations, R with S=10, the decay, lr 0.2, two more seeds, 96 iterations, last-frame-only at 48.
Page (best R / C / L of stage 1): https://claude.ai/artifact/CUKZhQDy4DUzSGS5wQAJ7R

## 9. Stage 2 (partial) and the disk incident, 2026-09-25 11:00-13:00

Stage 2 results so far (Genesis last-frame chamfer, mm; `sweeps/letters/stage2_summary.md`):

| arm | pick | best member | note |
|---|---|---|---|
| R last-15, 48 it | 5.52 | 5.51 | no gain over 24 it (5.03) |
| C last-15, 24 it | **5.71** | 5.71 | first reliable C pick |
| L last-15, 24 it | **6.22** | 6.22 | first reliable L pick (old cfg47 15.21) |
| I last-15, 24 it | 10.74 | 9.13 | worse than all-frame I (8.62) |
| C last-15, 48 it, phantom 1 | 5.45 | 5.45 | penalty fixes the 48-it failure (18.69 without it) |
| L last-15, 48 it, phantom 1 | 6.98 | 5.96 | no failure, no gain over 24 it |
| R last-15, 48 it, phantom 1 | 5.19 | 5.19 | penalty costs nothing on R |
| **C all-frame, 48 it, phantom 1** | **3.55** | 3.55 | best letter result of the study; same setting without the penalty: 18.69 |
| I last-15, 48 it, phantom 1 | 9.18 | 9.18 | gap 5.7 mm: NOT phantom motion (term 1.1 mm). The plan's one big early stroke moves the dough 6-8 mm less in Genesis than imagined and the error never recovers |
| C last-15, 48 it, phantom 3 | 5.59 | 5.59 | weight 1 is enough |
| R last-15, 48 it, 0.2->0.02 | 5.45 | 5.45 | plateau |
| R last-15, 24 it, lr 0.2 | 5.48 | 5.48 | plateau |
| R last-15, 24 it, seed 1 / seed 2 | 6.29 / 5.22 | 6.29 / 5.22 | three seeds: 5.0-6.3 in sim, imagined 4.2-5.0 |
| R last-1, 48 it | 5.57 | 5.28 | plateau |

R sits at 5.0-5.5 mm in sim for every late-frame arm; the 1-2 mm imagined-to-sim gap, not the optimizer, is the cap.

Disk: `/data/user_data/mbronars` hit its quota at 10:44 (the 11 GB `pwm_final_paper/results` plus ~12 GB of letter runs). The L 48-it
late-frame arm died mid-save and the sweep driver crashed. All completed letter runs were moved to
`/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_letters/runs/mould/` (symlinks left under `runs/mould/`), and every later
arm writes there directly (`base.runs` in the stage JSONs; `bin/letter_sweep.py` honours it). Two stage-1 runs were lost when three copies
of the relocation script raced (`letters_R_it24_last1`, `letters_I_it48_lr0.1`); their numbers survive in `stage1_summary.json` and both are
queued for re-run at the front of stage 3. Lesson: one relocation process, and never `nohup` without `setsid` (a session restart killed the
server, the sweep and the relocation). Stage 2 restarts automatically from `scratch/run_stage2_then_3.sh`, then stage 3 (18 arms).

## 10. Final table after stages 1-3 (67 arms, 2026-09-25 05:07-16:24, one RTX PRO 6000)

Genesis last-frame chamfer of the planner's pick (member with the lowest optimized objective), mm. Runs live in
`/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_letters/runs/mould/` (symlinked from `runs/mould/`);
tables `sweeps/letters/stage{1,2,3}_summary.md`. Page: https://claude.ai/artifact/CUKZhQDy4DUzSGS5wQAJ7R

| letter | best recipe (population 3, S=6, unitgrad lr 0.1, shell, churn, t_freeze 0.8, 150 frames) | pick | old cfg47 | old best of 50 seeds |
|---|---|---|---|---|
| R | chamfer on the last 15 frames, 96 iterations | **4.84** | 5.93 | 3.31 |
| C | all-frame chamfer + phantom penalty (w=1), 48 iterations | **3.55** | 8.83 | 6.40 |
| L | chamfer on the last 15 frames, 24 iterations | **6.22** | 15.21 | 4.58 |
| I | all-frame chamfer + phantom penalty, 48 iterations | **7.86** | 9.17 | (none) |

Every pick above was also the best member of its population in simulation.

What the 67 arms established:
- Late-frame scoring (last 15 frames or last frame) makes the pick reliable on R, C, L: the pick was best-in-sim in 17 of 19
  late-frame arms, against 3 of 16 all-frame R arms in stage 1.
- The phantom-motion penalty is what makes long optimization safe on C (48 it: 18.7 mm without, 3.55 with). It does not help R
  (every penalized R arm is 5.2-6.9 vs 4.8-5.5 without), L (12.8) or I (7.9 either way).
- Budget is letter-specific: R improves to 96 iterations with late-frame scoring (5.03 -> 4.84); L is best at 24 and worsens at
  48 (6.22 -> 7.96); C needs 48 with the penalty and does not gain at 96 (5.92).
- Seeds: R picks over six seeds at 24 it span 5.03-6.29; C with its recipe over three seeds gives 3.55 / 9.49 / 14.34, but the
  imagined chamfer ranks them correctly (4.20 / 7.16 / 8.08), so "several seeds, one pick by imagined chamfer" is a sound protocol
  and is what a paper run should do (the old showcase used 50 seeds).
- t_freeze below 0.8, S=10, 200 frames, Adam, lr 0.05 / 0.2 and the decays never beat the recipe above on the pick.
- I stays at 7.9: its imagined-to-sim gap (3-8 mm) comes from an over-predicted first push (centroid diverges 6-8 mm at the
  first contact), not phantom motion. Options not tried: a per-stroke dough-displacement cap in the objective, or receding-
  horizon replanning against Genesis.
- Repeats with the same seed are not bit-reproducible on the compiled bf16 server (objective spread 1-3 mm).
- Cost: 9.5 s per iteration at population 3 (contended: up to 19 s), Genesis replay 90-150 s per member. One arm at 24 it = 5 min
  plus 5-8 min of replays; the whole study was about 11 GPU-hours.

Incidents: the home volume hit its quota mid-sweep (the runs were relocated to group storage); two run folders were destroyed
by two relocation scripts racing from two branches of the same session (both re-run, numbers consistent); one replay
(`letters2_R_last15_it48_S10`) fails on a Genesis pickle write every time and has no sim number.
