# Unguided sampler check, 2026-09-14

**Keep pyramid/25 as the default for this checkpoint.** This four-clip check did
not find a consistent contact improvement from more steps or joint denoising.
Joint denoising reduced object movement relative to the hand in three clips,
but finger gaps remained large and placement ended farther above the recorded
resting height. These results do not establish that sampler tuning in general
cannot help; they rule out these four settings as an obvious fix on the pinned
examples.

All 128 rollouts completed: four clips × eight draws × four sampler settings.
All four baseline batches reproduced the previously repaired samples **bit for
bit**, including the native rigid poses. Every arm passed initial-noise pairing,
clean-context, hidden-pose-action and forward-count checks. No model weights,
production sampler defaults or guidance objectives changed.

**Protocol and provenance.** EMA weights from
`/data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond/checkpoints/final.pt`,
SHA256 `f4c4d559d1ef520fa0465650e1fe2a48b6e4a4abe991cb84e159da8d68f8b65c`.
This is the bidirectional, articulated, pose-native UNet checkpoint saved at
step 40,000, trained with `noise_context=true` and `action_dropout=1.0`.
Inference used clean context at timestep zero, generated the gripper as well as
the object, and used no guidance. The checkpoint-owned normalization and original
random FPS policy were preserved. FP32, no autocast or compilation.

The four existing fixtures were reused without reselection. They were selected
near the median ground-truth motion of each pickup collection among the first
512 stratified validation clips, with the registered asset-holdout exclusion
(`holdout_TRAIN_block.json`). The combined quality blocklist is training-only;
the separate place-strict file was not applied by this validation loader.
Seeds 54321/54322/54323/54324 belong to pick/place/convex/upright,
respectively. Each seed generates a batch of eight draws. Four observed frames
precede 36 generated frames; corrected windows start at absolute frames 4, 20
and 36 and generate 16, 16 and 4 frames. Frozen source is the same snapshot used
for the repaired baseline, with file hashes verified before and after sampling.

| Setting | Schedule | Nominal steps | Calls by window | Total calls | Mean batch time |
|---|---|---:|---|---:|---:|
| P25, current | pyramid | 25 | 40 + 40 + 28 | 108 | 30.39 s |
| P50 | pyramid | 50 | 65 + 65 + 53 | 183 | 50.04 s |
| J36 | full_sequence | 36 | 36 + 36 + 36 | 108 | 28.53 s |
| J61 | full_sequence | 61 | 61 + 61 + 61 | 183 | 48.32 s |

Times are for an eight-draw batch, averaged over the four clips. Pick/place used
GPU 0 and convex/upright GPU 1 in allocation 10434620 on babel-l9-16, both RTX PRO
6000 Blackwell Server Edition. Loading and file output are excluded; first-call
noise hashing is included. Each arm was timed once per clip in the table's
order, so cold-start/order effects are possible. Equal forward counts do not
mean equal FLOPs: the schedules allocate different counts to the short tail.
P50 took 1.65× the baseline time in this check.

Pairing required more than resetting a rollout seed. Each model call consumes
randomness for FPS, so increasing the call count otherwise changes the next
window's initial noise. The harness captured CPU/CUDA RNG states immediately
before each baseline window, replayed them for the other arms, and checked hashes
of both particle and masked-pose initial noise. FPS remains random; corresponding
forward indices use the same RNG progression, and extra forwards consume extra
FPS draws within that window. Subsequent contexts are each arm's own predictions.

`uncertainty_scale=1` was fixed. Consequently, doubling pyramid's step count also
halves the normalized delay between adjacent frames: P50 is not a pure Euler
step-size ablation holding the continuous pyramid path fixed. The joint schedule
uses the same noise level at every future frame within a window. Neither schedule
denoises all 36 rollout frames in one model window.

**Finger proximity.** The following is the mean, over eight draws and the indicated
frames, of the larger of the left/right nearest-particle gaps. Both fingers must
be close for this value to be small. Values are millimetres. The reference column
describes the recorded trajectory, not the only valid unconditional future.

| Clip / fixed phase | Recording | P25 | P50 | J36 | J61 |
|---|---:|---:|---:|---:|---:|
| Pick, frames 24–39 | 3.14 | 19.51 | 19.74 | 18.14 | 18.58 |
| Place, frames 4–15 | 4.84 | 5.01 | 5.21 | 5.91 | 5.90 |
| Convex, frames 24–39 | 3.05 | 9.39 | 9.41 | 9.81 | 9.91 |
| Upright, frames 24–39 | 3.69 | 11.35 | 12.46 | 11.88 | 12.15 |

The fraction of those frames with **both fingers within 5 mm** was:

| Clip | P25 | P50 | J36 | J61 |
|---|---:|---:|---:|---:|
| Pick | 0.8% | 2.3% | 1.6% | 0.8% |
| Place | 50.0% | 56.2% | 47.9% | 47.9% |
| Convex | 19.5% | 20.3% | 18.0% | 15.6% |
| Upright | 31.2% | 24.2% | 14.8% | 14.1% |

The positive finding is reduced motion of the object centre in the hand's frame:
P25 → J36 was 1.92 → 1.44 mm/transition for pick, 2.38 → 1.55 for convex, and
2.15 → 1.68 for upright. Place was 3.39 → 3.43. These are descriptive relative
motion measurements over transitions wholly inside the same fixed phases. They
are not labels for slipping: a valid regrasp or release can change this quantity.
Reduced relative motion can coexist with an incorrect finger–object separation.

**Placement height.** Over frames 32–39, the mean absolute difference between
the lowest sampled object point and its recorded settled-height reference was:

| P25 | P50 | J36 | J61 |
|---:|---:|---:|---:|
| 9.19 mm | 11.30 mm | 14.48 mm | 14.64 mm |

The reference was 11.62 mm in world z. Mean predicted lowest-point heights were
19.15, 22.54, 26.10 and 26.21 mm, respectively. Joint denoising did not solve the
early stop in lowering the object. This comparison does not measure signed
mesh penetration or physical table clearance; the table plane was not calibrated.

Object path lengths and endpoint spreads remain comparable; there is no obvious
collapse to stationary or identical draws. Mean pairwise endpoint spread for
P25 → J36 was pick 89.9 → 93.3 mm, place 46.1 → 46.0, convex 64.5 → 74.6,
upright 92.6 → 83.5. Object-motion second differences generally increased slightly.
Lower second differences alone would not establish better physics, since true
contact and acceleration also produce them. All per-draw traces and boundary
steps are retained in the measurements.

**Visuals and exact artifacts.** Start with the
[visual report and GIF index](/data/user_data/robertre/particle_world_model/validation/sampler_check_2026-09-14/README.md).
The first-draw GIFs use draw 1 for every clip, selected before inspecting outputs.
Two additional grids per clip show all eight draws across all four arms. They use
common axes, camera and playback timing; the first four frames are observed.

- [Paired measurements, all draws](/data/user_data/robertre/particle_world_model/validation/sampler_check_2026-09-14/paired_measurements.png)
- [Placement height and finger-gap traces](/data/user_data/robertre/particle_world_model/validation/sampler_check_2026-09-14/place_height_and_gap.png)
- [Exact measurements](/data/user_data/robertre/particle_world_model/validation/sampler_check_2026-09-14/measurements.json)
- [Run plan](/data/user_data/robertre/particle_world_model/validation/sampler_check_2026-09-14/plan.json) and [verification receipt](/data/user_data/robertre/particle_world_model/validation/sampler_check_2026-09-14/verification.json)
- [Pinned input selection, normalization and exclusion provenance](/data/user_data/robertre/particle_world_model/validation/phantom_regeneration_2026-09-14/inputs.json)
- Per-clip `*_experiment.json` and per-arm `*_receipt.json` in the artifact directory
  contain the checkpoint, input, source and noise hashes, calls, runtime and
  software versions. Saved `.npz` files contain all particles and native R/T tracks.

The versioned sampling harness is
[scripts/dev/check_uncond_sampler.py](/data/user_data/robertre/particle_world_model/scripts/dev/check_uncond_sampler.py).
The artifact directory contains the exact executed harness, `analyze.py`,
`render.py`, logs and captured per-window RNG states. Reproduction requires the
pinned fixture/source directory and checkpoint; use a fresh output directory.

```bash
PYTHONPATH=.:validation/overnight_2026-09-14/vendor \
  /data/user_data/robertre/conda_envs/pwm_blackwell/bin/python \
  scripts/dev/check_uncond_sampler.py \
  --reference validation/phantom_regeneration_2026-09-14 \
  --out validation/sampler_check_repeat --collections pick,place,convex,upright
```

This is a targeted diagnostic on **four clips**, not a population benchmark or a
robot success-rate evaluation. Eight draws from one context do not create eight
independent test scenes. Fixed frame slices can penalize different valid timing;
nearest-particle proximity is unsigned and cannot certify grasping or absence of
penetration. No automatic phantom-motion success rate is claimed.

The next useful generation-only experiment would be a controlled clean-context
fine-tune of this unconditional checkpoint, with the same pinned evaluation.
That remains a separate training decision; this sampler check did not launch it.
