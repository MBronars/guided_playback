# Distinct guidance goals from the same ordinary sample

The user requested destinations that are visibly different from the ordinary
sample. Eight experiments started on dev-blackwelle, allocation 10446392, on
September 15 at 11:43 a.m. New York. Two GPU queues run opposite directions first.

Open the local [results and GIF comparisons](../../validation/guidance_distinct_goals_2026-09-15/RESULTS.md)
in Markdown Preview. The report updates as each experiment completes. Its table
retains all declared goals, including unsuccessful steering and geometric failures.

## Declared experiment

| Scene | Targets | What is fixed |
|---|---|---|
| Syrup bottle 8270 | Lift the object origin 15 cm above its observed start, then shift it 15 cm left, right, near, or far; hold near that point over the last six frames. | Same observed scene, checkpoint, initial seed, horizon, and physical objective. |
| Teacup 7803 | Move the requested placing position 15 cm left, right, near, or far; retain its requested orientation and require settling and release. | Same observed scene, checkpoint, initial seed, horizon, and physical objective. |

Directions are defined in world coordinates and labeled in the top view:
left/right = −x/+x; near/far = −y/+y. Opposite targets are 30 cm apart. The targets
are 12.1–18.3 cm from the corresponding previously saved ordinary endpoint.
The goal locations were frozen before viewing the new optimization results.
Their reachability by a full robot arm has not been established.

The exact target coordinates, original task pose, observed anchors, fixture paths,
and baseline distances are in the immutable
[manifest](../../validation/guidance_distinct_goals_2026-09-15/manifest.json).
This is an exploratory diagnostic on previously inspected scenes, not an untouched
holdout benchmark. The ordinary future is used to check that the targets differ
from the baseline, not to prescribe a successful gripper or object path.

## Optimizer and compute

We reuse the qualified final campaign source at
`validation/guidance_noise_2026-09-15/source_v17`. The selected source files and
the existing real-checkpoint qualification are recorded in
[qualification.json](../../validation/guidance_distinct_goals_2026-09-15/qualification.json).
No model architecture, loss implementation, or sampling algorithm changes for
these experiments. The run flags use existing held-point picking and mesh-based
pick/place objectives.

The fixed EMA checkpoint is
`/data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond_cleanctx20k_from40k_0914/checkpoints/final.pt`,
SHA256 `99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba`.
It is an unconditional bidirectional articulated UNet with clean context and
hidden future actions. Four observed frames, 36 future frames in 16+16+4 windows,
pyramid25, FP32, seed 8311. Model weights remain frozen.

Each run uses 24 Adam noise updates, learning rate 0.03, all valid object/gripper
pose-noise channels, RMS change cap 0.25 per window, and noise prior weight 0.001.
Best-so-far trajectories are saved through updates 6, 12, and 24. Selection uses
the fixed total objective. The independent geometric screen does not secretly
replace the selection objective or relax its thresholds.

The queues have a two-hour deadline, per-run timeouts, and durable logs/status.
The existing packet driver stops after two consecutive infrastructure failures.
These are existing-checkpoint inference experiments, not training or a comparison
between models.

## What the report checks

- Completed runs in the same scene must have bitwise-identical initial particle
  trajectories and pose translations. A target-only change must not resample the
  ordinary trajectory. The check is performed on the new saved outputs.
- Final distance to the requested point and maximum distance during the final
  six frames. The held-goal tolerance is 20 mm in 3D.
- Final displacement from the ordinary sample and separation between opposite
  guided endpoints. These measure observable steering independently of success.
- Peak collision-mesh overlap and every failing component of the full geometric
  task screen. Low endpoint error does not hide missing contact or release.

The figures put the ordinary path, every requested target, and all completed
guided paths in the same coordinates. The movies combine 3D and top views with
shared bounds, labeled targets, and per-frame goal error and displacement from
the ordinary sample. They contain no GT panel that could be mistaken for a
guided result. Pending directions remain labeled pending.

The figure/report implementation is
[`render_distinct_guidance_goals.py`](../../scripts/dev/render_distinct_guidance_goals.py).
The report reads saved results; it never adjusts trajectories to make them look
more different. Predicted physical plausibility and execution are distinct:
**these new goals have not been replayed in physics**. No simulator or robot
success is claimed from these movies.

## First completed pair, 11:53 a.m. New York

The left/right bottle runs completed all 24 updates. Their ordinary particle
trajectories and pose translations are bitwise identical to each other and to
the previously saved ordinary sample. Their guided endpoints are **163.2 mm
apart**, compared with 300 mm between their requested targets.

| Bottle goal | Ordinary final error | Guided final error | Shift from ordinary | Peak mesh overlap | Remaining geometric failures |
|---|---:|---:|---:|---:|---|
| Left | 151.2 mm | 71.5 mm | 89.6 mm | 0.00 mm | Goal hold, finger proximity, relative translation |
| Right | 149.5 mm | 83.7 mm | 83.9 mm | 2.49 mm | Goal hold, penetration |

This establishes visible steering for this pair, not successful task completion.
Maximum goal error during the final six frames remains 86.8/129.9 mm.
Runtime is 529/531 seconds for the two runs.

![First completed opposite-goal comparison](../../validation/guidance_distinct_goals_2026-09-15/figures/bottle_8270_02_complete_paths.png)

[First comparison GIF](../../validation/guidance_distinct_goals_2026-09-15/figures/bottle_8270_02_complete.gif)
· [Updating report for the full eight-goal batch](../../validation/guidance_distinct_goals_2026-09-15/RESULTS.md)

The figure/GIF output was inspected, its saved numbers were checked directly
against the arrays, and the initial-draw identity check passed. The remaining
six experiments were running or queued at this checkpoint; consult the updating
report for their actual status.
