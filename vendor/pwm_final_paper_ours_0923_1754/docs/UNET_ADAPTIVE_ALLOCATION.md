# Adaptive UNet particle tokens

`configs/unet_adaptive.yaml` is an opt-in experiment with the same UNet widths,
depths, geometric conventions, optimizer and training horizon as `unet.yaml`.
It allocates coarse particle tokens by each object's valid particle count and
rigidity instead of its material label. This is an engineering option, not an established accuracy
improvement; the material-table recipe remains the baseline.

## Allocation

The new `unet_l1_rule: capped_quarter` starts with the nearest power of two to
`N / 4`, clamped between `unet_l1_rule_min` and `unet_l1_rule_max`. The experimental
recipe uses a preferred minimum of **128**, a deformable-object maximum of **512**,
`unet_l1_rigid_max: 128`, and `unet_l1_scene_budget: 1024`.

The rigid ceiling applies to an object only when **all its particles belong to
rigid bodies**, using `body_type` and `particle_body`. It covers both actuated and
passive rigid objects. A gripper with several rigid links still shares one object
budget; the ceiling is not multiplied by its number of links. An object containing
deformable particles retains the larger adaptive budget. Material IDs and action
flags do not select the ceiling.

Sidecar-free particle streams such as MPM/water have no known rigid members and
keep the general budget. Partial body metadata raises an error. Batches must
share the same per-object rigid/deformable classification as well as object
counts, because their coarse tensors share one layout.

An object cannot supply more anchors than it has particles. Available capacity is
rounded down to a multiple of the product of the later strides (8 for `4,4,2`).
For example, 100 particles supply 96 anchors; 128 particles supply 128. Objects
smaller than that stride product keep all their particles at L1; later levels
bottom out at one token instead of becoming empty.

When the sum of requests exceeds the scene cap, every object first receives its
feasible minimum, `min(128, available capacity, applicable ceiling)`. The remaining budget is shared
in proportion to requested extras above those minimums, rounded down to complete
hierarchies. Rounding can leave some capacity unused; the cap is an upper bound.
No object receives preferential treatment because of its ID or order.

If the cap cannot accommodate the feasible minimums, construction of the scene
graph fails with the counts and an actionable error. For example, nine large
objects need at least 1,152 L1 particles at a 128 minimum, so the 1,024 cap cannot
satisfy that policy. Increase the cap or explicitly lower the minimum.

| Example object particle counts | Material-table L1 counts | Adaptive L1 counts |
|---|---|---|
| Pushing: 128 + 128 | 128 + 128 | 128 + 128 |
| Pickup: 512 + 671 | 128 + 128 | 128 + 128 |
| Four MPM objects: 1,024 each | 256 each | 256 each |
| Cloth + two end effectors: 10,000 + 1,000 + 1,000 | 512 + 128 + 128 | 512 + 128 + 128 |
| Three MPM objects: 4,096 each | 256 each | 336 each (1,008 total) |

For two deformable objects and one rigid object with 4,096 particles each, requests
are 512 + 512 + 128. The scene cap yields 448 + 448 + 128. With only one deformable
and two rigid objects, requests total 768 and the remaining 256 slots stay unused.
A cap is a ceiling; it does not force the allocator to fill the scene budget.

These are illustrative counts, not corpus frequencies. The policy is material
agnostic; physical-property and material conditioning still enter the network.
Point count measures sampling density as well as object extent, so this rule
does not directly measure physical complexity. It can allocate *more* tokens and
cost more than the fixed recipe for dense deformable objects. The rigid ceiling
removes the automatic increase for densely sampled grippers, but 128 is still a
conservative starting value, not a proven optimum for every rigid shape/contact.

## What the caps cover

- The scene cap covers **L1 particle tokens**. Native pose tokens and learned
  registers are additional. Register counts still use the configured maximum,
  not the observed batch maximum, so parameter shapes do not vary between scenes.
- L0 still contains every valid input particle. Its within-object attention,
  stem and skip activations are not bounded by the L1 cap. This is not an overall
  GPU-memory limit.
- Counts depend only on the scene layout. They do not change with diffusion noise
  or between frames in a trajectory. FPS still uses the clean first-frame geometry;
  spatial neighborhoods still update from candidate positions.
- Objects can have unequal counts; batches must still share an object layout,
  and each particle must retain its identity throughout the trajectory. This
  change does not add arbitrary padding or change the data sampler.
- The existing KNN implementation shares a neighbor count across objects, clamped
  to the smallest source segment. A one-token coarse object therefore reduces
  interpolation K for that whole seam. Tiny-object support is valid, but its
  effect on other objects' neighborhood resolution needs quality evaluation.
- Static regional compilation can specialize on additional coarse shapes.
  A small compile test cannot establish compile-cache behavior over a full corpus.

## Compatibility and use

The previous material-table policy and `pow2_quarter` rule retain their semantics.
`unet_l1_scene_budget` and `unet_l1_rigid_max` default to zero (disabled), including
when loading old checkpoints. Count-only adaptive checkpoints therefore retain
their original counts. The new rule rejects material tables and a nonzero `l1_default` to
prevent an apparently adaptive recipe from silently using fixed counts.

The rigid ceiling must be a positive multiple of the later stride product and
no larger than `unet_l1_rule_max`; zero disables it. It may be below the preferred
minimum, in which case it also lowers that object's feasible minimum. Changing
the rigid ceiling changes the training contract and is rejected as a resume even
though learned parameter shapes stay identical.

Water still requires `model.num_mat_types=7` and the usual water normalization
configuration. It needs no material-table override under this rule: 4,096 water
particles receive 512 L1 tokens before any scene cap is applied.

The new recipe keeps the old maximum of 512, so its learned parameter shapes
match the baseline. Equal shapes allow explicit weight transfer; they do not make
the two allocation policies equivalent. Resume validation rejects policy changes
even when the weights are shape-compatible. Inference uses the policy saved in
the checkpoint. Changing the maximum can also change register parameter shapes.

Correctness tests:

```bash
WANDB_MODE=disabled PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m pytest tests/test_unet_adaptive_allocation.py tests/test_unet_rigid_allocation.py -q
```

On a CUDA-enabled interpreter the same suite includes FP32/BF16 forward/backward,
native particle/pose objectives and raw/EMA checkpoint reloads. Synthetic optimizer
steps verify that gradients update the network; they are not prediction-quality
benchmarks. Real-corpus training requires the dataset/evaluation preflight in
`CLAUDE.md`. Before adopting this recipe, compare identically initialized runs on
the same pinned training and held-out windows, with matched noise and sampling
protocols, and report both accuracy and compute. Include density changes and
scenes where the scene cap actually binds.

## Initial count-only validation, 2026-09-13

The affected CPU checks passed (298 in the broader suite plus two added water
checks). The final GPU-enabled allocation suite passed all 67 cases, including
28 CUDA FP32/BF16 cases, on an RTX PRO 6000 Blackwell. Actual regional Inductor
forward/backward/evaluation agreed with eager execution across changing shapes
on a reduced-width native-pose model with dropout disabled.

Graph checks passed on 51 previously pinned real windows across MPM, pickup,
cloth and pushing. Thirteen of those windows also completed a backward pass and
16-frame diffusion sampling with reduced-width random weights. Real water was
not sampled; material-id/configuration and synthetic water-channel checks passed.

A separate full 331,618,989-parameter eager BF16 benchmark completed objective,
backward, clipping and AdamW steps with unchanged weights on two synthetic scenes.
Shared-GPU timing variability prevented a reliable speed conclusion, and reducing
L1 counts did not consistently reduce total peak memory. The complete protocols,
seeds, source hashes, sample receipts and measurements are in the local ignored
[validation report](../validation/unet_allocation_2026-09-13/README.md). No
prediction-quality comparison or real-corpus optimization run was performed.

## Rigid-ceiling refinement

The revised GPU-enabled allocation suite passed all **91** cases, covering the
new ceiling, mixed objects, passive rigid bodies, material/action independence,
invalid metadata, native optimizer steps and raw/EMA loading of all three policy
generations. The [refinement report](../validation/unet_rigid_allocation_2026-09-13/README.md)
records the final CPU, compiled and real-sample checks separately from the initial
count-only experiment. Earlier timing/memory measurements describe that earlier
policy and should not be read as measurements of this refinement.
