# Clean-context pickup fine-tuning preflight, 2026-09-14

**Prepared and checked; training has not been submitted.** Continue the 40K
unconditional checkpoint we just evaluated, using its EMA weights and a fresh
optimizer. The existing `train.init_from` path supports this. `--resume` of the
old noisy-context run correctly refuses the changed objective.

The existing `launch/legacy/train_pickup_uncond_unet.sbatch` is a valid route for its
documented experiment, but defaults to the older 200K **conditional** checkpoint.
For this continuation use the dedicated
[sbatch script](/data/user_data/robertre/particle_world_model/launch/legacy/train_pickup_uncond_cleanctx.sbatch)
and [config](/data/user_data/robertre/particle_world_model/configs/experiments/pickup_uncond_cleanctx.yaml).
Its effective model configuration and normalization match the actual 40K source
checkpoint, checked with `validate_initialization`.

The proposed source is
`/data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond/checkpoints/final.pt`,
SHA256 `f4c4d559d1ef520fa0465650e1fe2a48b6e4a4abe991cb84e159da8d68f8b65c`.
It was saved at step 40,000 and trained with context noise. This experiment uses
the same bidirectional, articulated, pose-native UNet and generates both the
gripper and object (`action_dropout=1`, `unet_actions=false`).

| Setting | Proposed value |
|---|---|
| Initialization | Source EMA; fresh AdamW, schedule and EMA tracking |
| Compute | One PRO 6000, one process, batch 8, BF16, regional compilation |
| Budget | 10,000 training attempts, full cosine schedule |
| Learning rate | Peak 1e-5, 500 warmup attempts |
| History | Clean; no independent particle noise or light jitter |
| Window | Four observed frames, sixteen future transitions |
| Targets | Same pose-native velocity objective; pose targets remain unclipped |
| Sampler | Pyramid/25, uncertainty scale 1, no guidance |
| Checkpointing | Every 1,000 steps; validation every 1,000 |
| Training diagnostics | 40 validation batches; eight sampler batches |
| Run directory | `/data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond_cleanctx_from40k_0914` |

The output directory did not exist at preflight. The prepared launcher refuses
an existing directory unless an explicit checkpoint from this new run is passed
with `--resume`, so repeating the start command cannot overwrite the run silently.

**Data.** The same registered pickup `*_success` collection family as the source
checkpoint. The last rounded 10% of sorted shards in each root is validation;
the rest is training. Actual inventories were read without training or scanning
all retained trajectories.

| Canonical root | Training shards | Validation shards |
|---|---:|---:|
| `/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success` | 57 | 6 |
| `/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_convex_success` | 44 | 5 |
| `/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success` | 44 | 5 |
| `/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_upright_success` | 44 | 5 |

Total: 189 training shards (60.08 GB) and 21 validation shards (6.48 GB), before
row filtering. Counts, paths, byte sizes and modification times are pinned in the
preflight inventory; retained episode counts were not rescanned. The saved
source configuration's data fields match the proposed resolved data fields.
This inventory records the current corpus; the source checkpoint itself does not
contain a historical shard-byte manifest.

Training uses `data/manifests/robert_pickup_data_4success_combined.json` and
`/data/group_data/katefgroup/VLA/3DPCWM/stats/unified_v1/blocklists/holdout_TRAIN_block.json`.
Validation uses only the latter, keeping validation assets matched to training.
The loaded lookup has 6,653 entries for training and 6,514 for validation; these
are lookup entries, not retained-episode counts. The standalone place-strict
file is not passed by this validation factory. This corrects the exclusion
description in the preceding sampler report; its input tensors and numerical
results are unchanged. The correction is recorded
[here](/data/user_data/robertre/particle_world_model/validation/sampler_check_2026-09-14/provenance_correction_2026-09-14.json).

Normalization remains `xy_p99_per_domain`, checksum
`2bc31504482f745a5bfce97558e7a366e1b6800d2bb149a8853e663d797587de`.
The full numerical snapshot is in the resolved configuration; no statistics are
refitted. Pickup is the only training domain. MPM sequence collection changes
do not enter this training stream.

**Checks on the current changes.** The retirement of noisy context for new
recipes, preservation of old checkpoints as initialization sources, and exclusion
of rigid rows from light particle jitter passed the training/checkpoint/objective
tests. The revised evaluation source selection and MPM sequence adapter tests
also pass. Counts: 340 training/checkpoint/config/objective tests, 270 evaluation
and MPM-related tests, and two new UNet context-parity tests: **612 unique checks**.

There is no context-timestep discontinuity for this UNet. Although metadata
changes from explicit zero to `None` when context noise is disabled, its wrapper
converts `None` to zero before embedding. A regression test exercises the full
sampler with nonzero weights for both causal and bidirectional UNets and confirms
bitwise equality with fixed noise. The comments suggesting a literal zero
embedding applied to the Transformer and have been clarified. No new timestep
option or network change is required.

No real-corpus optimizer update, new GPU training smoke or compiled training run
was performed in this preflight. Those begin only after the training plan is
confirmed. The CPU tests cover fresh initialization, optimizer/EMA ordering and
save/resume behavior; they do not constitute a new GPU throughput measurement.

**Proposed evaluation after confirmation.** Retain the already pinned four clips
and eight draws per clip from the sampler check, using pyramid/25, C4/F16/H36 and
paired initial window noise. Inspect step 1K, 5K and 10K for finger proximity,
object movement relative to the hand, placement height and sample diversity.
These four scenes remain development diagnostics. For broader coverage, score
the source checkpoint and final fine-tuned checkpoint on the same 512 stratified
validation clips, H36, seed 0, with hidden actions and EMA weights. Recompute both
sides under one frozen evaluator; do not compare to historical aggregate scores.
The trainer's `best.pt` still selects by unconditional `sample_mm`, which is not
a grasp-success or physical-plausibility metric.
Any improvement would reflect the whole fine-tuning recipe, including additional
optimization and the lower learning rate.

**Prepared launch.** The source is copied into
`validation/cleanctx_training_preflight_2026-09-14/source`; 69 files are hashed so
concurrent repository edits do not alter the reviewed training job. The launcher
checks those hashes, the resolved configuration, corpus inventory, exclusions,
checkpoint identity, initialization compatibility and output-directory state.

The dedicated script requests one `RTX_PRO_6000` GPU, eight CPUs, 64 GB RAM and
a 12-hour allocation. Its GPU resource name was checked against Slurm. It uses
the prepared local snapshot and dependencies, so keep the referenced `validation/`
directories available. Slurm logs go to the existing
`/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log` directory.

This command only checks the prepared run; it does not need a GPU allocation:

```bash
bash launch/legacy/train_pickup_uncond_cleanctx.sbatch --check
```

After confirming this preflight, submit with:

```bash
sbatch launch/legacy/train_pickup_uncond_cleanctx.sbatch
```

To continue an interrupted clean-context run:

```bash
sbatch launch/legacy/train_pickup_uncond_cleanctx.sbatch \
  --resume /data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond_cleanctx_from40k_0914/checkpoints/latest.pt
```

Do not pass the old noisy-context checkpoint to `--resume`. The script preserves
Slurm's GPU assignment and disables Wandb. Under an existing development
allocation, it can also be invoked with `srun ... bash launch/legacy/train_pickup_uncond_cleanctx.sbatch`,
pinning the selected GPU UUID in the `srun` environment. The declared budget and
source/model/data decisions above are the training preflight to be confirmed
before submission.

[Resolved configuration](/data/user_data/robertre/particle_world_model/validation/cleanctx_training_preflight_2026-09-14/resolved_config.yaml),
[machine-readable preflight](/data/user_data/robertre/particle_world_model/validation/cleanctx_training_preflight_2026-09-14/preflight.json),
[training inventory](/data/user_data/robertre/particle_world_model/validation/cleanctx_training_preflight_2026-09-14/train_shards.json),
[validation inventory](/data/user_data/robertre/particle_world_model/validation/cleanctx_training_preflight_2026-09-14/val_shards.json).
