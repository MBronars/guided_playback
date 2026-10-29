# Open issues

Recorded from the 2026-09-14 review of `robert/clean` at
`6cb6d1a50ec09c8df4b469b2f0a5ff2bfd02667e`. ISSUE-005 to ISSUE-007 come from the
afternoon follow-up review at `7c2aa5b`. All are resolved.

## ISSUE-001 — Pickup fine-tune launcher re-enables inconsistent noisy context

**Priority: high. Status: resolved 2026-09-14.**

Code: [launch/legacy/train_pickup_uncond_unet.sbatch](launch/legacy/train_pickup_uncond_unet.sbatch),
`train.noise_context=true` (line 101 at the reviewed revision).

The launcher overrides the repaired shared default (`false`) to preserve the
older 200K recipe. Context particle displacements are still corrupted
independently, while rigid pose geometry starts from the clean pose anchor.
The same body can therefore have conflicting coordinates inside the model.

Evidence: a forward-input capture on the saved real articulated pickup sample
`pickup_0_articulated.pt`, CPU FP32, independent-noise seed 883, measured
**146.504 mm maximum / 48.448 mm RMS** rigid-coordinate disagreement. Keeping
context clean with the same future/pose draw reduced this to **0.00212 mm
maximum / 0.00054 mm RMS**. These are input-geometry discrepancies, not trained
prediction errors.

Resolution: noisy context is retired, not repaired. No sampler, evaluator or
guidance path uses a noisy history, and the only context-prediction path
(`train.supervise_context`) was never used by a saved run.

- New configs refuse `train.noise_context=true` and `train.supervise_context=true`
  (`utils/config.py`).
- `train.light_context_noise_std` no longer jitters the rigid-body rows of
  pose-native batches (`utils/trajectory_objective.py`), so rigid history stays
  on its pose. Deformable history is jittered as before.
- The launcher trains with clean context; its default run name is now
  `pickup_jointed_unet_bidir_uncond_cleanctx`.
- Checkpoints trained with noisy context still load, sample and serve as
  `train.init_from`: both 200K pickup runs, the first unconditional fine-tune
  and ten older unified runs. Resuming them is refused.

Tests: `test_noisy_context_is_refused_for_new_runs`,
`test_noisy_context_checkpoints_still_initialize_fine_tunes_but_do_not_resume`,
`test_light_context_noise_leaves_rigid_history_on_its_pose` (the UNet's rigid
history positions within 1e-5 m of the pose; 68 mm before the change), and the
frozen-objective parity test, whose light-noise draws now match the historical
objective everywhere except rigid history.

## ISSUE-002 — Offline evaluation ignores saved custom dataset roots

**Priority: high. Status: resolved 2026-09-14.**

Code: [scripts/unified/eval_unified.py](scripts/unified/eval_unified.py),
`eval_domain`, `rollout_domain`, `evaluation_dataset`, and `_input_provenance`.

Dataset construction omits `cfg.data.roots`, including its custom exclusions.
Training forwards those settings, but offline evaluation falls back to
registered datasets unless an explicit CLI root override is supplied. It can
therefore score a different validation corpus from the one associated with
the run. Receipt construction likewise defaults to registered roots/exclusions.

Evidence: a config with custom MPM train/val directories reached the production
evaluation constructor with `roots=None`. Its receipt selected
`robert_mpm_v2_test_seen` instead of the saved validation root. The probe
intercepted dataset construction; it did not score an external corpus.

Resolution: `evaluation_dataset` in `scripts/unified/eval_unified.py` builds all
evaluation data.

- It resolves the checkpoint's saved `data.roots` with `resolve_data_roots` and passes
  them to `build_domain_dataset`, so a saved release, its exclusion lists and the MPM
  drag task follow the checkpoint.
- The fixed-window and rollout evaluators, guidance and
  `scripts/dev/check_checkpoint.py` all use it.
- `--mpm_val_root` and `--pickup_roots` still re-point registered domains, and are
  refused for a domain whose roots the checkpoint saved.
- Each receipt's `inputs` is read from the dataset that was scored and records
  `roots_source` (`registered`, `checkpoint data.roots` or `command line`).

No saved run used custom roots, so no earlier receipt was affected. Tests:
`test_evaluators_score_the_release_a_checkpoint_saved` and
`test_receipt_describes_the_release_that_was_scored`.

## ISSUE-003 — Resume accepts a change of corpus or exclusions

**Priority: high. Status: resolved 2026-09-14.**

Code: [utils/checkpoint.py](utils/checkpoint.py), `_continuation_contract`
and `validate_resume_configuration`.

The continuation contract omits `data.roots` and its exclusion selections.
A resumed run can change dataset releases or membership while retaining
optimizer, EMA, schedule and runtime state.

Evidence: the production validator accepted otherwise identical configs after
independently changing train/val roots and exclusion lists. This is separate
from the documented limitation that WebDataset iterator positions are not saved.

Resolution:

- The continuation contract includes the canonical `data.roots` selection
  (release paths, exclusion files, MPM task).
- `train.py` hashes every exclusion file the run applies once at startup and saves the
  hashes in each checkpoint (`corpus`, from `utils.checkpoint.corpus_identity`):
  training-split lists for the training domains, val-split lists for the validation
  domains, registered lists included.
- Resume recomputes them and refuses any difference, naming the split, domain and file.
- `data.unified_dataset.exclusion_files` is the one source of which lists a split applies;
  `build_domain_dataset` uses it too.
- Compatibility: checkpoints saved before this change have no hashes. They resume with a
  warning, and their roots are still compared through the saved config.
- `train.init_from` compares only the network, so a new experiment may use other data.
- Not covered: shards added to a release directory at the same path, and edits to the
  registered root constants in code.

Tests: `test_resume_refuses_another_release_or_exclusion_list`,
`test_resume_refuses_exclusion_files_edited_in_place` (a release's list and a registered
list), `test_checkpoints_without_exclusion_hashes_resume_with_a_warning`,
`test_initialization_accepts_a_source_trained_on_an_absent_release`, and
`test_checkpoints_record_exclusion_contents_and_resume_refuses_an_edit`, which runs the
training loop.

## ISSUE-004 — Native zero-shot evaluation supplies actions to unconditional models

**Priority: medium. Status: resolved 2026-09-14.**

Code: [scripts/dev/native_zero_shot.py](scripts/dev/native_zero_shot.py),
`sample_window`, `rollout`, and checkpoint handling in `main`.

The entrypoint never reads `train.action_dropout`: it supplies actions even
for checkpoints trained with dropout 1. The main `eval_unified` and
`native_eval_unified` entrypoints correctly select hidden-action evaluation.

Evidence: with a synthetic active-weight UNet, sampler seed 109 and two
requested pyramid steps, changing only the given pose actions changed native
zero-shot predictions by up to **8.136 mm**. The explicitly hidden sampler
path was exactly invariant. This demonstrates conditioning access, not the
quality of a trained unconditional checkpoint.

Resolution: `scripts/dev/native_zero_shot.py` reads the checkpoint's action policy with
`hides_given_actions`, the helper `eval_unified` and `native_eval_unified` use.

- A checkpoint trained with `train.action_dropout=1` is sampled with actions hidden in
  the native windows, every rollout window and the pickup control.
- Positions are built from the unhidden batch.
- `zero_shot.json` records `actions` (`hidden` or `given`).
- The pickup control is built with `evaluation_dataset`, so saved `data.roots` apply
  there too.

Tests: `test_native_zero_shot_follows_the_checkpoint_action_policy`, and
`test_native_zero_shot_window_cannot_read_hidden_actions`: with actions hidden the
prediction is bitwise unchanged when the given actions change; with actions given it
changes.

## ISSUE-005 — Overnight robustness trusted cloth fixture hashes

**Priority: medium. Status: resolved 2026-09-14.**

`scripts/dev/overnight_robustness.py` `fixtures` copied each cloth manifest entry's
`fixture_sha256` without hashing the file. A tensor replaced at the same path could
therefore reuse or extend an existing run receipt. The stability and temporal-error
lanes share this loader. The review re-hashed all 28 fixtures of the completed run,
and all of them match.

Resolution: `verified_fixture` loads a fixture only while its bytes match the recorded
hash. Test: `test_cloth_fixtures_load_only_while_their_bytes_match_the_manifest`.

## ISSUE-006 — Real-cloth drivers ignored the action policy

**Priority: medium. Status: resolved 2026-09-14.**

`scripts/dev/real_cloth_probe.py` and `scripts/dev/real_cloth_short_forecasts.py`
call `windowed_rollout` with the recorded robot motion given and clamped, bypassing the
policy in `Predictor.__call__`. The overnight checkpoint was action-conditioned, so its
results are unaffected.

Resolution: both drivers call `require_given_actions` right after reading the
checkpoint configuration. A checkpoint trained with every action hidden is refused
before any preparation. Tests: `test_given_action_protocols_refuse_checkpoints_that_hide_actions`
and `test_real_cloth_drivers_check_the_action_policy_before_any_work`. The second needs
SciPy, which the model environment lacks; it passes with the overnight lane's vendored
SciPy on `PYTHONPATH`.

## ISSUE-007 — Overnight report labels were hard-coded to the 40K UNet

**Priority: medium. Status: resolved 2026-09-14.**

`scripts/dev/package_overnight.py` wrote `unified_unet_xy_p99_40k/final.pt` into
`morning_status.json`. Its README, findings and montage text, and the figure titles in
`overnight_report.py`, describe that checkpoint whatever the lanes actually ran. The
completed outputs are correctly labeled.

Resolution: `reported_checkpoint` requires every run receipt under the output root to
name the checkpoint the text describes; its sha256 is pinned in
`observation_probe_common.py`. Both scripts refuse another checkpoint, or a mix, before
writing any label. `morning_status.json` records the path and hash from the receipts.
Tests: `test_reports_describe_only_the_checkpoint_their_text_was_written_for` and
`test_packaging_refuses_another_checkpoint_before_writing_labels`.

## Verification and remaining research

At the reviewed revision:

- `tests/`: **1,503 passed, 91 skipped**.
- Focused CPU/CUDA checks with an allocated RTX PRO 6000: **273 passed**.
- MPM collection environment: **221 tests run, 10 skipped**, successful.
- The broader model-environment run had two collection-test failures because
  optional SciPy was missing. Collection tests need their documented environment
  or consistent optional-dependency skips.

The earlier SO(3), checkpoint-window, geometry-precision and Transformer AdaLN
fixes remain present. Adaptive UNet allocation remains opt-in, with a rigid-object
ceiling of 128 and a scene cap of 1,024 **L1 particle tokens**. L0 particles,
pose tokens and registers are additional. Prediction-quality benefits and the
optimal rigid-token ceiling remain unverified.

Radius cutoffs, accumulated positional uncertainty, spatial RoPE robustness and
generalized boundary conditioning remain research questions. The differing
causal action access across backbones remains an accepted modeling choice.

The full review, probe source/results and test receipts are local artifacts under
`validation/repo_review_2026-09-14/`. That directory is ignored and is not included
in a fresh clone; the evidence summarized above is preserved here for discussion.
