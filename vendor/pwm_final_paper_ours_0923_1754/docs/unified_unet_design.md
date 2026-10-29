# Unified UNet — design (2026-08-26)

Bring `models/unet_std.py` (the 331M hierarchical particle UNet, trained MPM-only) to the
four-domain unified corpus, matching the Tier-2 v2 transformer's task so the two are
directly comparable under the protocol in `scripts/unified/eval_norm_arm.sh`.

User decisions (2026-08-26): generalize the layout (not resample), side pose-transformer,
train from scratch, all four domains, **visualizations at every stage on every dataset**.

## The four gaps (measured, `reports/unified/unet_probe/layout_probe.json`)

| # | gap | evidence |
|---|---|---|
| 1 | uniform per-object layout | `GraphProvider` asserts `N % points == 0` and every object exactly `points`=1024 contiguous. Measured: mpm 1024/1024/1024 ✓; pickup 512/671; cloth 12000/1000/1000; pushing 128/128 |
| 2 | isotropic delta stats | wrapper asserts `_dscale` all-equal; all four unified stats files are per-axis |
| 3 | no action conditioning | 10 input channels are delta/offset/z/physics; 3 of 4 domains are action-conditioned |
| 4 | no pose path | zero refs to `pose_native`/`body_type`/`pose_v_head`; pickup and pushing are pure-rigid batches |

## Resolutions

**1. Variable per-object counts — opt-in, pad-to-max + masks.**
New config flag `unet_variable_objects` (default **false** = today's exact fast path, so the
trained MPM model is bit-preserved and regression-testable). When true, each level is laid
out `[B, O * n_max_l, ...]` with a validity mask `[B, O * n_max_l]`, where
`n_max_l = max_o n_{l,o}` and `n_{l+1,o} = max(1, ceil(n_{l,o} / stride_l))`.
Padding to the per-level max keeps every index computation arithmetic
(`obj * n_max + local`), which is the part that is risky to change. Padding slots are
excluded from FPS (`lengths`), from the interp kNN (`lengths2`), and from attention
(additive `attn_mask`), and are zeroed after each pool/unpool.

**2. Stats** — the arm-B isotropic set (`unified://iso_p9999`), which satisfies the assert.
Its MPM scale (132.2 mm) is within 1 % of the unet's own MPM normalizer (133.2 mm).
Note the A/B measured arm B as 6–24 % *worse* than the current files **for the transformer**;
the unet's RoPE geometry currency (`pos / r`, `r = delta_scale/2`) requires isotropy, so this
is a constraint, not a free choice. Record it as a confound in any unet-vs-v2 comparison.

**3. Actions** — extend the input channels 10 → 14: `[action delta (3), action mask (1)]`,
normalized by the same delta stats as the evidence channel. Zero-init the stem's new columns.

**4. Pose** — a side transformer over the unet's bottleneck tokens, porting v2's proven parts
(`pose_token_embed`, `pose_action_skip`, token-type embedding, zero-init `pose_v_head`):
per frame, pose tokens cross-attend to the bottleneck, then temporal self-attention, then the
head. Rigid rows still enter the particle trunk as geometry reconstructed from the noisy pose
(v2's self-consistency loop), so the trunk sees one uniform particle scene. Pure-rigid
batches (pickup, pushing) return `(None, pose_v)` exactly as v2 does.

## Stages, each gated by a visualization on all four domains

| stage | deliverable | visual check |
|---|---|---|
| 1 | `GraphProvider` variable-count + masks | FPS ladder L0→L3 per domain, coloured by object; padding never selected; per-object anchor counts |
| 2 | masked attention, Pool, Unpool | pool→unpool round-trip of a known field; error vs the equal-size path on mpm must be ~0 |
| 3 | forward smoke | output shapes, finite, gradients reach the stem, on every domain |
| 4 | action channels | action field overlay; zero-init ⇒ step-0 output unchanged |
| 5 | pose side-transformer | pose rollout vs GT (pickup/pushing), gripper + object |
| 6 | overfit one batch per domain | loss ↓, predicted vs GT point clouds |
| 7 | pre-flight + 2/4-GPU launch | (user-gated) |

**Regression gate, every stage:** with `unet_variable_objects=false` the MPM path must stay
bit-identical to `ecd90ba` — checked by running the trained checkpoint's forward on a fixed
MPM batch and comparing outputs exactly.
