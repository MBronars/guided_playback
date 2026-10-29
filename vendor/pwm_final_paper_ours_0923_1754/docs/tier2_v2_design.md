# Tier-2 v2 — Pose-Native Routing (design, locked 2026-08-07)

One unified model over four regimes that keeps Tier-1's uniform particle trunk and makes
rigid-body pose a **first-class diffusion variable** (v9's principle, generalized), with
pose-space action conditioning (v8's channel) — deleting the aux-loss machinery whose failure
modes consumed Tier-2 v1.

## Why v1 failed, in one line each
- Poses were a teacher-forced auxiliary readout → a per-frame mean bias was invisible to
  training (70 mm object drift, sinking block) — unrepresentable when pose error IS the main loss.
- The 3-D mean-pooled action skip cancels rotation → gripper 5.1° vs v9's 0.48°.
- Patching supervision through aux terms (Huber, then drift) fought the optimizer from scratch
  (shortcut attractor) — main-loss supervision needs no patches.

## The architecture

### Diffusion variable (the core change)
Per domain-pure batch, the denoised state is TWO jointly-noised channel groups sharing one
per-frame DF timestep schedule:

| group | shape | normalization | supervised by |
|---|---|---|---|
| deformable particle deltas | [B, F, N_def, 3] | per-domain delta stats (as today) | main FM v-loss |
| rigid-body pose deltas [Δt(3), Δaa(3)] | [B, F, K_rigid, 6] | **new per-domain pose-delta stats** (sidecar-derived, symmetric min-max) | main FM v-loss |

- **No open channel**: articulation stays emergent from per-link bodies (v1's proven choice).
- **No auxiliary losses**: the Huber aux, drift aux, rigid-row masking, x0→v rigid conversion,
  and BODY_POSE_*_SCALE constants are all deleted. `loss = mean(particle v-err) +
  λ_pose · mean(pose v-err)`, λ_pose = 1 (gate G1 revisits).
- Rigid particle rows are **removed from the output variable** — poses replace them.

### Input side (uniform encoding, unchanged in spirit)
- Deformable rows enter exactly as today.
- Rigid rows enter as particles **reconstructed from the current noisy pose**:
  `templates @ integrate(noisy pose deltas from the C−1 anchor)` — v9's self-consistency loop;
  geometry sharpens with denoising. They carry mat_type / physics / per-particle actions through
  the same embedding as everything else, so the trunk still sees one uniform particle scene.
- mpm batches have no pose channels → the pose machinery is a no-op (identical to Tier-1).

### Pose tokens (replace v1's mean-pool readout entirely)
Per rigid body per frame, a token = embed(normalized noisy pose delta) + template-geometry
summary embed + token-type embedding (v9 pattern), spatial RoPE at the body center from the
integrated noisy pose. Pose tokens join temporal attention beside the particle latents.
Readout: pose token → linear head → pose-delta v-prediction. (The pooled-feature head is gone;
the token IS the dedicated computation.)

### Actions (both worlds)
- Per-particle action field: kept unchanged (the union modality, trunk-level).
- **NEW pose-space action skip**: per actuated body, the commanded pose delta derived from the
  sidecar poses (achieved == commanded, audited), normalized by pose stats, added to that body's
  pose token. Passive bodies get a learned no_action_token. Rotation and articulation intent are
  now directly visible to the pose pathway.

### Inference / AR rollout
DF-pyramid over both channel groups. AR rollout carries predicted poses forward directly as the
next window's anchor (no Kabsch re-anchoring needed — the v1 harness's Kabsch path remains only
for evaluating Tier-1). Later, nearly free: per-BODY noise schedules via the existing
`per_object_noise` machinery (actuator-leads / body inpainting) — a capability no specialist had.

## Provenance of every piece
| piece | from | status |
|---|---|---|
| uniform particle trunk, union schema, templates, per-link articulation | Tier-1 / v1 | kept |
| pose as diffused main-loss variable; pose tokens; geometry-from-noisy-pose | pickup v9 | recovered |
| pose-space action conditioning | cloth v8 | recovered |
| hybrid particle/rigid routing | cloth v8 → v1 | kept (moved to variable level) |
| FK, open channel, dataset one-hots, aux losses | — | deliberately out |

## Implementation plan — staged, each stage with its own tests

### S0 · Per-domain pose-delta stats (~0.5 d)
`scripts/unified/compute_pose_delta_stats.py`: stream train-split sidecars (pickup / cloth /
pushing), collect per-frame `[Δt, Δaa]` per rigid body, symmetric robust min-max (p99.99, the
house convention) → `stats/unified_v1_pose_delta_{domain}.npz` (6 channels).
- **T0.1** ranges vs known physics: pickup Δt max ≈ its particle-delta stats (±30–37 mm),
  pushing rod ≈ 3.1 mm/frame + block ≈ 0, cloth links ≈ 5–7 mm; rot ranges eyeballed vs audit.
- **T0.2** normalize/denormalize round-trip identity; `normalize(0) = 0` (symmetry).
- **T0.3** clip rate < 0.1 % on a train sample.

### S1 · Batch plumbing (~0.5 d)
Adapters (at `emit_sidecar`) add: `pose_deltas_gt [B,F,K,6]` (dt + `so3_log(dR)` from sidecar
poses — the aux-GT math, now a first-class target), `pose_delta_stats`, `pose_actions
[B,F,K,6]` (= gt deltas of actuated bodies; passive → zeros + mask). Carried through
`to()`/collate.
- **T1.1** integrate `pose_deltas_gt` from the C−1 anchor → reproduces `body_poses` < 1e-5.
- **T1.2** `.to(device)`/collate carry-through; domain-pure shape consistency (the
  batch.domain lesson).
- **T1.3** loader determinism (same seed → identical tensors).

### S2 · Diffusion ops on the pose group (~0.1 d)
`TrajectoryFlowMatching` ops are shape-agnostic; verify `[B,F]` timesteps broadcast to
`[B,F,K,6]` (the `while t.ndim < 4` pattern fits).
- **T2.1** add_noise → x0-recovery identity on pose shapes at several t.
- **T2.2** the per-frame DF scheduling matrix applies unchanged (pose group shares the frame
  schedule).

### S3 · Model (~1 d) — `pose_native: bool` flag; v1's `predict_body_poses` left dormant for
the ablation record
New modules: pose-token embed (Linear(6,d)+MLP), pose-action skip (Linear(6,d) +
no_action_token), template-geometry summary embed (FPS-16 of template → small MLP, once per
body), token-type embed, zero-init pose head Linear(d,6).
Forward: (a) build the full per-particle noisy scene by writing
`templates @ integrate(denorm(x_t_pose), anchor)`-derived deltas into rigid rows → uniform
encoding unchanged; (b) K pose tokens per frame (noisy-Δ embed + geom + action-skip + type),
RoPE at integrated body centers (detached), appended to the latent set in spatial-KNN and
temporal attention; (c) readout: pose tokens → head → `[B,F,K,6]` v-pred; particle decode runs
only if N_def > 0; (d) returns `(particle_v, pose_v)`.
- **T3.1** forward smoke all 4 domains; **mpm bit-regression**: with K=0 the v2 model's output
  must equal the Tier-1 path exactly (same weights, same seeds).
- **T3.2** GT-round-trip (the v1 audit lesson, applied *before* training this time): stub the
  pose head to emit GT → full DF inference → rigid rows vs GT < 0.01 mm.
- **T3.3** gradient-flow probe: pose v-loss reaches trunk + every new module (per-module
  grad-norm > 0).
- **T3.4** step-time: pickup/pushing faster than v1 (no discarded particle decode).

### S4 · train.py (~0.5 d)
Loss = `mean(particle v-err over valid deformable) + λ_pose · mean(pose v-err)`, λ=1;
empty-group guards both ways (mpm: no pose term; pickup/pushing: no particle term). Same
noising path, same context noising, same spike guards. `validate()` mirrored; per-domain keys
unchanged + a blended `loss_pose` key. v1's aux blocks and rigid masking not executed under
`pose_native`.
- **T4.1** train_step unit per domain: finite, both terms logged, guards exercised.
- **T4.2 = G1** overfit gate (1.5 K steps, from scratch, production train_step): pushing +
  pickup pose v-loss → ~0; cloth both terms ↓; mpm trajectory matches a Tier-1 control run
  (same seed).
- **T4.3** teacher-forced bias probe at overfit end: per-body bias ≈ 0 (< 0.1 mm/frame).

### S5 · Inference + eval (~0.5 d)
`run_diffusion_inference_df` carries `(x_t_particles, x_t_pose)`; Euler updates both under one
schedule; outputs → `predictions_to_positions` (deformable) + integrate → `template @ pose`
(rigid). AR rollout: predicted poses ARE the next anchor (no Kabsch). `eval_unified` v2 branch
reads poses directly for °/mm (Kabsch path retained for Tier-1 rows).
- **T5.1** inference smoke ×4 domains.
- **T5.2** exactness: the DF chain's final pose x0 equals the head's t→0 prediction (Euler
  lands exactly — the v1 property, re-proven).
- **T5.3** rollout smoke + viewer dump renders sanely.

### S6 · Pre-launch gates (wall-clock ~3 h) — ALL must pass before a chain leg is released
- **G1** = T4.2.
- **G2 from-scratch 5 K mixed mini-run** (1 GPU, bf16, production config): pre-committed pass
  criteria: (a) pickup val descending through 5 K (no 3-consecutive-round rise), (b)
  teacher-forced object bias < 0.5 mm/frame @5 K, (c) spike-skip rate < 0.5 %, (d) mpm val
  within noise of a Tier-1 control (shared-trunk no-regression).
- **G3 DDP-2 smoke** 400 steps: 2 val rounds, reduce alignment, and a save→resume cycle.
- **G4 guard-interaction check** (the blended-EMA lesson): confirm healthy pure-rigid batch
  losses sit < 10× the blended train EMA; if not, decide per-domain spike EMA *before* launch,
  not mid-flight.
Launch = release held leg 9829813 under `configs/unified_v1_tier2v2.yaml`; bias audit joins the
every-10K snapshot eval pair; intervention points pre-committed at 10 K (pickup val + bias
trajectory).

**Total ≈ 3 focused days.** Known risks, each with its assigned measurement: λ balance (G1/G2),
rotation-chart noising (v9 precedent; G1), spike-guard scale mismatch (G4), pose-stats tails
(T0.3).
