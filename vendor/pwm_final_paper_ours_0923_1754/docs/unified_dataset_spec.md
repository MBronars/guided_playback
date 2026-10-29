# Unified Dataset v1 — Specification

Phase-1 artifact of the unified-model effort. Every numeric parameter here is sourced from the
Phase-0 measurements in `reports/unified/audit_report.json` (+ the supplementary scan logged in
that session), not from configs. Corpus roster locked 2026-08-05: **MPM dropping (passive),
pickup ×4 success, cloth_curated_med, pushing_actions**. Drag is excluded (point-force actions
would add a second action modality); with it gone, every action in the union is *actuator-body
motion*, one modality.

## 1. Corpora and splits

| corpus | root | train/eval split (preserved, not redefined) |
|---|---|---|
| mpm | `…/robert_mpm_v2/webdataset_full49/{1..4}obj` | train tars only; `test_seen`/`test_unseen(_clean)`/`ood` live in sibling dirs; 5–7 obj = OOD |
| pickup | `…/robert_pickup_data_{place,convex,pick,upright}_success` | val = last 10% sorted shards per collection; **asset-holdout blocklists stay applied** (train excludes the 142 held-out assets) |
| cloth | `…/robert_sim1/cloth_curated_med` (1,291 eps = 1,147 + 144) | val = the 38 held-out garments (`val_files` in cloth_fold_v10.yaml); `cloth_curated_{small,large}` = OOD bands |
| pushing | `…/VLA/3DPCWM/pushing_data_actions` (404 tars) | no prior split → **new: last 10% of sorted tars = val** (deterministic, matches pickup convention) |

## 2. Canonical frame

Per-domain **fixed rigid translation** (never per-scene centering — MPM wall geometry must stay
knowable; never any scaling — E and gravity are metric):

| domain | translation t (m) | provenance |
|---|---|---|
| mpm | (−0.75, −0.75, −0.10) | xy = exact domain center (walls at 0 and 1.5); z = measured floor 0.107→0.10 |
| pickup | (−0.45, 0.00, −0.10) | measured workspace center (0.458, −0.035); table 0.111→0.10 |
| pushing | (0, 0, 0) | already centered, plane at z≈0.012 |
| cloth | (−0.65, 0.00, −0.80) | measured center (0.662, 0.032); table top 0.80 (cloth rests 0.808) |

Invariant after transform: **support plane at z = 0 in every domain** (lowest sampled points rest
0.5–1.1 cm above it — a consistent particle-radius signature, verified per-domain).

**Shared metric position scale S = 0.8 m**: `p_canon = (p + t) / S`. Measured worst cases fit:
MPM wall half-extent 0.75→0.94, cloth x 0.62→0.78, pickup y 0.57→0.71, z max 0.75→0.94.
Implementation is zero-new-code: a position-stats file with `min = −S, max = +S` on all axes makes
the existing min-max `NormalizationStats` compute exactly `p/S`. **Deltas keep per-domain robust
min-max stats** (measured medians span 0.09 mm/frame (cloth) to 10.4 (pickup) — a global delta
scale would crush cloth precision).

## 3. Sample schema (TrajectoryBatch-compatible + sidecar)

Core (what the particle model trains on):
```
particle_pos   [T, N, 3] f32   canonical frame
obj_ids        [N] i64          scene-local body index 0..K-1 (REMAPPED — see §4)
valid_mask     [N] bool
actions        [T-1, N, 3] f32  per-particle displacement field (§5); zeros where passive
action_mask    [N] bool         particles of actuated bodies
materials      [N, 5] f32       [log10(E), ν, ρ/1000, μ, yield]  (§6)
mat_type       [N] i64          index into §6 vocabulary
```
Sidecar (hybrid head + eval; never consumed by the pure-particle model):
```
body_type      [K] enum         deformable_volumetric | cloth_shell | rigid | gripper
body_actuated  [K] bool
body_poses     [T, K, 4, 4]     where defined (pickup both bodies, cloth grippers; else identity)
templates / template_mask / finger_open / link metadata   (pickup + cloth grippers)
dt             scalar           §8
provenance     source corpus, shard, key, elem, transform id, adapter version
```

## 4. Body taxonomy — magic ids are retired

The three raw conventions collide (pickup 0=object/1=gripper; pushing 0=block/1=rod; MPM 0..K−1;
cloth has no ids at all). Adapters remap to scene-local 0..K−1 and emit explicit `body_type` +
`body_actuated`. Type and actuation are orthogonal (the pushing rod is `rigid` AND actuated).
Nothing downstream may branch on raw id values.

## 5. Action field

`actions[t, i] = pos[t+1, i] − pos[t, i]` for particles of actuated bodies; zero elsewhere.
Verified equivalent to each corpus's native action encoding (Phase-0, MAE = 0.0 in all three):
pickup `absolute_targets ≡ pos[t]` (achieved motion); cloth `delta_*_ee ≡ pose diff` (lag 0);
pushing `absolute_targets ≡ pos[t+1]`. MPM: no actuated bodies, field all-zero (passivity
verified: drag keys identically 0).

⚠ **Pushing label-leak trap**: `absolute_targets` is non-NaN for the *block* too, and equals the
block's future GT. Actions must be derived ONLY from rod particles (`body_actuated`); block
targets are never converted. A verification check asserts `actions == 0` on all non-actuated
particles in every corpus.

## 6. Materials

Slots `[log10(E), ν, ρ/1000, μ, yield]` + `mat_type` embedding. Measured ranges and fills:

| domain | E | ν | ρ | μ | yield | mat_type |
|---|---|---|---|---|---|---|
| mpm | 1.3e4–8.6e6 (real) | .05–.45 | 505–2992 | fill 0.5* | 0–1.99e4 (real) | `mpm_t{0,1,2}` (pass through raw col-3; semantics TBD from Genesis provenance) |
| pickup object | 1e9 (real) | 0.30 | 35–6941 (real) | .81–1.10 (real) | 0 | `rigid` |
| pickup gripper | 1e9 | 0.30 | sidecar | sidecar | 0 | `gripper` |
| pushing (both) | fill 1e9 | fill 0.30 | 500–1000 (real) | fill 1.0 | 0 | `rigid` / rod `rigid`+actuated |
| cloth | fill sentinel | fill sentinel | fill sentinel | fill sentinel | 0 | `cloth_shell` / `gripper` |

\*mpm μ is not recorded; filled with a constant, disambiguated by mat_type. Fills are constants,
documented here, and flagged per-particle by mat_type — the model learns regime from the
*combination*, with E genuinely spanning 1e4→1e9 across the union.

## 7. Windows and stride

C=2 everywhere. Native F per domain (batches are domain-pure, so variable F is free):
pickup F=38 (T=40); mpm F=16 (T=49, matches specialist eval protocol); cloth F=16 at
temporal_stride=2; pushing F=16 (T=40; provisional — align with the pushing specialist config in
Phase 2 if it differs).

## 8. dt registry

mpm **1/24 s (measured** by gravity fit; the |Δp|/|v| estimator is biased high under acceleration
— do not use it); pickup 0.05 s (documented: 0.002 × 25); cloth 1/30 s at stride 2 (fps=60
stored in-file); **pushing UNKNOWN** — no provenance in the dataset dir; rod moves a uniform
~3.1 mm/frame (scripted constant velocity). Open item; non-blocking (nothing consumes dt yet).

## 9. Spacing policy

Measured p50 bands (mm): mpm 5.0 · pickup object 2.7–6.6 (corr 0.83 with object size — the
fixed-512-count artifact) · pickup gripper 3.9 · pushing 9.1 / 5.4 · cloth in-plane ~9–11.5
(cross-layer tail 3.9) · cloth EE 3.3. Actuator *density* is nearly uniform (3.3–3.9) despite
count differences (671 vs 1000/gripper). **Decision: accept the 3–11.5 band, no resampling** —
equalizing would gut small pickup objects, KNN is count-based, per-domain stats absorb scale.
Revisit only if unified training shows cross-domain contact failures.

## 10. Spatial RoPE base under the shared scale

Legacy formula (`RotaryPositionEncoding3D`, `max_period` = `rope_spatial_base`):
`freq_k = base^(−k/10)`, k=0..9 → base 0.1 puts the coarse/fine crossover (5 of 10 pairs) at
wavelength = 2.0 normalized units — correct when data fills [−1,1], which per-domain min-max
guaranteed. Under the shared scale the typical occupied span is ~1.1–1.5 units (pushing 0.7,
pickup 1.1, cloth 1.55, mpm ≤1.9). Preserving the crossover rule at span ≈ 1.2:
`base = (span/2π)² ≈ 0.04`. **rope_spatial_base = 0.04 for unified training** (validated in the
Phase-3 overfit smoke; explicit banded override `rope_f_min/f_max` exists as fallback).

## 11. Mixing and storage

Domain-pure batches, interleaved; default sampling weights **equal quarters** (raw frame counts
2.14 M pickup / 561 K mpm / ~600 K cloth / pushing TBD would otherwise let pickup dominate).
Static-fraction imbalance (cloth 79 % particle-level vs pickup 11 %) is a *training*-side concern,
recorded here, decided there. Storage: cloth is materialized npz→WDS (multi-worker npz over NFS
thrashes — measured before); the three WDS corpora stay in place behind adapters. Every adapter
records its transform + version in provenance; the Phase-4 manifest freezes exact counts.

## 12. Open items

1. pushing dt (provenance hunt / ask the collector).
2. MPM mat-type column semantics ({0,1,2} — Genesis provenance).
3. pushing F (confirm against its specialist config).
4. Phase-3 must include: full-pass in-bounds check under §2 transforms (sampled bounds ≠ proof),
   the §5 zero-action assertion, and floor-at-0 verification per domain.
