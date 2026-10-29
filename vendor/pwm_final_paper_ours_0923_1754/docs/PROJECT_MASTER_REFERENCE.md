# Particle World Model — Master Reference

> Historical pre-cleanup snapshot. For the supported clean checkout, see
> [SUPPORTED.md](SUPPORTED.md). The cleanup record (`docs/cleanup/`) is in git history, last present at `79a3c84`.
> Historical source and report paths refer to the frozen reference branch.
> Round four narrows active scope to the unified Transformer/UNet. The user has
> reopened the normalization decision: both per-domain20 and global4 are supported.
> The historical conclusion below that normalization is "settled" is not the
> current experimental policy.

**State of the project as of 2026-09-04**, branch `robert/unified-unet-combined` at `7091f3a`. Written as the basis for a full codebase cleanup. Every code claim carries a `file:line` pointer into that commit; every number carries its receipt (a JSON under `reports/`, a checkpoint, or a memory note). Items that could not be verified against code or disk are marked (unverified).

How this document was produced: six independent read-only passes over the code, the configs, the tracked reports, the SLURM and storage state, and the project memory notes, each returning facts with pointers; the cross-cutting findings below were then re-checked by hand. The per-file inventory tables behind chapter 6 (`docs/cleanup_2026-09/*.tsv`) are in git history, last present at `79a3c84`.

## Contents

1. Data sources and generation (MPM, SIM1 cloth, pickup, pushing; formats, vintages, defects, blocklists, splits)
2. Normalization (mechanics, every stats set, how each was computed, every result)
3. Model architectures, diffusion setup and samplers
4. Unified dataset adapter, training recipes, evaluation protocols, checkpoints
5. Infrastructure, environments and operational pitfalls
6. Codebase inventory and cleanup map
7. Consolidated action list and open decisions

## What matters most (read this first)

1. **The pickup blocklists have not been applied to any unified run since 2026-08-26.** All 8,506 entries in the phantom blocklist and the asset-holdout blocklist store shard paths under `/data/user_data/robertre/Genesis/3DPCWM_Data/...`; since `GENESIS = GROUP` the loader's shard URLs are `/data/group_data/katefgroup/VLA/3DPCWM/...`, and `_blocklist_filter` compares strings without resolving the symlink, so nothing matches (`data/unified_dataset.py:113-151`, `:600-611`). Every unified run launched after that date (norm arms, the UNet xy_p99 40K baseline that is now the shared handoff, the joint overfits, the staged 200K pair) trained on the 142 held-out pickup assets and the 299 phantom windows, and unified pickup validation is not asset-matched. The 130K rebuttal model predates the switch. The mpm and cloth lists ship both path forms and do match. Fix: normalize prefixes in `_load_blocklists` (or ship both forms) and add a shipped-list match test like `tests/test_episode_blocklists.py:48-60`. Do this before the 200K launch.
2. **Normalization is settled: per-domain `unified://xy_p99` (20 numbers).** It is the only set that beat the v1 files on trained rollouts (17/17 cells, UNet 40K). Every shared-scale scheme lost: global linear C (cloth 6.3×, pushing 3.4× worse at 40K), `unified_asinh` (0/17), and the 4-number `xy_p99_global` joint overfit (2026-09-04: pushing 5–11×, cloth ~4×, pickup 1.3–2.5× worse on both architectures; mpm unchanged). The default `per-domain` set is not one recipe (asymmetric 0.5/99.5 cloth file, unrecorded pushing protocol) and 3 of its 7 files are untracked in git; the group directory is the canonical copy.
3. **Checkpoint hygiene.** Three `final.pt` files were overwritten by smoke runs (`robert_mpm_object_v2_c2` step 3, `robert_mpm_joint_v6_1_c8` step 2, `cloth_fold_v10` step 6); their `best.pt` are the real models. The frozen rebuttal checkpoint `unified_v1_tier2v2/step_130000.pt` no longer exists (snapshots pruned 2026-08-31); only its JSON receipts remain. No `step_*.pt` snapshots survive for any unified run.
4. **The handoff README overstates the eval script.** `scripts/unified/eval_unified.py` uses the windowed diffusion-forcing sampler for every architecture; it does not switch to streaming for UNets (streaming lives in `scripts/eval_object_rollout_unet.py` and `eval_paper_mpm_unified.py --streaming`). Corrected in this commit in both the repo copy and the shared directory.
5. **Git does not hold what a fresh clone needs.** `CLAUDE.md` is gitignored and untracked despite its own header; `docs/` is an ignored directory with 4 force-added files and 22 local-only docs; several stats files required by live configs and by the deployed checkpoints' `config.yaml` are untracked (`pushing_actions_stats.npz`, `robert_mpm_v2_object{,_6key}_p9999_stats.npz`, `cloth_curated_med_stride2_stats.npz`, `gripper_template.pt`, ...). A `git clean -x` would delete them.
6. **Storage is at the edge.** The group VLA volume that holds all unified data and every handoff is at 100 % (184 GB free of 23 TB); `/data/user_data/robertre` is at 91 % with `runs/` alone at 313 GB (94 run dirs) and `inference_results/` at 31 GB.
7. **A third of the launch files cannot run from this account.** 82 of 119 `sbatch_scripts/` write logs to and `cd` into a colleague's home directory; 95 launchers hardcode this checkout's absolute path; `fastrin_test` (L40S-only pytorch3d) is the env in 40 of them.
8. **Whole tiers are dead weight.** The VAE/latent tier (last content commit 2026-03-08), `models/unet.py` with its 21 configs, the pushing-era guidance stack (unaware of pose-native and UNet), ~60 joint-transformer sweep configs with no checkpoints, 65 scripts unreferenced since before July, and a set of `ModelConfig` flags enabled in zero configs (`per_object_noise`, `use_windowed_temporal`, `predict_body_poses`, `banded_rope`, ...).
9. **Doc/code drift to fix in place.** README lists a `context_adaln` flag that does not exist and wrong defaults for `all_relative`/`bypass_compression`; CLAUDE.md cites archived sweep paths and the superseded "sample mode is always v-loss" rule (since 2026-07-26 sample mode trains in x0 space, `train.py:1997-2006`); `docs/scope_unified_unet.md` is cited 21× in code and never existed; `DOMAIN_DT["pushing"]` is `None` (real rate ≈450–500 fps); `train.spike_grace_steps` is a dead key on `train.py`; `domain_weights` is plumbed in `dataset.py` but unreachable from `train.py`.
10. **The 200K pair is staged, not launched**, and its pre-flight has seven open decisions (chapter 4 §6), now eight with the pickup blocklist fix.

---

# 1. Data sources and generation

### 0. Shared plumbing (unified v1 loader)

All four domains reach training through `data/unified_dataset.py` (`format: unified`, dispatched at `data/dataset.py:1791`). Constants that are **decisions baked into code**, not configs:

- Roots: `GROUP = /data/group_data/katefgroup/VLA/3DPCWM` (`:595`), `GENESIS = GROUP` since 2026-08-26 (`:600`; the `/data/user_data/robertre/Genesis/3DPCWM_Data` tree is a symlink onto GROUP). `MPM_TRAIN_ROOT`/`MPM_VAL_ROOT` `:630-631`, `PUSHING_ROOT` `:632`, `CLOTH_WDS_ROOT` `:633`, `PICKUP_ROOTS` `:602-603`, stats `STATS_DIR/STATS` `:636-642`, stats sets `:657`.
- Canonical frame: fixed per-domain translation `CANONICAL_TRANSLATION` (`:35-40`), shared scale `UNIFIED_SCALE = 0.8` m (`:33`); support plane at z=0 everywhere (spec `docs/unified_dataset_spec.md:19-39`).
- `DOMAIN_DT = {mpm 1/24, pickup 0.05, cloth 1/30, pushing None}` (`:54`) — nothing consumes it. `DOMAIN_FUTURE_LEN = {mpm 16, pickup 38, cloth 16, pushing 16}` (`:692`), C=2; configs may override to a uniform window (e.g. `configs/unified_unet_norm_xy_p99.yaml:43-44` uses C=4/F=16).
- Split: `_split_shards` = last 10 % of sorted shards per root (`:74-81`); shards containing `_quarantine` are skipped (`:270`).
- Material vocabulary: MPM raw col-3 `{0,1,2}` passed through; `MAT_RIGID=3, MAT_GRIPPER=4, MAT_CLOTH=5` (`:45-47`); fills `RIGID_FILL_E=1e9/ν=0.30`, `CLOTH_FILL_E=1e5/ν=0.30/ρ=300` (`:51-52`).
- Blocklist mechanism: `_load_blocklists` (`:91-110`) reads JSON `{"blocklist":[{shard, key, elem?}]}` into `(shard,key) -> set(elem) | None`; `_blocklist_filter` (`:113-151`) matches the sample's absolute `__url__` first, then each root-relative form, and drops whole samples (`elem` absent) or rows of the pre-batched B=4 slab. Unit tests: `tests/test_episode_blocklists.py` (mpm + cloth shipped lists only).

### 1. MPM (deformable multi-object dropping + drag)

**Generator.** `/data/user_data/robertre/mpm_simulation` — custom Warp/Taichi explicit MPM (`generate_mpm_data_multi.py:18-19`), ObjaverseXL meshes, Z-up, gravity −9.8. Per object: 1024 particles via FPS of 20× surface oversampling, size 0.06–0.10 m, materials `MATERIAL_TYPES = ['elastic','plasticine','sand']` → `mat_type` 0/1/2 (`generate_mpm_data_multi.py:37`; yield stress 0 / U(5e3,2e4) / 0, `:520-528`), E ∈ [1e4,1e7] drawn uniformly in log-space windows, ν ∈ [0.05,0.45], ρ ∈ [500,3000], 5 scenarios (random/collision/scatter/stack/drop_on), 48 frames + initial at ~24 fps, substep 1e-4 (417/frame), floor friction 0.3 (`mpm_simulation/docs/dataset_v2.md:33-37`). This answers the spec's open item "mat-type semantics TBD" (`docs/unified_dataset_spec.md:89,144`).

**Vintages.**
- `robert_mpm` (**v1**, paper Table-1 training set): `webdataset/` 910 tars (24-frame materialization), `webdataset_full49/` 414 tars; test = `robert_mpm_test/webdataset_full49` 10 tars. Known v1 defects: ~6 % of E draws piled on exactly 1e4/1e7 by `np.clip`, 4.3 % multi-part objects, OOD split shared 12.5 % byte-identical clouds with train (`robert_mpm_v2/fixes_report/README.md` §1-4).
- `robert_mpm_v2` (**current**): box 1.5 m / 96³ grid, disjoint asset pools (4,726 train / 525 holdout UIDs), unique seed strings; **passive dropping only** (`drag_active`/`is_grabbed`/`grab_velocity` present but all-zero — verified on `2obj/00000.tar`). Train `webdataset_full49/{1..4}obj` = **208 tars / 6,617 clean clips / 19 G** (wall-contact scenes removed: 2,654 skipped; `_wallincl` variant 290 tars / 9,228 clips keeps them); `test_seen` 10 tars/270 clips (0.7 G), `test_unseen` 11/286 (0.7 G), `ood` (5–7 obj, held-out assets) 17/492 (3.1 G). Derived views: `robert_mpm_v2_val_unseen_fulltar` (00000.tar of each count = 128 eps, selects best.pt for specialists), `robert_mpm_v2_test_unseen_00001plus` (remaining 158 eps), `_test_unseen_clean`, `_viz_pinned`. Provenance: `robert_mpm_v2/provenance_report_v2.json` (0 shared scenes/clouds/draws; 906 shared UIDs with test_seen by design, 0 with unseen/ood).
- Drag: `robert_mpm_drag_setA/webdataset/{1..4}obj` 337 tars / 15 G (~7.1K trajs, T=24, grab from t=0, 16- or 4-frame drag) — loader `MPMDragDataset` (`data/dataset.py:953`), actions = `grab_velocity × SIM_DT(1/24)` (`:983,1018-1020`; without it 97 % of actions clipped). `robert_mpm_v2_drag/webdataset_full49` **1,318 tars** (+ drag ood/test_seen/test_unseen 6/16/14) — unused by any model.

**On-disk format** (pre-batched, 8 samples/shard × B=4): `particle_pos [4,49,N,3] f32` (N = 1024 × n_obj, coords in [0,1.5]), `particle_vel`, `initial_vel`, `valid_particles [4,49,N] bool` (constant, no padding), `material_properties [4,N,5]` = (E, ν, ρ, type, yield), `object_ids [4,N]`, `n_objects`, plus inert drag keys. Loader `MPMMultiObjectDataset` (`data/dataset.py:831-950`) reads cols 0-3 (docstring says 4 cols — stale, `:838`), random 49→C+F sub-window; `UnifiedMPMDataset` (`unified_dataset.py:431-475`) adds the shift and pins eval windows by `crc32(__key__)` (`:466`).

### 2. SIM1 cloth (bimanual folding)

**Generator.** `/data/user_data/robertre/sim1` (InternRobotics SIM1 stack: Newton/Warp XPBD cloth, ARX5 bimanual, replay fps 60, table top z=0.80). Pipeline: garment mesh → `scripts/ingest_garment.py` (component/triangulate at 11.5 mm edge/orient) → `scripts/generate_fold_tasks.py` (half/side bimanual, diag single-arm, warm-start IK; `generate_ik_tasks.py` for lift_random) → replay → `scripts/export_pointcloud_dataset.py` → `scripts/qualify_garment.py`. Garment source history: SIM1 builtins (3 garments; `sim1_cloth_variant_0002`, 91 trajs, and `sim1_cloth_diverse_v1`, 1,245 trajs/7 categories: bimanual 150, single_arm 201, ik_lift_random 200, ik_push 200, ik_tap 200, primitive_chain 190, primitive_menu 104) → the HF **ClothTransformer** dataset attempt (1,056 sims; rest states stored draped/crumpled → settle into blobs; the 168-fold "deep" set was abandoned) → **Zenodo "Dataset of 3D Garments with Sewing Patterns"** (Korosteleva 2021, 23.5K garments): panels triangulated from the 2-D pattern, stitched per layer via the `stitches` graph (`assemble_stitched.py`), front layer z=+0.4 cm over back z=−0.4 cm, **front↔back never welded** (held by contact), settled 400 frames, then folded (memory `project_sim1_datagen.md`; script location `archive/sim1_data_audit/` per memory — **not found on disk, unverified**). Registry `/data/user_data/robertre/sim1/assets/cloth`: 697 `v2_*` + 20 `g20_*` USDC (edge 11.5 mm, e.g. `v2_000.usdc.report.json`). Simulator bug fixed 07-22: open gripper re-dragged cloth on retreat.

**Vintages.** `robert_sim1/cloth_fold_v2` = 2,626 raw npz / 191 G (4 actions per garment: diag/half/lift_random/side); 2,420 passed grading + integrity → band-split by particle count into `cloth_curated_small` 367 (93 garments) / **`cloth_curated_med` 1,291** (symlinks into cloth_fold_v2; 7–12k particles) / `cloth_curated_large` 762 (212 garments), episodes trimmed to [grasp−60, release+90] (`configs/cloth_fold_v10.yaml:1-27`). Older: `cloth_fold_v1` 1,117, `_med/_large/_small/_diagfix`. The unified corpus is the **materialized** `unified_v1/cloth/{train,val}` = 136 + 18 tars / 42 G built by `scripts/unified/materialize_cloth_wds.py` (stride 2 → dt 1/30, shift applied, pad to N=12000+2000 with tail padding, val routed by the 38 held-out garments of `cloth_fold_v10.yaml:34`, `:33-40,95,126-128`); 68 of 1,291 episodes with |coord| > 0.8 m were excluded (`reports/unified/cloth_exclusion.json`) → 1,223 (1,086 train / 137 val).

**Raw npz schema** (`cloth_fold_v2/g20_05_half.npz`): `cloth_points [T,Nc,3]`, `left/right_ee_points [T,1000,3]`, `ee_link_ids [1000]`, `joint_q [T,19]`, `openness [T,2]`, `left/right_ee_pose [T,7]` (xyz + xyzw quat), `delta_*`, `fps=60`, `record_id`, `table_top_z=0.8`; T 113–509, Nc 7,001–11,981. Note diverse_v1 deltas are achieved, but `variant_0002` `delta_*` are COMMANDED (memory `project_sim1_cloth_variant_dataset.md`). Materialized keys: `particle_pos/obj_ids/valid_particles/n_cloth/dt`; obj_ids 0=cloth, 1=left EE, 2=right EE. Adapter `UnifiedClothDataset` (`unified_dataset.py:477-593`): strips tail padding (`:487`), motion-aware window sampling `motion_alpha=1.0` (`:495` — the sampler that suppressed gripper rotation 100× vs eval windows, per the normalization study), fills E/ν/ρ, sidecar link ids from `unified_v1/cloth_sidecar.npz` (`:489`). Specialist loaders: `SIM1ClothVariantDataset` (`data/dataset.py:1298`, stride/heterogeneous N, `val_files`), legacy `SIM1ClothDataset` hardcodes `N_CLOTH=7021`, `SKIP_FRAMES=30` (`:1158-1161`).

### 3. Pickup (IsaacLab rigid pick/place, "He's data")

**Generator.** `/data/user_data/robertre/RLSkills_IsaacLab/scripts/collect_pickup_3dpcwm.py` replays Zhe He's (hez2) rl_games policies (`/data/group_data/katefgroup/datasets/hez2/{pick,pick_convex,pick_convex_lying,place}.pth`, asset whitelists, init-arm npz) in IsaacLab and writes WDS directly — collector and converter are one script; launcher `collect_pickup_data.sbatch` (`:115-129`: `--success_only_window`, `--no_early_success_termination`, `STEPS_OVERRIDE=40`, 256–512 envs). Assets: ~250 curated tabletop objects per variant (Kitchen_Table + libero), union **946**; per-asset mass/μ, E/ν/ρ from sidecars. **Not** `VLA/3DPCWM/he_data/` (mbronars, old grasp-style schema T=12, N=3436).

**Vintages.** Originals `robert_pickup_data_{place,convex,pick,upright,convex_lying}` (99/155/127/144/143 tars, ~125K windows, 10.35 % artifacts on place) → the four **`_success`** collections (`gpu0/` only): place 63 tars/16,052 windows, convex 49/12,536, pick 49/12,512, upright 49/12,500 = **210 tars / 53,600 windows / 2.14M frames** (du 15/11/11/12 G). Eval extras: `pickup_holdout_longhorizon/{place,convex,pick,upright}` 800 trajs × T=190 on the 142 held-out assets; `teleop_long_horizon_clean` etc. (57 spline-teleop trajs, training assets only).

**Format** (64 samples/shard × B=4, `pick_success/gpu0/00000.tar`): `particle_pos [4,40,1183,3]` (671 gripper + 512 object), `valid_particles`, `obj_ids` (**0=object, 1=gripper**), `absolute_targets` (NaN on object = 43 %), `material_properties [4,1183,4]` = (E=1e9, ν=0.30, ρ, μ), `link_pos/link_quat [4,40,14,·]`, `link_local_pts [14,250,3]`, `link_local_npts`, `link_names`, `joint_pos/vel_obs`, `joint_pos_target [4,40,9]`, `reward`, `asset_id`, `goal_pose`. dt = 0.05 s (20 fps). Loaders: `UnifiedPickupDataset` (`unified_dataset.py:324-378`; actions = achieved gripper motion, gripper link rows `EE_LINK_ROWS=(8,9,10)` `:336`), pose specialist `PickupPoseDataset` (`data/pickup_pose_dataset.py:191-196` keys, `:73-107`/`:382-425` blocklists, `:135-160` per-root 10 % split), legacy `RLDataset` (`data/dataset.py:1049`).

### 4. Pushing (rod pushes block)

**Generator.** Colleague's Genesis collectors `/data/user_data/robertre/Genesis/examples/push_data_collection{,_glb}.py` (`SIM_DT = 2e-3`, `:25`; rest 70 + push 200–300 + observe 200 steps, one frame per step) and `push_with_rigid_new.py` (dt 1/450). Frame rate therefore **≈450–500 fps, not 24**; the code carries `DOMAIN_DT['pushing']=None` and the spec says UNKNOWN (`unified_dataset_spec.md:110`); the 24-fps assumption lived in the 08-25 normalization study (memory). Exact writer subsample needs the collector's config (mbronars) — unverified.

**Vintages** (all 64 samples/shard × B=4, N=256 = 128 rod + 128 block): `pushing_data` (T=40, no actions; 399 tars + `test/…` subdirs = 568) → **`pushing_data_actions`** (T=40, **404 tars / 24 G**, unified corpus) → `pushing_data_actions_04_18` (T=149, 482 train + 16 test tars, adds `start_state/end_state [4,12]`; the F9 pose-guidance work) → `push_data_08_07` (T=149, 175/16/16 train/test/val, adds `block_physics [4,7]`; rebuttal `pose_push_*_0807` configs). Also `pushing_data_actions_filtered` (404), `_04_04` (144), `pushing_data_pointworld_05_03` (20,517 tars, PointWorld export).

**Format/semantics** (verified on `pushing_data_actions/00000.tar`): `particle_pos [4,40,256,3]`, `absolute_targets` **non-NaN for every particle and exactly `pos[t+1]` for rod AND block** (MAE 0.0) — a label leak if used naively; `material_properties [4,256,4]` only ρ (500–1000) populated; `obj_ids` 0=block, 1=rod; rod 3.1 mm/frame median. Legacy `RLDataset` hotfix: when no target is NaN, `action_mask = obj_ids > 0` (`data/dataset.py:1095-1098`), then remaps ids to `obj_ids > 0` (`:1112`). `UnifiedPushingDataset` (`unified_dataset.py:381-428`) never reads `absolute_targets`, derives actions from rod motion, asserts zero action on the block (`:406-407`), fills E/ν and ρ NaN→750 (`:409-411`).

### 5. Blocklists and defects

| domain | defect | fix / list | scope |
|---|---|---|---|
| mpm v2 | 77/18,933 objects (0.41 %) in 71 episodes collapse below 30 % Rg (6 to a point); all `type 0` elastic with E<1e5 Pa and mostly ν<0.18 (bulk modulus ~8 kPa) | `reports/unified/data_audit_0903/mpm_blocklist_train.json` (64 train eps ×2 path forms = 128 entries) via `MPM_BLOCKLISTS` (`:617`), train only → 6,553 eps; 7 eval eps left in (`mpm_blocklist_eval.json`) | applies to runs after 2026-09-03; UNet 40K baseline trained without it |
| mpm | no rest jitter (0.012 mm/f median), no explosions; type-2 spreading is physics | none | — |
| cloth | **39 % of episodes (472; 140 garments, v2 family) start with front/back layers coincident** (z-extent <0.75 spacings — the sewing pattern laid flat, an assembly failure); garment v2_549 mis-assembled (detached pieces) | `cloth_blocklist_train.json` (418 = 415 flat + 3 v2_549) and `cloth_blocklist_val.json` (57) via `CLOTH_BLOCKLISTS` (`:626-629`), both splits → **668 train / 80 val** | all cloth numbers before 09-03 were on the unfiltered 144/137-episode val — not comparable |
| cloth | left in: per-episode uniform garment rescale (median 13 %, max 26 %), diag folds push cloth 20 mm below its plane (table penetration), lift_random closes the left fingers + right idle, 6 eps >3× stretch (XPBD tail) | none (user decision) | — |
| cloth diverse_v1 (v8) | 32 % of trajs have zero cloth motion, ~80 % frames static; bimanual = symlinks to variant_0002 cut mid-motion; commanded deltas in variant_0002 | v8 used 4 of 7 categories (751 trajs); v10 curation + trimming | — |
| pickup | sim artifacts (teleport, rot_glitch, phantom, dropped, far_majority): originals 10.35 %, `_success` 0.13–1.52 %; GT phantom baseline ~1 mm pre-contact | `reports/phantom_blocklists/robert_pickup_data_4success_combined.json` (299 elem entries) + asset holdout `stats/unified_v1/blocklists/holdout_TRAIN_block.json` (142/946 assets, 8,207 windows) via `PICKUP_BLOCKLISTS` (`:608-611`) | see finding below |
| pushing | `absolute_targets` label leak (above); frame rate undocumented | assert in adapter `:406` | — |

**New finding (verified live, not yet fixed):** both pickup blocklists store shard paths with the `/data/user_data/robertre/Genesis/3DPCWM_Data/…` prefix, but since `GENESIS = GROUP` (commits 1b71b6e 2026-08-26 / 9ce15ac 08-27) the loader globs `/data/group_data/…` and `_blocklist_filter` compares strings — absolute and root-relative forms both miss. Running the real pickup train pipeline: 200 raw samples → **0 blocklist matches; 97 would match if the URL had the user prefix**. Consequence: every unified run launched from 08-26 on (norm A/B/C arms, round-2 arms, UNet xy_p99 40K, joint overfits, the staged 200K pair) trains on the 142 held-out pickup assets and the 299 phantom windows, and unified pickup val is not asset-matched. The 130K rebuttal model (08-07..12) predates the switch. Fix is trivial (normalize prefixes in `_load_blocklists`, or ship both forms as the mpm/cloth lists do) plus a shipped-list test like `tests/test_episode_blocklists.py:48-60`. `ALLOW_MISSING_BLOCKLISTS` (`:83-88`) only guards unreadable files, not silent non-matches.

### 6. Splits and eval sets

| domain | train | val (unified) | test/eval sets | numbers that depend on them |
|---|---|---|---|---|
| mpm | `robert_mpm_v2/webdataset_full49` 6,617 → 6,553 eps | `robert_mpm_v2_test_seen` 270 eps (`MPM_VAL_ROOT :631`) — doubles as an eval set | test_unseen 286 (val_unseen 128 + 00001plus 158), ood 492; **paper Table-1 = v1 `robert_mpm_test/webdataset_full49` first-60/count, AR C=8**, ≈90 % seen-like assets vs v2 train | 6.48 (paper) vs 5.05 (v2-c2) vs 7.49 (unified@100K) all Table-1 protocol; audit scanned test_seen 270 / unseen 158 / ood 96 (why 96 ≠ 492 unverified) |
| cloth | 1,086 → 668 eps | 137 → 80 eps (38 held-out garments) | OOD bands `cloth_curated_{small,large}`; v8-era: 8-traj diverse_v1 val + `variant_0002/000086.npz` 1000-frame rollout (`docs/DATASET_SPEC_FOR_POINTWORLD.md:12,36-45`) | cloth_n144/n25, UNet 40K, rebuttal 16.8/26.7 all pre-blocklist |
| pickup | 189 of 210 shards | last 10 % shards per root (21 shards) — asset-matched only if the holdout list matches (currently not) | asset-disjoint holdout 142/946 (v10: held-out 11.25° vs 12.40° in-dist, split-difficulty artifact); `pickup_holdout_longhorizon` 800 × T=190 | v9 val 0.0819 / 12.10° baseline are in-distribution (95 % shared assets) |
| pushing | 364 tars | last 10 % of sorted tars (40) (`split_per_root=False`, `:710-713`) | 04_18 `test/` 16 tars; 08_07 `val/test` 16/16 (rebuttal cond 25.7 mm full-147f) | pushing rows dropped from the rebuttal table |

### 7. Unused / available data

MPM v2 drag `robert_mpm_v2_drag/webdataset_full49` 1,318 tars (+ drag ood/seen/unseen); `robert_mpm_v2/webdataset_full49_wallincl` (+2,611 wall-contact clips, per-frame `wall_contact` flag in h5); moulding `deformable_mould_new_12k_wds` (655 tars, memory); pickup non-success originals 668 tars; `cloth_curated_{small,large}` 367/762 eps and `cloth_fold_v1` 1,117; `sim1_cloth_diverse_v1` primitive/ik_tap categories; `deepmind/water_ramps`; ClothTransformer download at `sim1/downloads/clothtransformer` (18 GB); `pushing_data_pointworld_05_03`.

### 8. Data cleanup candidates

1. **Pickup blocklist prefix mismatch** (§5) — fix before any further unified launch; add a shipped-list match test.
2. `docs/scope_unified_unet.md` is referenced 21× in code/docs (e.g. `unified_dataset.py:87,510`) but never existed in git.
3. Stale docs: `docs/data.md` (no mpm_drag/sim1/pickup_pose/unified entries; mpm_multi says 4 material cols, v2 has 5), `docs/datasets_corl2026.md` (v1 MPM 24 frames/E≤1e6, sim1 91 trajs, He's pilot 400 trajs, pushing 04_18 — all superseded), `docs/unified_dataset_spec.md:16` (cloth 1,291=1,147+144 → 1,223 → 668/80), `:89,144` (mat-type semantics now known), `:110,143` (pushing dt). CLAUDE.md "Dataset Locations" still leads with grasp_30k/rl/mpm_multi v1/v2 paths.
4. Legacy loaders in `data/dataset.py` with no live consumer: `Grasp30kDataset`, `GraspNewDebugDataset`, `UncondDebugDataset`, `MPMDataset` (single-object), `SIM1ClothDataset` (N_CLOTH=7021 hardcoded; only teleop scripts); `RLDataset` remains live for the pushing pose configs (14 `format: rl` configs). `data/pickup_dataset.py::PickupDataset` survives only for `RigidStats` (`pickup_pose_dataset.py:115`) and `archive/pickup_rigid_legacy`.
5. Configs pointing at missing/unreadable data: 34 `mpm_multi` configs reference `/data/user_data/mbronars/packages/pwm_production{,_unet}/…` (permission denied from this account — unverified), `default.yaml`/`vae_grasp.yaml`/`latent_diffusion_grasp.yaml` (`./stats/unified_stats.npz` missing), `grasp_debug.yaml`, `latent_diffusion_rl.yaml`, `grasp_rl_diffusion.yaml`, `uncond_debug.yaml`. 53 configs reference `RigidFormerSphere` (a fifth, colleague-owned dataset outside the four domains) — scope decision.
6. `DOMAIN_DT['pushing']` should become 1/450–1/500 once the writer subsample is confirmed; `DOMAIN_FUTURE_LEN` and the cloth `motion_alpha` sampler are code-baked decisions to surface in pre-flights.
7. Scripts still hardcode the user-data prefix (`scripts/unified/audit_consistency.py:44`, `eval_paper_cloth_table3.py:86`, `eval_paper_mpm_unified.py:104`, `materialize_cloth_wds.py:35`) — same class of bug as item 1.

---

# 2. Normalization

### 1. Mechanics

#### 1.1 `NormalizationStats` (affine min-max) — `data/dataset.py:25`

Formula (`data/dataset.py:90`): `y = 2·(x − min)/(max − min + 1e-8) − 1`; `clip=True` clamps to [−1, 1] (`:91-92`). Inverse `x = (y+1)/2·scale + min` (`:107`), never clipped. `scale = max − min` (`:35`). Shape-agnostic over the last dim, so the same class serves 3-ch particle deltas, 6-ch pose deltas and the 20-D SIM1 pose actions.

Loaders: `load_all(path)` (`:40-65`) returns `(delta, position, offset)` from keys `min/max_delta`, `min/max_position`, `min/max_offset`; it dispatches to `AsinhNormalizationStats` when the npz carries `delta_norm_kind == "asinh"` (`:47`). `load_pose_delta_stats` (`:67-75`) reads the 20-D `min/max_pose_delta` keys (SIM1 v8, called at `:1366`). `normalize_actions/denormalize_actions` (`:109-141`) assume gripper particles come first — **only `tests/test_normalization.py:173` calls it**; production normalizes actions via `batch.delta_stats.normalize(batch.actions)` with no clip (`models/transformer.py:1114`).

`batch.prediction_stats` (`data/batch.py:237-243`) picks `position_stats` / `offset_stats` / `delta_stats` by `prediction_target`; the same `pose_delta_stats` field holds 6-ch unified pose stats or the 20-D SIM1 stats (`data/batch.py:73, :85-87`, consumers disjoint).

#### 1.2 Where clip is / is not applied

| tensor | site | clip |
|---|---|---|
| future target `x_0` (train) | `train.py:1916` | `cfg.train.clip_deltas`, default True (`:1381`, `:1875`) |
| future target (one-step debug / validate) | `train.py:2314`, `:2816` | True, hard-coded |
| context deltas (train + val) | `train.py:1934, :1953, :2336, :2843-2858` | **False** |
| pose targets `pose_x0` (train + val) | `train.py:1976-1977`, `:2867-2868` | True |
| rigid rows rendered into the particle channel (input-only) | `utils/pose_native.py:65` | False; safe because the particle loss keeps only deformable rows (`train.py:2063, :2071`) |
| actions | `models/transformer.py:1114` | False |
| RoPE / KNN positions | `models/transformer.py:963-966`, `joint_transformer.py:869-872`, `unet_std.py:1469` | False, `.detach()`ed; pose centers use `clip=True` (`transformer.py:1269, :1329`) |

The group README's "all applied with clip=True" is true for targets only.

#### 1.3 Position stats and RoPE

Positions are normalized by `position_stats` before spatial RoPE; `rope_spatial_base` (default 0.1, `models/transformer.py:253`; unified configs use 0.04, `configs/unified_v1_tier2v2.yaml:42`) sets the frequency spectrum. The unified loaders ignore file position stats and use `unified_position_stats()` = ±`UNIFIED_SCALE`(0.8 m) (`data/unified_dataset.py:31, :57-62, :264`), so normalize computes exactly `p/S` after the fixed per-domain shift. `unet_std._geo_radius` (`models/unet_std.py:1397-1406`) asserts isotropic `delta_stats.scale` unless `unet_geo_r_m` is set (unified UNet configs set it).

#### 1.4 `AsinhNormalizationStats` — `data/dataset.py:154`

Per channel **group** (particle `[0,0,1]` = {x,y} | z; pose `[0,0,1,2,2,2]`), the magnitude `r` is warped `y = asinh(r/x0)/asinh(x_max/x0)` and the vector rescaled by `y/r` so direction is untouched (`:212-230`; the radial form keeps yaw-equivariance, `:161-172`). `clip=True` clamps only the magnitude at 1 (`:225-226`). The inverse `r = x0·sinh(y·asinh(x_max/x0))` **saturates at |y|=1** (`:228, :235-248`) because the unclamped inverse is exponential (a 1.5 on z came back as 7 m). Exposes `min_val=−x_max`, `max_val=x_max`, `scale=2·x_max` for interface parity (`:201-210`). Loaded from `delta_x0/delta_xmax/delta_groups` or `x0/xmax/groups` via `from_npz` (`:251-259`); pose files dispatch at `data/unified_dataset.py:293`.

#### 1.5 Pose deltas (6 ch)

`_augment_pose_channels` (`data/unified_dataset.py:195-225`) builds `pose_deltas_gt [B,F,K,6] = [Δt(3) m, so3_log(ΔR)(3) rad]` and attaches `pose_delta_stats` (`:224`); stats are loaded lazily per domain from `POSE_STATS_SETS[stats_set]` (`:286-295`, keys `min_val/max_val`). Loss weight per channel therefore goes as `1/scale²` — the source of cloth's 63× ω_x-vs-ω_y asymmetry under the v1 files.

### 2. Stats registry — `data/unified_dataset.py`

`STATS_DIR = /data/group_data/katefgroup/VLA/3DPCWM/stats/unified_v1` (`:636`), `STATS` (`:637-642`), `POSE_STATS_TMPL` (`:192`), `STATS_SETS` (`:657-668`), `POSE_STATS_SETS` (`:670-678`). Selection: `data.stats_location: unified://<name>` → `parse_stats_set` (`:681-689`; None/"" → `per-domain`, unknown name raises), threaded through `create_dataloader` (`data/dataset.py:1800`) and `build_domain_dataset` (`:696-700`). No mpm pose file exists in any set (mpm has no rigid bodies).

| set | particle pattern | pose pattern | on disk | numbers per domain |
|---|---|---|---|---|
| `per-domain` (default) | `STATS_DIR/{robert_mpm_train_1to4obj_v3_p9999, unified_v1_pickup_particle, cloth_curated_med_stride2, pushing_actions}_stats.npz` | `STATS_DIR/unified_v1_pose_delta_{d}.npz` | all 7 in group dir (also in `stats/`; pickup/cloth/pushing particle files **untracked** in git) | 3 per-axis particle half-ranges (cloth 6 asymmetric bounds) + 6 pose half-ranges |
| `iso_p9999` | `./stats/unified_iso_p9999_particle_{d}.npz` | `…_pose_delta_{d}.npz` | yes (7) | 1 (trans; rotation copied from v1) |
| `global_iso_p9999` | `./stats/unified_global_iso_p9999_…` | same | yes (7) | 2 total: 130.2 mm, 0.3414 rad |
| `xy_p99` | `./stats/unified_xy_p99_…` | same | yes (7) | 2 particle + 4 pose (20 total) |
| `xy_p99_global` | `./stats/unified_xy_p99_global_…` | same | yes (7) | 4 total |
| `unified_asinh` | `./stats/unified_asinh_…` | same | yes (7) | 3 (x0,X) pairs shared, 5 independent (r0=32 mm) |
| `pd_asinh` | `./stats/unified_pd_asinh_…` | same | yes (7) | (x0,X)×{xy,z,rot} = 6 |

### 3. Inventory of `stats/` (94 files, 5.1 MB)

| family | files | consumer | status |
|---|---|---|---|
| unified v1 baseline | `robert_mpm_train_1to4obj_v3_p9999`, `unified_v1_pickup_particle`, `cloth_curated_med_stride2`, `pushing_actions`, `unified_v1_pose_delta_{pickup,cloth,pushing}` | `STATS`/`POSE_STATS_TMPL` (group copies), `configs/robert_mpm_c8_v6_1.yaml`, `cloth_fold_v10*.yaml`, many `pose_push_*` configs | **live** |
| unified sets | 42 `unified_{iso_p9999,global_iso_p9999,xy_p99,xy_p99_global,asinh,pd_asinh}_*` + 3 manifests | `STATS_SETS` templates only (exact-name grep finds nothing except `diag_pickup_pose.py`) | live via registry; see cleanup |
| MPM | `mpm_sand` (mpm_sand.yaml), `mpm_multi_v1` (sbatch 1-4obj), `mpm_multi_v2` (train_mpm_multi.sh), `robert_mpm_drag_setA_C2_p9999` (drag v6.1), `robert_mpm_v2_object_{6key_,}p9999` (object v2 configs), `robert_v2_full49_isodist_p9999` (unetstd configs) | as listed | live/legacy |
| MPM orphans | `mpm_multi_stats`, `mpm_multi_object_stats`, `robert_mpm_train_1to4obj{,_v2}_stats`, `robert_mpm_drag_setA_C2_p9999_stats_OLD` | none | **unreferenced** |
| pushing | `uncond_pushing`, `pushing_actions_04_04/04_18`, `push_08_07*`, `pushing_pose_se3_stats{,_2000b_rodtrans,_…_world,_…_world_04_18}` | push configs / sbatch / guidance sweeps | legacy |
| pushing orphans | `pushing_actions_filtered`, `pushing_pose_se3_stats_500b`, `pushing_pose_stats`, 7 `.png` previews | none | **unreferenced** |
| SIM1 | `sim1_cloth_diverse_v1_stride2` (v8), `sim1_cloth_diverse500` (sim1_cloth*.yaml), `sim1_cloth_stats` (compute script only), `cloth_fold_{v1,med}_stride2` (v9 configs) | as listed | live/legacy |
| SIM1 orphans | `sim1_cloth_variant_0002{,_stride2}` | none | **unreferenced** |
| pickup pose | `pickup_pose_stats_4dataset_success_robust{,_r6d}`, `pickup_pose_stats_place_robust` (via `RigidStats`, `data/pickup_dataset.py:49`) | v9 / uncond_place configs | live |
| pickup orphan | `pickup_3dpcwm_stats` | none | **unreferenced** |
| grasp/RL | `unified_stats_rl` (grasp_rl/vae_rl/latent_rl), `unified_stats_30k_1000_samples` (`inference.py`, `scripts/inference.sh`), `gripper_template.pt` | as listed | live |

**Referenced but absent**: `stats/unified_stats.npz` (default.yaml, vae_grasp, latent_diffusion_grasp — the CLAUDE.md quick-start path), `latent_stats{,_rl}.npz`, `unified_stats_rl_rectangle.npz`, `uncond_stats_v2.npz`, `grasp_debug_v2.npz`.

### 4. How each set was computed

| script | output | recipe |
|---|---|---|
| `scripts/unified/compute_pose_delta_stats.py:1-14` | `unified_v1_pose_delta_*` | full train episodes, symmetric p99.99 of |x| per channel |
| `scripts/unified/compute_pickup_particle_stats.py:1-15` | `unified_v1_pickup_particle` | train split with blocklists, symmetric max(|p0.01|,|p99.99|) per axis |
| legacy `scripts/normalization/compute_stats_{mpm,rl,sim1,v2}.py`, `compute_pose_stats.py` | mpm/pushing/cloth/sim1 files | `percentile` arg, default 0.5 → asymmetric 0.5/99.5 (cloth file is this); pushing protocol unrecorded; `compute_stats_{full,local}_isodist.py` = vector-norm p99.99 isotropic |
| `scripts/unified/compute_unified_iso_stats.py:1-30` | `iso_p9999`, `global_iso_p9999` + `unified_norm_ab_manifest.json` | train stream as training sees it (600/600/400/2000 batches), p99.99 of |dp|; C = equal-weight mixture; rotation copied (B) or max over domains (C) |
| `scripts/unified/compute_norm_round2_stats.py:1-33` | `xy_p99`, `unified_asinh` + `unified_norm_round2_manifest.json` | Bernoulli-thinned ~8M particle-frames/domain; xy = max(p99|dx|, p99|dy|) (hypot variant recorded as `alt_p99_xy_magnitude_mm`), z own p99; pose grouped {dt_x,dt_y}|dt_z|{ω_x,ω_y}|ω_z. **Window-skew fix** (`:135-152`, `:361-363`): cloth's motion-aware training sampler suppresses gripper rotation 23–118× vs eval prefix windows, so pose scales = elementwise max over training AND eval window streams (cloth ω 0.028° → 2.80°/0.38°) |
| `scripts/unified/compute_pd_asinh_stats.py:1-25` | `pd_asinh` + manifest | per-domain x0 = p50, X = p99.99, x0 ≥ X/1350 |
| none in repo | `xy_p99_global` | commit 8dd2ca7: each number = max over per-domain xy_p99 (no generator script committed) |

Per-domain `xy_p99` values (mm/frame; deg/frame), from the npz files:

| domain | particle s_xy / s_z | body Δt_xy / Δt_z | ω_xy (tilt) / ω_z (yaw) |
|---|---|---|---|
| mpm | 26.83 / 96.84 | — | — |
| pickup | 17.17 / 28.82 | 15.38 / 28.21 | 7.45° / 10.82° |
| cloth | 5.66 / 5.09 | 9.09 / 5.40 | 2.80° / 0.377° |
| pushing | 3.88 / 0.165 | 3.88 / 0.165 | 0.149° / 3.11° |

`xy_p99_global` = the column maxima: **xy 26.83 mm, z 96.84 mm, tilt 7.45°, yaw 10.82°**, shared by particles and body origins in every domain. v1 baseline half-ranges (mm): mpm ±76.2/72.9/135.5; pickup ±29.6/27.6/37.0; cloth [−4.60,+3.63]/[−5.08,+4.32]/[−4.77,+4.61]; pushing ±3.79/3.88/0.16; pose pickup ±25.0/25.9/40.0 mm, ±19.6/18.9/18.9°; cloth ±14.1/8.2/8.3 mm, ±0.47/3.72/0.50°; pushing ±3.95/3.95/1.23 mm, ±2.14/1.96/4.37°.

### 5. Results history

**Study (2026-08-25, `reports/unified/normalization_study.md`, `scripts/unified/normalization_study.py`)**: |Δp| p99.99 = mpm 135 / pickup 42 / cloth 17 / pushing 5.1 mm; v1 files clamp cloth 5.3 % / pushing 3.0 % per component; visibility proxy t* = RMS/(1+RMS) under a global scale drops cloth/pushing to 0.02.

**Round 1, real corpus, transformer T2v2 82M @40K, 17 whole-set cells** (`reports/unified/norm_ab/summary_40k_wholesets.json`, runs `runs/unified_norm_{B,C}`): particle mm — mpm seen v2 125.2 / B 142.5 / C 137.2; unseen 105.6/124.8/114.2; OOD 123.4/151.5/129.6; pickup 9.3/9.8/11.7; pushing 2.8/3.5/9.7; cloth 2.2/2.4/13.9. **Global scale (C) refuted (cloth 6.3×, pushing 3.4×); iso p99.99 (B) not better than the v1 files.** t* predicted every ordering it separated → optimum tighter than p99.99.

**Velocity (2026-08-27, `reports/unified/velocity_study.md`, untracked)**: with per-domain scales velocity is a no-op (targets identical to ≤1e-6); cross-domain spread stays 21.8× in either unit; pushing is ≥100 fps (likely 450-500), so `DOMAIN_DT["pushing"]=None` and nothing consumes `DOMAIN_DT`.

**Round 2, real corpus, UNet 331M @40K, corrected C=4/F=16 harness** (`reports/unified/norm_ab_trained_windows/final_comparison.txt`, runs `runs/unified_unet_norm_{perdomain,xy_p99,asinh}`): xy_p99 wins **17/17 cells** (e.g. pushing 7.9 vs 8.4 vs 14.7; cloth 5.6 vs 10.3 vs 7.1; pickup ≈19.5 vs 22-27.5 vs 26-39; mpm seen-1obj 133.7 vs 176.1 vs 162.6); `unified_asinh` lost 17/17 and is worst on mpm OOD/unseen and cloth. `pd_asinh` was only tried on one mpm batch (2000 steps: asinh 1.9, pd_asinh 1.9, xy_p99 7.5, per-domain 12.8 mm UNet; tf 3.6/3.3/7.9/17.5) — "the mpm lever" but never trained.

**Round-2 single-batch overfits (2026-09-01, xy_p99, 10K, 8-seed sample mm)**: tf/UNet mpm 23.5/13.4, pushing 0.28/0.44, pickup 2.45/3.20, cloth 2.32/1.54 (artifact "Overfit Round Two"). **Joint four-batch overfit (09-02/03, xy_p99, 40K)**: tf 24.02 / 0.45 / 2.49 / 1.82 vs UNet 15.30 / 0.54 / 3.18 / 2.14 → transformer 3-1 (UNet wins mpm). **4-number `xy_p99_global` joint (09-03/04, page "Unified Overfit" 0063d527, `reports/unified/overfit_v2/build_joint_norm.py`)**: tf 19.86 / 2.32 / 3.23 / 7.17; UNet 15.28 / 6.20 / 8.08 / 8.65 — the shared scale loses on every small-motion domain (pushing 5-11×, cloth ~4×, pickup 1.3-2.5×), mpm unchanged/better.

### 6. Recommendation and open idea

Use per-domain `unified://xy_p99` (xy-isotropic, z separate, p99, pose scales covering both window distributions) — it is the only set that beat the v1 files on trained rollouts, and every one-number/four-number scheme (C, unified_asinh, xy_p99_global) lost on cloth/pushing/pickup. Open research idea: keep per-domain scales but feed `(log s_d, log Δt_d)` as conditioning so physical scale is recoverable; deferred because with four fixed domains it is a dataset one-hot in disguise (unverified as an experiment).

### Cleanup candidates

- **Unreferenced npz** (delete or move to `archive/`): `mpm_multi_stats`, `mpm_multi_object_stats`, `robert_mpm_train_1to4obj{,_v2}_stats`, `robert_mpm_drag_setA_C2_p9999_stats_OLD`, `pushing_actions_filtered_stats`, `pushing_pose_se3_stats_500b`, `pushing_pose_stats`, `sim1_cloth_variant_0002{,_stride2}_stats`, `pickup_3dpcwm_stats`, plus 7 `*.png` previews.
- **Superseded registry sets** (lost their A/B; keep manifests as provenance): `iso_p9999`, `global_iso_p9999` (round 1), `unified_asinh` (round 2, 0/17), `xy_p99_global` (09-04 overfit), `pd_asinh` (never trained) → all superseded by `xy_p99`. Removing them also retires `compute_unified_iso_stats.py`, `compute_pd_asinh_stats.py`, `unified_norm_{A,B,C,asinh}.yaml`, `unified_unet_norm_{perdomain,asinh}.yaml`, and the `asinh` branch of `load_all`/`_pose_stats` if `AsinhNormalizationStats` goes.
- **Dead code**: `NormalizationStats.normalize_actions/denormalize_actions` (test-only); `data/rotation_aug.py:104 xy_isotropic_stats` (never wired in); `DOMAIN_DT` (nothing consumes it, pushing value wrong).
- **Broken references**: `default.yaml`/`vae_grasp.yaml`/`latent_diffusion_grasp.yaml` → missing `stats/unified_stats.npz`; `latent_stats{,_rl}.npz`, `unified_stats_rl_rectangle.npz`, `uncond_stats_v2.npz`, `grasp_debug_v2.npz`; CLAUDE.md dataset table lists the missing `unified_stats.npz`/`latent_stats*.npz`.
- **Provenance gaps**: 3 of the 7 default-set files untracked in git (group dir is the canonical copy); `xy_p99_global` has no generator script; `reports/unified/velocity_study.md` untracked; cloth file's asymmetric 0.5/99.5 and pushing's unrecorded protocol are the reason the default set is not one recipe.

---

# 3. Model architectures, diffusion setup and samplers

Repo `/data/user_data/robertre/particle_world_model`, HEAD `7091f3a` (branch `robert/unified-unet-combined`, 2026-09-04). Paths below are relative to that root; line numbers are from this HEAD. One shared dataclass, `ModelConfig` (`models/transformer.py:49-448`), configures classic, joint and both UNets; `architecture` is not a field, it is read off the raw YAML (`train.py:678`, `inference.py:921`) and dropped by the `fields(ModelConfig)` filter (`inference.py:913-914`).

### 0. Model inventory

| class | file | selected by | used by configs | params | status |
|---|---|---|---|---|---|
| `TrajectoryTransformer` | `models/transformer.py:453` | `architecture: classic` (default, 45 configs without the key) | unified `unified_v1_tier2v2_c4f16`, `unified_norm_*`, `unified_v1_tf333m_*`; SIM1 cloth v8/v9/v10 (`sim1_variant_pose_rigid_v8.yaml`, `cloth_fold_v10*.yaml`); MPM drag v6.1; pushing `push_uncond_*` | 81,960,201 (dim 512/depth 8, c4f16); 333,046,537 (dim 768/depth 15) | live |
| `JointTrajectoryTransformer` | `models/joint_transformer.py:116` | `architecture: joint` (75 configs) | `robert_mpm_c8_v6_1.yaml` (MPM object v6.1), `robert_mpm_object_v2{,_c2,_fast}.yaml`; 60+ `sph64_*`/`mul*`/`movisphere_*` sweeps with no checkpoints in `runs/` | 78.5M v6.1 / 78.6M v2-c2 (memory, dim 576/depth 12/192 latents) | live for MPM object; sweep configs stale |
| `UNetWrapper`/`UNet` (std) | `models/unet_std.py:1254`, `:971` | `architecture: unet_std` (12 configs) | `unified_unet_norm_xy_p99.yaml` (the shipped 40K handoff), `unified_unet_c4f16*.yaml`, `unetstd331_*.yaml` (MPM-only 331M) | 331,618,989 (+512 with `unet_actions`) | live |
| `UNetWrapper`/`UNet` (research) | `models/unet.py:1263`, `:1110` | `architecture: unet` (21 configs: `unet_*.yaml`, `sedge166_split108`, `spikehunt_full_rope`) | mixer ablations only | — | legacy, still selectable (`train.py:735-739`) |
| `PoseTransformer` | `models/pose_transformer.py:39` | `train_pose.py:311` | `pose_push_*.yaml`, `pickup_pose_uncond_place_0810.yaml` | (unverified) | live (pushing pose, last trained 2026-08-22) |
| `PickupPoseTransformer` | `models/pickup_pose_transformer.py:76` | `train_pickup_pose.py` | `pickup_pose_v9_*` | ~80M (memory) | live, deployed v9 |
| `TrajectoryVAE`, `LatentTransformer` | `models/vae.py`, `models/latent_transformer.py` | `train_vae.py`, `train_latent_diffusion.py` | `vae_*.yaml`, `latent_diffusion_*.yaml` | — | legacy (last content commit 2026-03-08) |

Not in the tree despite memory/prompt references: `models/pose_branch.py`, `models/unified_unet.py`, `UNetConfig.variable_objects` (only `robert/unified-unet-sidepose`; dropped in merge `9e7193b`). `models/_unet_std_ref.py` is gitignored (`.gitignore:74`) and byte-identical to `git show 30b04f6:models/unet_std.py` (pre-fork snapshot for the bit-identity gate).

### 1. `TrajectoryTransformer` ("classic")

Pipeline (`models/transformer.py:459-464`): embed → FPS+KNN encode N→L → `depth` × `TransformerBlock` → cross-attention decode L→N → `LayerNorm → Linear(d,3)` head (`:642-643`).

- **FPS+KNN**: one `sample_farthest_points` on the anchor frame (`:983`), `knn_points` per frame when `per_frame_knn` (`:992`), else anchor-only (`:1012`). `bypass_compression` (default **True**, `:89`) skips it. **Stratified** encode/decode (`:2103`, `:2383`): per-object FPS/KNN with a latent budget proportional to valid particle count (`_compute_latents_per_object_single`, `:2050-2100`), each object decoding only from its own latents; `batched_stratified_{encode,decode}` collapse the per-object loop for uniform layouts (`:92-100`). `cloth_attends_all_latents` (`:369-376`) is the SIM1 exception.
- **all_relative** (`:90`, default True): context = C−1 deltas through the same `delta_physics_embed` (`:493`), positions rebuilt as frame0+cumsum; `C_ctx = C-1` set once (`:878`).
- **Context noising / "context AdaLN"**: `noisy_context` + `context_timesteps` enter `forward` (`:858-859`); the context timestep embedding `ctx_t_emb` (`:1063-1065`) is applied through the *same* `AdaLayerNormZeroSingle` as the queries in every block (`models/components/attention.py:540-549`). There is **no `context_adaln` flag** in code; it exists only in `README.md:182` and `CLAUDE.md`. `inference.py:912-919` derives `_ctx_timesteps_trained` from `cfg.train.noise_context` because the light-context-noise recipe trained with `context_timesteps=None`.
- **Per-object noise** (`per_object_noise`, `:339`): `[B,F,N]` timesteps gathered to latents via `fps_idx` (`:1190-1227`), per-token AdaLN plus additive `noise_level_proj` (`:526-532`). Enabled in **zero** configs (also zero in `archive/`); its consumer `run_per_object_inference_df` (`inference.py:567`) is reached only from `train.py:2696`.
- **Windowed temporal** (`use_windowed_temporal`, `:72`): sink+window mask from `create_sink_window_mask` (`attention.py:26`) applied only to the decoder temporal mask (`:834-852`, `:1317`). Enabled in zero configs.
- **RoPE**: `RotaryPositionEncoding3D(max_period=rope_spatial_base)` (`:548-551`); positions normalized by `position_stats` and `.detach()`ed (`:965-966`, `:1270`, `:1330`). Unified configs use `rope_spatial_base: 0.04` (12 configs), all specialists 0.1 (94). Banded RoPE (`banded_rope`, `rope_f_min/f_max`, `:257-261`) is opt-in.
- **Pose-native head** (`pose_native`, `:364`, Tier-2 v2 per `docs/tier2_v2_design.md`): rigid rows of `noisy_future` are rebuilt from the noisy 6-D pose via `utils/pose_native.py:21` (`:912-914`); pose tokens = `pose_token_embed(Δ6)` + masked-amax template geometry + type vector (`:1243-1246`), context pose tokens from GT `so3_log` deltas (`:1249-1255`), pose-space action skip on the block's action channel (`:1258-1263`); tokens are concatenated onto the latent streams and RoPE recomputed at body centres (`:1264-1273`); zero-init `pose_v_head` reads `[B,F,Kr,6]` (`:1308`); pure-rigid batches return `(None, pose_v)` (`:1311-1312`). Asserts: no `predict_body_poses`, no `predict_rigid_pose`, no `per_object_noise`, stratified+rotary only (`:713-716`, `:1237`). Output materialization is `utils/pose_native.py:110 materialize_pose_native` (world-frame `R = ΔR@R`, `t += Δt`).
- **Legacy pose flags**: `predict_body_poses` (v1 pooled readout head, `:696-711`, `_body_pose_projection` `:1442`) used only by `configs/unified_v1_tier2.yaml`; `BODY_POSE_T_SCALE/R_SCALE` (`:45-46`) survive for it and `train.py:1810`. `predict_rigid_pose` (SIM1 v8 head, `:646-694`, `_rigid_gripper_projection` `:1549`): per-gripper 10-D `[Δt(3), ΔR6d(6), Δopen(1)]`, identity-bias init on the r6d columns (`:690-694`), integration and FK via `utils/gripper_kinematics.py:103,140`, x0→velocity conversion of the rigid rows (`:1372-1397`). Live in the 5 SIM1 configs together with `pose_action_dim: 20`, `cloth_attends_all_latents`, `commanded_gripper_tokens_enabled` (Fix C, `:731-769`).
- **Contact stripper**: `utils/state_dict.py:7` strips `backbone.` and drops `contact_head.*`; called in `inference.py:946` and `train.py` (4 call sites).

Live flags in the unified transformer configs (`configs/unified_v1_tier2v2_c4f16.yaml:17-47`): stratified+batched enc/dec, `causal_temporal`, `all_relative`, `per_frame_knn`, `embed_actuatable`, `concat_anchor_z`, `rope_spatial_base 0.04`, `flow_matching_prediction_type: velocity`, `pose_native`; windowed temporal, per-object noise, banded RoPE, decoder-identity, pointnet/pyramid/direct paths all off.

### 2. `JointTrajectoryTransformer` (`architecture: joint`)

Reuses `ModelConfig` (`models/joint_transformer.py:54`). Differences (docstring `:1-30`): context and future latents are **one stream** `[B, C+F, L, d]` under a block-causal temporal mask (`_get_joint_block_causal_mask` `:685-729`, `causal_context` optional) instead of cross-attention; the encoder runs **once** on the concatenated sequence (`:1068`, `:1108`); blocks are `JointTransformerBlock` (`attention.py:636-706`): temporal self → spatial self → GEGLU FFN at ratio 2.67, RMS qk-norm, PixArt adaLN-single with a shared `Linear(d,9d)` + per-block `[9,d]` table; no external `time_mlp`; decoder is cross-attention only, with opt-in `decoder_temporal`, `decoder_direct` (self-edge attention `_SelfEdgeAttention` `:58-114`), `pyramid_decode`, and the PointNet++ FP decoder (`:1394-1428`). **Actions are refused** (`:1062-1066`), which is why MPM drag v6.1 uses classic (`configs/robert_mpm_drag_v6_1.yaml:32`). No `pose_native` (0 hits). `torch.compile` note at `train.py:897-903` says joint fails dynamo tracing; `configs/robert_mpm_object_v2_dit_recipe_bigpyr.yaml:1-13` claims a compiled joint run — (unverified) which is current.

Joint-only research flags now living in the shared `ModelConfig`: `latents_per_object`, `register_token`, `pyramid_*`, `encoder_pointnet_*` (`models/pointnet_encoder.py`, imported only by joint), `encoder_temporal`, `decoder_*`, `concat_noisy_z`, `rope_shift_frame_start`, `f0off_*` (`models/transformer.py:105-338`).

### 3. UNet family

`models/unet_std.py` is the single-configuration extraction of the winning "sedge" run (`:1-9`); `models/unet.py` keeps five mixers (`:450-617`), edge graphs, the legacy spatial-first block order and an eager-only `_scatter_amax` (`unet.py:180-190`).

`UNetConfig` (`unet_std.py:114-256`): `points/strides/widths/enc_depths/dec_depths/in_dim` required; defaults `fps_random_start=True` (`:128`; note `ModelConfig.unet_fps_random_start` defaults False, `transformer.py:421`), `pool_mode/unpool_mode="attn"` (`:147-148`), `l1_by_mat={}` (`:152`), `l1_rule=""` (`:158`, only `"pow2_quarter"`), `pose_native=False` (`:164`). `stratified` is a **property** (`:214-216`): true when any of the L1 table/default/rule is set. `l1_for_object` (`:228-234`): table wins per material, else `2**round(log2(n0/4))` clamped to `[32,1024]`.

Forward (`:1095-1232`): `GraphProvider` (`:358-506`, compiler-disabled) does one ragged FPS on clean frame 0; every coarser level is a **prefix** of that ordering; per-frame kNN for interp (`interp_k`) and, for pointnet pools, groups (`pool_k`). Stem → `[enc, pool]×L` → bottleneck → `[unpool, dec]×L` → head; `Block` = temporal (block-causal) → spatial (flash SDPA, per-object segments at stratified L0 via `_segment_attention` `:306-334`) → GEGLU, all zero-init adaLN gated; long skips concat→LN→Linear (`:617-623`); registers from `register_from_level`. `Pool` (`:732-821`) lift-first attentive or PointNet reduce; `Unpool` (`:859-964`) inverse-distance FP trunk + gated read. Pose tokens ride every level as trailing rows and cross seams via `pose_up/pose_down` linears (`:1012-1015`, `:1163-1164`, `:1197`, `:1208`); zero-init `pose_v_head` (`:1230`). `UNetWrapper` (`:1254-1625`) maps `TrajectoryBatch` to 10 channels `[delta, frame0 offset, z, physics]` (`:1480-1490`), two currencies (geometry `pos/r`, `r = unet_geo_r_m` or `delta_scale/2`, `_geo_radius` `:1397-1405`), same forward signature as the transformer plus rejected `sup_idx` (`:1407-1414`), and a `frame0_offset` path `_forward_f0off` (`:1559`).

Opt-in flags, all default off: `unet_actions` (+4 stem columns zero-init, `:1319`, `:1349-1354`; off in the launch config after the 08-28 correction: actuated particles are all pose-token bodies), `unet_l1_rule`, `unet_supervise_context` (stashes `_last_ctx_pred`, `:1523-1531`; consumed at `train.py:2110-2134`, gated `train.py:703`). **Compile-NaN fix**: `Pool._read_pointnet` reduce `amax(dim=3)` → `.max(dim=3).values` at `:855`, commit `8a03325` (2026-08-28), bisect JSONs in `reports/unified/unet_probe/compile_nan/`; the pose-geometry `amax` at `:1078` is untouched.

Design provenance: `docs/unified_unet_design.md` is Robert's **pre-merge** side-pose plan (4 gaps `:10-17`, 4 resolutions `:19-45`); the "5 design points → theirs" (ragged L0 layout, within-object L0 attention, material-keyed L1 table, PointNet seams, pose tokens at every level) is recorded in merge commit `9e7193b` and memory, not in the doc. `docs/scope_unified_unet.md`, cited ~20× in code (`unet_std.py:39,151`, `transformer.py:434,439`), **does not exist** in this repo.

### 4. Pose family

- `utils/pose_ops.py`: `so3_exp/log` (`:58`, `:80`), **identity-anchored r6d** (`_IDENTITY_R6D` `:107`; `rotation_to_r6d` subtracts, `r6d_to_rotation` adds back before Gram-Schmidt, `:110-135`; commit `afd2bfd` 2026-06-21), `integrate_pose_deltas(rotation_rep=)` (`:216`).
- `utils/pickup_pose_ops.py`: `pose_dim(rotation_rep)` = 7/10 (`:60`), `pack/unpack_pickup_delta` (`:114-129`, open scalar last), parametric Panda gripper (`:131-216`), `integrate_pickup_pose_deltas` (`:308`).
- `utils/gripper_kinematics.py` (SIM1 v8): its own **raw** `r6d_to_R` with eps 1e-6 (`:37-50`, not identity-anchored — v8 relies on the identity bias init instead), `integrate_world_deltas` (`:103`), `apply_gripper_kinematics` (`:140`).
- `PoseTransformer` (`models/pose_transformer.py`): 6-D per-body deltas, geometry tokens = template xyz + twist velocity (`:46-51`), FPS/KNN over template@integrated-noisy-pose, used by `train_pose.py`/`inference_pose.py`/`guided_inference_pose.py` for pushing pose.
- `PickupPoseTransformer` (`models/pickup_pose_transformer.py`): `rotation_rep` threaded through embed/head (`:87`, `:97`, `:127`) and `_encode_geometry` (`:213`); `_embed_time` scales continuous t by `num_train_timesteps` (`:320-329`).
- Unified pose-native (Section 1/3) shares only `pose_ops.so3_*` and `utils/pose_native.py`; it has no open channel and no FK (`docs/tier2_v2_design.md:26-29`, `:60-68`).

### 5. VAE / latent tier — legacy

Closed dependency graph: `TrajectoryVAE`/`LatentTransformer`/`load_frozen_vae`/`latent_stats` are imported only by `train_vae.py`, `train_latent_diffusion.py:38`, `inference_vae.py`, `inference_latent_diffusion.py`, `scripts/debug_vae_recon.py`, `scripts/normalization/compute_latent_stats.py`, `tests/test_latent_diffusion_forcing.py`, `tests/test_ar_rollout_latent.py`; `VAETransformerBlock` (`attention.py:827`) exists only for it. Last content commit 2026-03-08 (`2484357`); `inference_latent_diffusion.py`'s last message is "trying to fix latent diffusion forcing inference". Six sbatch and six shell launchers remain. README/CLAUDE.md still present it as tiers 2/3.

### 6. Diffusion, samplers, guidance

- `diffusion/flow_matching.py`: OT-CFM `x_t=(1-t)x_0+t·noise` with linear `t = timesteps/T` (`:87-116`); `shift` only reaches `FlowMatchEulerDiscreteScheduler` (`:82`) and is 1.0 in all 143 configs that set it; the DF path uses linear `dt` (`inference.py:532-537`) so `shift` is effectively dead. `get_target` always returns velocity (`:117-141`); `as_velocity` (`:280-311`) converts sample-mode output; timesteps are continuous floats (`:175-228`, `time_sampler` uniform|beta). **Doc/code conflict**: `train.py:1997-2006` (since `a607563` 2026-07-26 / merge `ecd90ba`) trains `prediction_type: sample` **directly in x0 space** (`_fm_x0_loss`), as the `__init__` comment `:70-78` says, while the `get_target` docstring `:126-129`, `compute_loss` `:151-153`, and CLAUDE.md ("always v-loss, min_snr structurally required for sample") still describe the old velocity-space recipe. The pose-native pose term stays v-loss (`train.py:2095-2097`) — a mixed convention if `sample` is ever paired with `pose_native` (no live config does; all unified configs are velocity).
- `diffusion/ddim.py`: `TrajectoryDiffusion` wraps `DDIMScheduler` (`:55`), `epsilon|v_prediction|sample` targets (`:117-131`), integer timesteps. Selected by `denoise_method: diffusion` (`inference.py:960-967`).
- `diffusion/scheduling.py`: `generate_scheduling_matrix` `full_sequence|pyramid` (`:15-58`); per-object `actuator_leads|object_leads|inpaint_actuator|inpaint_object|per_object_pyramid` (`:60-160`), consumed only by the dead per-object path.
- `inference.py`: `load_model_and_diffusion` (`:876-984`) dispatches `classic|joint|unet_std|unet`, aliases legacy flag names (`:897-911`), strips the contact wrapper, loads EMA if present (`:945-947`). `run_diffusion_inference_df` (`:432`) denoises per-frame with `update_mask` (`:517`), jointly denoising `x_pose` under `pose_native` (`:485-495`). `autoregressive_rollout` (`:51`, strided) vs `streaming_rollout` (`:166`, delta-only, staggered buffer). Sampler by family (memory 2026-08-24, verified in scripts): UNet scored with `streaming_rollout` S=50 (`scripts/eval_object_rollout_unet.py:47,184`); transformers with strided/chunked `autoregressive_rollout` (`scripts/eval_mpm_ar_rollout.py:23,101`); the unified harness `scripts/unified/eval_unified.py:37,67,225` uses windowed `run_diffusion_inference_df`.
- Guidance: `guided_inference.py` modes `standard|df|noise` (`:1621-1622`), last functional commit `eef0676` 2026-04-19; `guided_inference_pose.py` modes `df|noise|action_opt|joint_opt` (`:1319-1335`, no argparse `choices`) on `PoseTransformer`, last touched by the 2026-06-21 cleanliness sweep; `noise_opt_guidance{,_pose}.py`, `interactive_pushing.py` likewise. None mention `unet`, `pose_native`, `unified`, or `frame0_offset`. The six sweep scripts listed in `CLAUDE.md:278` (`scripts/sweep_tier1_guidance.py`, `sweep_novel_pose.py`, `sweep_noise_pose.py`, `sweep_noise_hparam.py`, `sweep_action_opt.py`, `sweep_joint_opt.py`, `compare_all_phases.py`, `plot_novel_results.py`) all live in `scripts/archive/` since `603ca24`; only `scripts/test_all_guidance.sh`, `test_novel_difficulty.sh`, `build_sweep_report.py` remain at their documented paths. Verdict: unmaintained since the pushing era.

### 7. Parameter counts and capacity matching

From `reports/unified/**/*.json` `n_params`: **333,046,537** (tf333m, `overfit_v2/runs/tf333m*`, `overnight_0830/runs/tf333m_*`); **331,618,989** (unet331m, `unet_actions=false`); **331,619,501** (`unet_probe`, = +512 = 4 action channels × width 128, (unverified) that the probe had `unet_actions=true`); **81,960,201** (tf_c4f16 82M). Memory: MPM v6/v6.1 78.5M, v2-c2 specialist 78.6M, pickup v9 ~80M. Decision (`configs/unified_v1_tf333m_xy_p99_c4f16.yaml:1-31`): scale dim 512→768, depth 8→15, heads 8→12 (head_dim 64) to land +0.4% of the UNet, keep `num_latents 512` (FPS-selected, param-free), `gradient_checkpointing: full` and `fps_random_start` to mirror the UNet arm, compile left off; counted 2026-08-30 by constructing both as `train.py` does.

### 8. Known architecture bugs, found and fixed

| bug | fix | reference |
|---|---|---|
| Dead FM time conditioning: `SinusoidalTimeEmbed` floored continuous t to 0 (pickup v9 only) | drop `.long()` (`embeddings.py:48-52`), `_embed_time` scaling (`pickup_pose_transformer.py:320-329`) | `8ef58b0` 2026-06-24, memory `project_pickup_time_conditioning_bug`; gripper 12.2→2.5 mm |
| Compiled PointNet pool backward NaN (`amax`) | `max(dim).values` `unet_std.py:855` | `8a03325` 2026-08-28 |
| Decoder identity loss (query routed only by noisy cumsum position) | opt-in `decoder_anchor_identity` / `decoder_query_self_attn` (`transformer.py:269-278`), merged `4f9f373`+`91b86e3` 2026-06-24 but enabled nowhere (no real-data benefit on drag; washed out at 84K on cloth) | memory `project_decoder_identity_ab`, `project_cloth_v8_audit` |
| r6d Gram-Schmidt divergence at ~13K | identity-anchored rep `pose_ops.py:107-135` | `afd2bfd` 2026-06-21 |
| Oscillating RoPE gradients under guidance | `.detach()` on RoPE positions `transformer.py:965-966,1270` | `5a9da01` 2026-04-02 |
| Accelerate halves cosine `T_max` with N GPUs | `effective_T_max = steps * num_processes` `train.py:808-837` | memory `project_mpm_v6_1_audit_eval` |
| Delta-stat clipping erases fast frames | `clip_deltas` opt-in in `train_pickup_pose.py`; deployed v9 `noclip` | memory `project_rigid_component_audit` |

### 9. Doc/code drift found

`README.md:170-186` lists a non-existent `context_adaln`, wrong defaults for `all_relative`/`bypass_compression` (code: True, `transformer.py:89-90`), and says anchor-identity is "not merged". `CLAUDE.md` lists archived sweep paths and the superseded v-loss-for-sample rule. `docs/architecture.md`, `architecture_diagrams.md`, `pose_transformer_architecture.md` are gitignored (`.gitignore:55`) and predate joint/UNet/pose-native. `docs/tier2_v2_design.md:26` says `BODY_POSE_*_SCALE` are deleted; they remain (`transformer.py:45-46`).

### 10. Cleanup candidates

1. **VAE tier** (`models/vae.py`, `latent_transformer.py`, 4 root scripts, `data/latent_stats.py`, 2 tests, 12 launchers, `VAETransformerBlock`) — no inbound dependency; remove or move to `archive/`.
2. **`models/unet.py` + 21 `unet_*` configs + `models/components/pool_draft.py`** (dead, no importer) + `models/_unet_std_ref.py` (gitignored gate reference; keep only if the bit-identity gate stays).
3. **Guidance stack** (`guided_inference.py`, `guided_inference_pose.py`, `noise_opt_guidance*.py`, `interactive_pushing.py`, `scripts/*guidance*`, `scripts/archive/sweep_*`) — pushing-era, unaware of pose-native/UNet; archive or port.
4. **Dead `ModelConfig` flags**: `per_object_noise` (+ `run_per_object_inference_df`, `train.py:2696`, per-object schedules), `use_windowed_temporal` + `create_sink_window_mask`, `use_cross_object_edge`, `predict_body_poses` + `BODY_POSE_*_SCALE` + `configs/unified_v1_tier2.yaml`, `rope_position_scale`, legacy `AdaLayerNorm` (`attention.py:372`), `encoder_pointnet_embedded_in`; opt-in-but-unused `decoder_anchor_identity`, `decoder_query_self_attn`, `banded_rope`, `use_continuous_physics_adaln` (only in joint sweeps).
5. **Joint sweep configs** (~60 `sph64_*`, `mul*`, `movisphere_*`, `robert_mpm_object_v2_fast_*`/`arch_big*`/`dbg4tar*` + matching `sbatch_scripts/`) with no checkpoints in this checkout; keep `robert_mpm_c8_v6_1`, `robert_mpm_object_v2{,_c2,_fast}`.
6. **`archive/`** (MPM ablation configs, SIM1 v1–v7, `pickup_rigid_legacy` rigid transformers): nothing imports it; safe to leave or relocate.
7. **Contact stripper** (`utils/state_dict.py`): tiny; keep only while pre-`4cbcd70` checkpoints are loaded.
8. **Two r6d conventions** (`gripper_kinematics.r6d_to_R` raw vs `pose_ops.r6d_to_rotation` identity-anchored) — document or unify.
9. **FM `shift`** knob (dead on the DF path) and the stale `get_target`/`compute_loss` docstrings; decide the pose-term loss space for `sample` mode.
10. Docs: fix `README.md` flag table and `CLAUDE.md` paths; restore or remove the `docs/scope_unified_unet.md` references.

---

# 4. Unified dataset adapter, training recipes, evaluation protocols, checkpoints

Branch `robert/unified-unet-combined`, HEAD 7091f3a. All `file:line` pointers are to the working tree at that commit. "(unverified)" marks claims taken from memory/README text that I could not confirm against code or on-disk artifacts.

### 1. Adapter `data/unified_dataset.py` — the decisions baked into code

CLAUDE.md ("Before Launching Any Training", item 4) makes every constant below a DECISION requiring explicit user approval before any launch; origin 2026-08-11, when the unified-vs-specialist MPM comparison silently paired a `robert_mpm_v2`-trained unified model against a v1-trained specialist. Nothing in this table is reachable from YAML unless noted.

| Decision | Value | Location |
|---|---|---|
| Metric scale S (position stats = ±S ⇒ min-max normalize = p/S) | `UNIFIED_SCALE = 0.8` m | `unified_dataset.py:33`, `unified_position_stats()` :57-62 |
| Canonical per-domain rigid shift (support plane → z=0) | mpm (-0.75,-0.75,-0.10), pickup (-0.45,0,-0.10), pushing (0,0,0), cloth (-0.65,0,-0.80) | :35-40 |
| Material vocabulary | mpm raw 0..2 verbatim; RIGID=3, GRIPPER=4, CLOTH=5, `NUM_MAT_TYPES=6` | :44-48 |
| Fill physics for corpora without values | rigid E=1e9/ν=0.30; cloth E=1e5/ν=0.30/ρ=300; pushing ρ NaN→750 | :51-52, :409-412 |
| `DOMAIN_DT` | mpm 1/24, pickup 0.05, cloth 1/30, pushing **None** (memory: pushing is likely 500 fps, stale) | :54 (consumers: `build_manifest.py`, `velocity_study.py`, `build_samples_artifact.py` only) |
| Native windows | `DOMAIN_FUTURE_LEN = {mpm 16, pickup 38, cloth 16, pushing 16}`, C=2 default; `data.future_len>0` overrides uniformly, `data.context_len` threads through | :692, `build_domain_dataset` :695-727, `dataset.py:1801-1805` |
| Split conventions | pickup: last-10%-of-sorted shards **per root** (`_split_shards` :74-81, `val_fraction=0.10`); pushing: same rule over the pooled sorted list (`split_per_root=False` :709-713); cloth: pre-split on disk `{train,val}` by the 38-garment holdout (:716); mpm: train = `robert_mpm_v2/webdataset_full49`, val = `robert_mpm_v2_test_seen` (:630-631, :721) |
| Data roots | `GROUP=/data/group_data/katefgroup/VLA/3DPCWM` (:595); `GENESIS = GROUP` (:600, symlink farm resolved 2026-08-26); pickup 4 `robert_pickup_data_{place,convex,pick,upright}_success` (:602-603); pushing `pushing_data_actions` (:632); cloth `unified_v1/cloth/{train,val}` materialized (:633) |
| Blocklists | `HOLDOUT_BLOCKLIST` (142/946 pickup assets, applied to train AND val) :606; `PICKUP_BLOCKLISTS` = phantom combined + holdout, train only :608-611; `MPM_BLOCKLISTS` (64 collapsed eps, train only) :617-619; `CLOTH_BLOCKLISTS` train 415+3 / val 57 eps :626-629; wiring :707-708, :718, :724. `ALLOW_MISSING_BLOCKLISTS` (probe-only escape hatch, contaminates pickup holdout) :84-88, set from `data.allow_missing_blocklists` (`train.py:501-503`, `dataset.py:1795-1796`) |
| Stats sets | `STATS` (per-domain frozen files under `GROUP/stats/unified_v1`) :636-642; `STATS_SETS` {per-domain, iso_p9999, global_iso_p9999, xy_p99, xy_p99_global, unified_asinh, pd_asinh} :657-669; `POSE_STATS_SETS` :670-678; selected by `data.stats_location: unified://<set>` via `parse_stats_set` :681-689 |
| Sampling / interleave | `create_unified_dataloader` :795-843: domain-pure loaders, `rng.choice(names, p)` with `weights` default `{d: 1.0}` ⇒ **equal 25%** (:806, :824-827); domain-draw `seed=0` fixed and rank-independent (:821, rationale :730-735). ⚠ `domain_weights` and `seed` are read from `dataset_kwargs` in `dataset.py:1809,1811` but **`train.py` never populates them** (grep: no `domain_weights` in train.py), so the mix is not configurable from YAML today |
| Validation | `UnifiedValRoundRobin` :749-792: strict mpm→pickup→cloth→pushing rotation; `no_shard_split=True` so every rank sees every val shard (:771-778); `log.val_batches: 40` ⇒ 10 per domain |
| Bucketing | batches bucketed by particle count N (`dataset.py:440-444`); `bucket_max_items` flushes the oldest partial bucket **padded by repeating its own samples** (`dataset.py:445-456`); plumbed via `data.bucket_max_items` (`train.py:497-500`, `unified_dataset.py:768,813`) |
| Training window samplers | generic: uniform random `t0` in training, prefix at eval (`_window` :305-313); **cloth motion-aware**: `p(t0) ∝ (mean future cloth motion + 2e-5 m)^1.0`, class attrs `motion_alpha=1.0`, `motion_floor_m=2e-5` (:495-514), training only; mpm: parent `MPMMultiObjectDataset` random window (`dataset.py:909`) plus optional `train.window_movement_min_mm/power` gate (`train.py:511-516`, `dataset.py:872-901`); eval window pinned per episode key via crc32 seed (:464-470) |
| Cloth padding | `strip_padding=True` drops the 14000-row tail padding (:487, :526-529) — required by the UNet's ragged L0 |
| Actions | achieved displacement into frame f on actuated bodies, zero elsewhere (`_achieved_actions` :315-321); pushing derives from rod only, hard assert against label leak (:406-407); pickup actuated = `oid==1` (:346); cloth actuated = `oid>=1` (:532) |
| Pose-native sidecar (`emit_sidecar`) | `_rigid_sidecar_from_particles` Kabsch vs frame-0 templates → `body_poses [B,T,K,4,4]`, `body_templates`, `body_template_mask`, `body_type` (:156-189); `particle_body`; `_augment_pose_channels` → `pose_deltas_gt [B,F,K,6]` = `[Δt, so3_log(ΔR)]` from the C-1 anchor, `pose_actions`, `pose_action_mask`, `pose_delta_stats` (:195-225). Bodies: pickup K=4 (object, hand, lf, rf; segments from `link_local_npts[8:11]`, :336, :356-367), pushing K=2 (:416), cloth K=7 (cloth + 3 links × 2 grippers, :557-591) |
| Cloth sidecar file | `SIDECAR_IDS = GROUP/unified_v1/cloth_sidecar.npz` (:489); 10/1,223 episodes have unusable ids and raise `sidecar-skip` (:574-576) |
| `fps_random_start` | NOT an adapter constant: `ModelConfig.fps_random_start` default **False** (`models/transformer.py:104`), `unet_std` default **True** (`models/unet_std.py:128`, read from `unet_fps_random_start` :1325); set True in the tf333m config |

Also note `rope_spatial_base: 0.04` is config-level but re-derived for S=0.8 (`configs/unified_v1_tier1.yaml:15,74`).

### 2. `train.py` — loss composition and loop mechanics

- **Noising**: `x_0 = prediction_stats.normalize(prediction_x0, clip=clip_deltas)` (:1915-1916, clip default True); DF timesteps `[B,F]` with `low=1` (:1924); context noised only when `noise_context` (all_relative) (:1932-1945); `clean_ctx_timesteps` opt-in (:1965-1966).
- **Pose-native joint noising**: `pose_x0 = pose_delta_stats.normalize(pose_deltas_gt[:, :, rigid], clip=True)`, same `timesteps` (:1971-1979); model returns `(pred, pose_pred)` (:1986-1990); `pred is None` on pure-rigid batches (pickup/pushing) (:2038-2041).
- **Main term**: FM velocity target `noise - x_0` via `get_target`; `as_velocity` no-op in velocity mode; `sample` mode trains directly in x0 space (:2000-2006). Per-element MSE (:2031); optional `min_snr` weights (:2032-2037). Pose-native masks rigid rows out of the particle term (:2070-2073); `loss = mse[valid_mask].mean()` (:2088).
- **Pose term**: `pose_native_loss = mean((as_velocity(pose_pred) − v_target)²)`, added with **`lambda_pose_native`** (default 1.0; `lambda_pose` is the unrelated v8 gripper-aux) (:2090-2107). Same `min_snr` option (:2099-2104).
- **Aux losses available but OFF in every unified config**: `aux_losses` dict (pos_cum/accel/com/spread …, :2136-2147), `lambda_pose` (v8 FK head, :2161-2171), `lambda_body_pose` (Tier-2 v1 routed head Huber + drift, :2173-2216), `lambda_mesh` (:2218-2232), `supervise_context` (:2109-2134). UNet path asserts `loss_weighting == uniform` (:718-719) and restricts aux keys (:726-730).
- **Spike guard**: skip optimizer step if `loss > loss_spike_threshold × loss_ema` or `grad_norm > grad_norm_spike_threshold` (:2243-2251); defaults 10.0 / 50.0 (:1058-1059), `loss_ema_decay=0.99` updated on non-skipped steps only (:1061, :1402-1407); up to 8 spike batches dumped to `spike_batch_step*.pt` (:1386-1396); scheduler still steps on skipped steps (:1398), EMA does not (:1399-1400). ⚠ **`train.spike_grace_steps: 500` appears in every unified config but is read only by `train_pickup_pose.py:753`, never by `train.py`** — a dead key on this trainer.
- **EMA**: `EMA(unwrapped_model, decay=cfg.train.ema_decay, warmup=cfg.train.ema_warmup)` (:894-895); unified configs use 0.99.
- **LR**: cosine to 1% of lr, `effective_T_max = steps × num_processes` to compensate Accelerate's per-process scheduler stepping (:807-831); warmup 5000 in unified configs; `wsd` optional.
- **Checkpoints**: `best.pt` gated by `log.best_gate` (default `ema` val loss) (:1538-1548); `latest.pt` every val round (:1551); `step_<N>.pt` every `log.save_every` (:1556-1559; save_every was dead until 2026-07-18); `pre_decay.pt` for wsd (:1568-1578); `final.pt` at loop exit unless `log.save_final=false` (:1776-1778). Atomic tmp+`os.replace` (:3168-3206). Payload = step/model/ema/optimizer/scheduler/cfg (:3194-3201).
- **stop_after_step**: loop condition `while step < steps and not (stop_after and step > stop_after)` (:1301-1305) — the schedule still spans `train.steps`; `final.pt` lands at step 40001 for 40K arms (confirmed in checkpoints). Added in 9ce15ac (2026-08-27).
- **Resume**: `--resume` → `load_checkpoint` (model strict=False via `strip_contact_wrapper`, optimizer, scheduler; :3209-3229), scheduler rebuilt with correct T_max preserving `last_epoch` (:874-884), EMA reloaded (:1030-1031); val/sample rounds guarded by `step > start_step` (:1440, :1582).
- **Accelerate/DDP**: `find_unused_parameters=True` (or `static_graph` opt-in), `broadcast_buffers=False`, NCCL timeout `train.nccl_timeout_min` (:401-432); dataloaders deliberately NOT prepared (:886-890); val split must be disjoint (:572-645); per-domain val accumulation with draw-matched seed 12345 (:2788-2801); sampler-in-the-loop val `log.sampler_batches` (default 8) (:1473-1475, `validate_sampler` :3084+).
- **Pitfalls still live** (CLAUDE.md): use `batch.predictions_to_positions()`; EMA weights at inference; FM `sample` mode needs `as_velocity` before any Euler step and structurally needs `min_snr`; `all_relative` callers must pass `noisy_context`/`context_timesteps`; `scripts/` need `PYTHONPATH=.`; compile+DDP+gradient checkpointing needed `optimize_ddp=False`; the compile NaN in PointNet pool (`amax` backward) fixed at 8a03325.

### 3. Config inventory (154 YAML; 153 tracked, `cloth_fold_v10_mw.yaml` untracked)

#### Unified family (all: `format: unified`, `wds_location: unified://v1`, `emit_sidecar: true` except tier1, FM velocity + DF pyramid, `loss_weighting: uniform`, bf16, lr 1e-4, warmup 5000, `steps: 200000`, EMA 0.99, spike 10/50(/500 dead), `log.dir ./runs`, `val_every 1000`, `val_batches 40`)

| config | arch / params | C/F | stats set | bs/GPU | stop_after | save_every | launched by |
|---|---|---|---|---|---|---|---|
| `unified_v1_tier1` | classic 512×8, 81.7M (no sidecar, no pose head) | 2 / per-domain | per-domain | 4 | – | 10000 | `repro_sbatch/exp_unified_tier1.sbatch` (2×L40S) |
| `unified_v1_tier2` | classic + `predict_body_poses` (routed head v1), `lambda_body_pose 1.0` | 2 / per-domain | per-domain | 4 | – | 10000 | **none** (`exp_unified_tier2.sbatch` actually runs tier2v2) — superseded |
| `unified_v1_tier2v2` | classic + `pose_native`, 82.0M | 2 / per-domain | per-domain | 4 | – | 10000 | `exp_unified_tier2.sbatch`, `exp_unified_tier2v2_4gpu.sbatch` |
| `unified_v1_tier2v2_clean` | same | 2 | per-domain | 4 | – | 10000 | `exp_unified_tier2v2_clean.sbatch` (no run dir found) |
| `unified_v1_tier2v2_c4f16` | same | **4 / 16** | per-domain | 4 | – | 10000 | `exp_unified_tier2v2_c4f16{,_2g,_kq}.sbatch` |
| `unified_norm_A/B/C` | tier2v2 verbatim | 2 | per-domain / iso_p9999 / global_iso_p9999 | 4 | 40000 | 10000 | `exp_unified_norm_{A,B,B_katef,C,C_katef}.sbatch`; runs exist for B, C only |
| `unified_norm_xy_p99`, `unified_norm_asinh` | tier2v2 verbatim | 2 | xy_p99 / unified_asinh | 4 | 40000 | 10000 | `exp_unified_norm_{xy_p99,asinh}_katef.sbatch`; **no run dirs — never trained** |
| `unified_unet_c4f16` | unet_std 331.6M, `compile: true` | 4/16 | per-domain | 4, `bucket_max_items 64` | – | 10000 | `exp_unified_unet_c4f16_2g.sbatch`, `probe_unified_unet_c4f16_preempt.sbatch` (mbronars' checkout) |
| `unified_unet_c4f16_combined` | unet_std + merged flags (`unet_actions: false`, `unet_l1_rule ""`) | 4/16 | per-domain | 4, bmi 64 | – | 10000 | `exp_unified_unet_c4f16_combined_katef.sbatch` (2×A100); no run dir |
| `unified_unet_norm_{perdomain,xy_p99,asinh}` | combined verbatim | 4/16 | per-domain / xy_p99 / unified_asinh | 4, bmi 64 | 40000 | 10000 | `exp_unified_unet_norm_*.sbatch` (2×A100-40GB) |
| `unified_v1_tf333m_xy_p99_c4f16` | classic 768×15×12 heads, 333.0M, `gradient_checkpointing: full`, `fps_random_start: true`, compile off | 4/16 | xy_p99 | 4 | 40000 | 10000 | only overfit sbatches (`overfit_v2_*.sbatch`); **never trained on the corpus** |
| `unified_v1_tf333m_xy_p99_c4f16_200k` | same + `bucket_max_items 64` | 4/16 | xy_p99 | 4 (A100 script overrides 2) | 0 | 20000 | `train_tf333m_xy_p99_{a100x4,pro6000x2}.sbatch` — NOT LAUNCHED |

Param counts: tf333m 333,046,537 / unet 331,618,989 (from config headers + checkpoint tensor sums 333.0M / 331.6M); T2v2 82.0M and tier1 81.7M read from checkpoints.

#### Specialists (config → run → launcher)
`robert_mpm_c8_v6_1` (joint 576×12/192, C=8/F=16, v1 `robert_mpm/webdataset_full49`, `mpm_obj_v6_1_repro.sbatch`); `robert_mpm_drag_v6_1` (classic, C=2/F=22, `mpm_drag_v6_1_repro.sbatch`); `robert_mpm_object_v2{,_c2}` (joint, v2 data, C=8 / C=2; `_c2` has no sbatch reference); `robert_mpm_object_v2_fast{,_c2}` + `exp_mpm_obj_v2_fast6k_c{8,2}.sbatch`; `unetstd331_lightctx_sample_mpmfull` (unet_std, C=4/F=16, FM `sample`, `exp_unetstd331_mpmfull_a100.sbatch`); `pickup_pose_v9_4success_cfg{,_resume}` (`pickup_v9_repro.sbatch`, `exp_pickup_holdout.sbatch` with `train.clip_deltas=false`); `pickup_pose_v9_6d_4success_cfg` (`pickup_v9_6d.sbatch`); `pickup_pose_uncond_place_0810`; `sim1_variant_pose_rigid_v8` (`sim1_v8_repro.sbatch`); `sim1_cloth{,_x0,_fold_v9,_fold_v9_med}`, `cloth_fold_v10{,_mw}`; `pose_push_{cond,uncond}_df_0807`, `pose_push_uncond_04_18`, `pose_push_*_fullpoints`, `push_uncond_*`.

#### Legacy/debug/colleague families
`default`, `grasp_*`, `vae_*`, `latent_diffusion_*`, `uncond_debug`, `mpm_sand`; mbronars' unet R&D (`unet_*`, `unetstd*`, `sph64_*`, `mul{3,4,10}_*`, `movisphere_*`, `sedge166_split108`, `spikehunt_full_rope`, `bypass_*`, `direct_noreg_*`, `robert_mpm_object_v2_{dbg4tar,fast_lpo,fast_temporal,dit_recipe,arch_big}*`).

**42 configs have zero references** in sbatch/scripts/docs (list from grep): notably `unified_v1_tier2.yaml`, `robert_mpm_object_v2{,_c2}.yaml` (the c2 run IS the matched specialist — keep the config, add provenance), `pickup_holdout_eval.yaml`, `sim1_cloth_fold_v9.yaml`, `push_uncond_*` (4), and 27 mbronars `unet_*`/`unetstd_dfctx*`/`direct_noreg*`/`bypass_*` overfit configs.

### 4. Eval protocols and metric definitions

| Harness | Protocol | Metric (file:line) |
|---|---|---|
| `scripts/unified/eval_unified.py` `eval_domain` | one window per val episode, DF-pyramid full denoise from the checkpoint's own C/F (defaults read from `cfg.data` since 7abba79, :427-440; before 2026-08-30 the C=2 default silently mis-evaluated the C=4 UNets); pose-native poses read directly, else Kabsch on predicted particles | per-particle L2 mm, mean over valid particles and frames, episode-weighted; `particle_final_mm` = last frame (:81-86); `deform_*` on body_type-0 rows (:90-98); body pose: geodesic deg + trans mm per rigid body, split `pose_passive`/`pose_actuated` (:100-121) |
| `eval_unified.py --rollout` `rollout_domain` | overlap-commit strided AR with predicted pose track carried (:142-238); `--rollout_frames` (48 in the norm sweeps), `--dump_dir/--dump_k`, `--dump_pick max_motion`; `--chamfer_marks` = paper convention symmetric Chamfer = SUM of directional mean-NN distances ×100 (cm) on cloth rows (:293-301) | same mm/deg definitions (:276-318) |
| ⚠ sampler note | `eval_unified.py` always calls `run_diffusion_inference_df` (strided AR) for every architecture — it does **not** switch to streaming for UNets. The handoff README's line "the script selects the sampler by architecture" is not backed by this file. Streaming lives in `scripts/eval_object_rollout_unet.py` (S=50/frame, stagger 0.5, MPM only) and `eval_paper_mpm_unified.py --streaming` (:113-115) |
| `scripts/unified/eval_paper_mpm_unified.py` | Table-1 MPM: `robert_mpm_test/webdataset_full49/{1..4}obj`, first 60 files/group, AR to frame 48, context cropped so it ends at native frame 7 (:7-16, :49-51, :121); `--no_shift --stats_npz` for raw-frame specialists; per-frame mean per-particle distance **cm**, mean over frames 8-23 (`train_h`), 8-48 (`long`), frame 48 (`final`), then across files, then across groups (:158-166, :174-179); published ID 6.48/7.78/8.74 (:19-21) |
| `scripts/unified/eval_paper_cloth_table3.py` | 2 diverse_v1 bimanual episodes, stride-2 rollout from post-stride frame 30, symmetric-Chamfer-SUM cm at t=10/50/100/500/1000 (m = t+1) vs published 0.179/0.578/0.919/2.459/3.476 (:1-17, :39-41) |
| `scripts/unified/eval_specialist.py` | pushing specialist (plain transformer, format rl) on the unified val shards with the same overlap-commit code and mm/deg metrics (:1-13); caveat: pushing specialist had no shard holdout — specialist-favorable |
| `scripts/eval_pickup_pose.py` | replays `train_pickup_pose.validate()` with EMA; `--denoise oneshot|full_sequence|pyramid`; per-body trans mm (:77) + geodesic deg from `Rpᵀ Rg` (:79-82) |
| Fixed norm-A/B protocol | `scripts/unified/eval_norm_arm.sh` / `eval_norm_worker.sh`: `--rollout --rollout_frames 48 --seed 0`, stratified cells (mpm test_seen/unseen-00001+/ood per object count, pickup per task n=100, pushing n=200, cloth n=144) (:1-13, worker :48-58) |
| Overfit diagnostics | `scripts/unified/overfit_prod.py`: production model + one saved production batch per domain, `train.train_step`, real sampler every `--eval_every`; `--domain a,b,c,d` round-robin joint mode; `--holdout_name` scissor test (never trained, `fork_rng`, :322-328); 8-seed finals (:366-386); baselines zero-motion / hold-velocity (:246-247); `part_mm` = mean mm over valid particles+frames and final frame (:231-235), `vel_mm` jitter (:222-228). Results/pages: `reports/unified/overfit_v2/{runs,figs,round2_artifact.html,joint_artifact.html,joint_norm_artifact.html}` (untracked), builders `build_round2.py`, `build_joint.py`, `build_joint_norm.py` |

### 5. Checkpoints and results with receipts

| model | config | train corpus + vintage | steps | checkpoint | key numbers | protocol / receipt |
|---|---|---|---|---|---|---|
| Unified T2v2 82.0M (rebuttal) | `unified_v1_tier2v2` | unified v1: mpm **robert_mpm_v2** full49, pickup 4×success, cloth materialized, pushing_actions; C=2, per-domain F | stopped 134K/200K | `runs/unified_v1_tier2v2/checkpoints/best.pt` (118K), `latest.pt` (134K). ⚠ **`step_130000.pt` (the frozen rebuttal checkpoint) no longer exists** — snapshots deleted 2026-08-31; only JSON receipts remain | eval_unified@130K: mpm 24.6/37.2 mm, pickup 10.7 mm (obj 17.0°/19.8 mm, grip 0.71°/1.3 mm), cloth 0.37 mm, pushing 0.88 mm (block 2.0°/1.4 mm); Table-1 7.57/9.53/11.08 cm; rebuttal table cloth 20.8 vs 24.7, pushing 2.52 vs 5.89, pickup 10.4 vs 7.7, FallingObjects 75.7 vs 64.8 mm | `reports/unified/eval_tier2v2_130k.json`, `paper_mpm_table1_130k.json`; memory rebuttal |
| Unified c4f16 82M | `unified_v1_tier2v2_c4f16` (+ `_2g`) | same, C=4/F=16 | 61K (4g) / 67K (2g) of 200K | `runs/unified_v1_tier2v2_c4f16/checkpoints/{best@59K,latest@61K}.pt`; `_2g/{best@65K,latest@67K}` | tf@40K column in `norm_ab_trained_windows/final_comparison.txt` (pickup 13.6-18.4, pushing 6.7, cloth 5.0, mpm 94-160 mm); step_40000.pt deleted | `reports/unified/eval_c4f16_*_native.json` |
| Unified tier1 81.7M | `unified_v1_tier1` | same, C=2 | 137K (knocked @120.8K) | `runs/unified_v1_tier1/{best@81K,latest@137K}` | last clean point 120K | memory rebuttal :327-333 |
| Norm arms B/C (tf) | `unified_norm_B/C` | same | 40K | `runs/unified_norm_{B,C}/final.pt` (40001) | B beats A at 10-20K; C (global scale) refuted | `reports/unified/norm_ab/` |
| UNet norm arms | `unified_unet_norm_{perdomain,xy_p99,asinh}` | same, C=4/F=16, **unfiltered corpus (pre-blocklist)** | 40K | `runs/unified_unet_norm_*/final.pt` (40001, 5.31 GB) | xy_p99 wins 17/17 corrected cells (e.g. pickup 16.5-20.5 mm, pushing 7.9, cloth 5.6, mpm seen 134-162 mm) | `norm_ab_trained_windows/final_comparison.txt` |
| **Unified UNet xy_p99 40K (shared)** | `unified_unet_norm_xy_p99` | as above | 40K | `/data/group_data/katefgroup/VLA/3DPCWM/checkpoints/unified_unet_xy_p99_40k/final.pt` (stripped 2.65 GB) + config/stats/blocklists/README | same as row above | `reports/unified/HANDOFF_unified_unet_xy_p99_40k.md` |
| UNet unetstd331 mpmfull | `unetstd331_lightctx_sample_mpmfull` | robert_mpm_v2 full49, C=4/F=16, FM sample | 200K, 4×A100 | `runs/unetstd331_lightctx_sample_mpmfull_physaux/{final@200K,best@105K}.pt` | streaming Table-1 files ID 3.19/4.27/5.12 cm; unseen test 2.69/3.61/4.33; OOD 5-7obj 3.44/4.33/4.94 | `reports/unified/unet200k_final_stream_*.json`; memory unet_integration |
| MPM v2-c2 specialist 78.6M | `robert_mpm_object_v2_c2` | robert_mpm_v2 full49, C=2/F=16 | best 171K | `runs/robert_mpm_object_v2_c2/checkpoints/best.pt` ⚠ its `final.pt` is a **step-3 smoke overwrite** | Table-1 5.05/6.36/7.25 cm (beats published v6.1 6.48/7.78/8.74) | `reports/unified/paper_mpm_v2c2_specialist.json` |
| MPM object v6.1 C=8 (paper) | `robert_mpm_c8_v6_1` | **robert_mpm v1** full49 | best 136K | `runs/robert_mpm_joint_v6_1_c8/best.pt` ⚠ `final.pt` is a step-2 smoke overwrite; `_repro/best@107K` | published 6.48 | `inference_results/paper_eval/v6_1_full49/summary.json` (unverified on disk) |
| MPM drag v6.1 / v7 | `robert_mpm_drag_v6_1` | drag_setA | best 48K/100K | `runs/robert_mpm_drag_v6_1/best.pt`; reference `pwm_rope_decode_wt/runs/mpm_drag_v7_control/checkpoints/step_70000.pt` | v7 ctrl@70K 8.81/12.68 mm vs v6.1 12.02/17.80 | memory best_models_round |
| Pickup v9 4success | `pickup_pose_v9_4success_cfg` | 4×success, C=2/F=38 | 200K | group `checkpoints/v9_4success_cfg/best.pt` (redeployed 06-28 = `pickup_v9_noclip`); repro `runs/pickup_pose_v9_4success_cfg_repro_timefix/best@182K` | obj 12.10°/11.6 mm, grip 0.48°/2.3 mm (50-batch pyramid); corrected 7.73°/8.90 mm per memory | group README; memory pickup_v8_v9 |
| Pickup v10 asset-holdout | v9 cfg + `exp_pickup_holdout.sbatch` | 4×success minus 142 assets | 200K | `runs/pickup_v10_assetholdout/final.pt` | 12.76°/11.4 mm; held-out assets 11.25° | memory pickup_asset_holdout |
| SIM1 cloth v8 | `sim1_variant_pose_rigid_v8` | sim1_cloth_diverse_v1 stride 2 | best 59.5K/100K | `runs/sim1_variant_pose_rigid_v8/best.pt` (82.5M); `_repro/best@25K` | deployed; rebuttal specialist cloth 24.7 mm | memory cloth_v8_audit |
| Pushing cond / uncond | `pose_push_{cond,uncond}_df_0807` | push_data_08_07, C=2/F=30, 61M | 150K | `runs/pose_push_cond_df_0807/best@137.5K`; uncond `best@113K`; group `push_0807_{cond,uncond}/` | cond test 25.7 mm particle (block 37.9 mm/20.4°); uncond 99.2 mm | group READMEs; memory rebuttal :702-706 |
| Place uncond | `pickup_pose_uncond_place_0810` | place_success | 200K | `runs/pickup_pose_uncond_place_0810/{best@187K,final@200K}`; group `place_0810_uncond/` | best val_main 0.2026@187K; fulldenoise ~58 mm | group README |

Other run dirs under `runs/` (93 total): smokes (`unified_smoke_1gpu`, `unified_smoke_ddp2`, `unified_t2_smoke{,2}`, `v6_smoke_100M`, `_smoke_*`, `smoketest_c8`), `unified_v1_tier2_huberonly_arm` (T2 v1 routed head, `clean_7k_frozen.pt`, `latest_damaged_8k.pt`), `tier2v2_overfit_viz_*` (no checkpoints), pickup ablations (`pickup_v9_{ctx4,depth12,noclip,objrotw5,posegeo}`), `cloth_fold_v10` (⚠ `final.pt` step-6 smoke overwrite, `best@70.5K` real). No `step_*.pt` snapshots survive for any unified run.

### 6. The 200K staging that is NOT launched

- `configs/unified_v1_tf333m_xy_p99_c4f16_200k.yaml` = 40K arm + `stop_after_step 0`, `save_every 20000`, `bucket_max_items 64` (header lists each delta "awaiting the user's pre-flight OK 2026-09-03").
- `repro_sbatch/train_tf333m_xy_p99_a100x4.sbatch`: 4×A100-40GB katef, `data.batch_size=2` (effective 8), 2d23h30 with auto-resume from `latest.pt` (~3.5 days needed). `train_tf333m_xy_p99_pro6000x2.sbatch`: 2×RTX PRO 6000, B=4. `train_unet_xy_p99_resume200k.sbatch`: resumes `runs/unified_unet_norm_xy_p99/checkpoints/final.pt` with `train.stop_after_step=0 log.save_every=20000` on 2×A100.
- Open pre-flight decisions: (1) hardware (A100×4@B2 vs PRO6000×2@B4 — different per-GPU batch, same effective batch); (2) snapshot cadence 20K (5.3 GB each) vs 10K used by the baselines; (3) `bucket_max_items 64` for the transformer (the 40K tf333m arm was never run with it; it duplicates rare-N samples); (4) `fps_random_start: true` (veto-able, no transformer arm has trained with it); (5) **blocklist asymmetry** — the UNet xy_p99 baseline trained on the unfiltered corpus (2026-08-29) while the adapter now applies mpm/cloth blocklists by default, so a fresh tf333m run and a UNet resume see different train sets and different cloth val (80 vs 137/144 eps); (6) the compile-off decision for the transformer vs compile-on for the UNet; (7) `spike_grace_steps` is inert here.

### Cleanup candidates

- **Configs superseded**: `unified_v1_tier2.yaml` (routed head v1; superseded by `pose_native` tier2v2, sbatch points elsewhere); `unified_norm_A.yaml` (identical to tier2v2 + stop; its run was never made); `unified_norm_{xy_p99,asinh}.yaml` (transformer round-2 arms never trained; the UNet arms answered the question); `unified_unet_c4f16.yaml` (pre-merge; superseded by `_combined`); `unified_v1_tier2v2_clean.yaml` (no run found; (unverified) purpose). Unreferenced legacy: `push_uncond_{diffusion,flow,flow_long,flow_relative_df_rope_fix}.yaml`, `pickup_holdout_eval.yaml`, `sim1_cloth_fold_v9.yaml`, and the 27 mbronars overfit/R&D configs (`unet_*`, `unetstd_dfctx*`, `direct_noreg*`, `bypass_*`, `sedge166_split108`, `spikehunt_full_rope`, `robert_mpm_object_v2_{dbg4tar,fast_lpo}*`) — move to `archive/` unless mbronars claims them.
- **Sbatch superseded**: `exp_unified_norm_A.sbatch`, `exp_unified_norm_{B,C}.sbatch` (non-katef duplicates of `_katef`), `exp_unified_tier2.sbatch` (misnamed, runs tier2v2), `exp_unified_tier2v2_c4f16{,_kq}.sbatch` (4-GPU variants of the 2g run that was kept), `exp_unified_unet_c4f16_2g.sbatch` + `probe_unified_unet_c4f16_preempt.sbatch` (mbronars' checkout paths), `eval_unet_trained_windows_{a,b}.sbatch` / `eval_norm_*` (one-off sweeps, done), the `sbatch_scripts/` unet R&D family (`train_sph64_*`, `train_mul*`, `train_sphere_overfit1_*`, `unet_*`, `noise_walk_probe*`, `_mem_probe_a100`, `_speed_a100_probe`).
- **Code/doc drift to fix rather than delete**: `train.py` should either consume or drop `train.spike_grace_steps`; `domain_weights`/`seed` plumbing in `dataset.py:1809-1811` is unreachable from `train.py`; `HANDOFF_unified_unet_xy_p99_40k.md` overstates `eval_unified.py`'s sampler selection; `docs/scope_unified_unet.md` and `docs/unified_unet_deferred.md` are cited in code comments (`unified_dataset.py:87`, `dataset.py:445`) but absent from the repo; `DOMAIN_DT["pushing"]=None` is stale.
- **Runs**: smoke dirs (`unified_smoke_*`, `unified_t2_smoke*`, `v6_smoke_100M`, `_smoke_*`, `smoketest_c8`, `fit_test`), `tier2v2_overfit_viz_*` (empty), `unified_v1_tier2_huberonly_arm`; and the three `final.pt` files overwritten by smokes (`robert_mpm_object_v2_c2`, `robert_mpm_joint_v6_1_c8`, `cloth_fold_v10`) should be renamed `final_SMOKE_step*.pt` or removed so `best.pt` is unambiguous.

---

# 5. Infrastructure, environments and operational pitfalls

Verification context: commands were run from a `dev-a100` allocation (babel-y9-12, 2x A100-SXM4-40GB, 8 CPUs, 64G, 12 h). Every number below is from a command run today or a cited file; items I could not confirm are marked (unverified).

### 1. Conda environments

Two project envs, verified by running each env's `python -c` imports:

| | `fastrin_test` | `pwm_blackwell` |
|---|---|---|
| prefix | `/home/robertre/miniconda3/envs/fastrin_test` | `/data/user_data/robertre/conda_envs/pwm_blackwell` |
| python | 3.10.18 | 3.10.20 |
| torch | 2.5.1+cu121, arch list sm_50..sm_90 | 2.11.0+cu128, arch list sm_75/80/86/90/100/120 |
| pytorch3d | 0.7.9, `_C.so` contains **31x sm_89 only** (cuobjdump) | 0.7.8 source build, `_C.so` has sm_80/86/89/90/120 (31 kernels each) |
| webdataset / numpy | 0.2.111 / 1.26.4 | 0.2.111 / 1.26.4 (parity pins) |
| diffusers / accelerate | 0.35.2 / 1.12.0 | 0.35.2 / 1.12.0 |
| scipy | 1.15.3 | **missing** (ModuleNotFoundError) |
| extras | onnx 1.22.0, onnxruntime 1.23.2 (teleop Phase A) | omegaconf 2.3.1, einops 0.8.2, playwright (website screenshots) |

Rules that follow (memory `project_blackwell_env.md`, `project_data_audit_0903.md`, `project_overnight_unet_audit_0830.md`):
- **`fastrin_test` is L40S-only** (sm_89 pytorch3d). On A100 (sm_80) or RTX PRO 6000 (sm_120) every pytorch3d op fails with "no kernel image is available". `pwm_blackwell` works on any node type (A100/L40S/H100/PRO 6000).
- **The "no kernel image" trap**: the dev allocation can change node type between sessions (dev-blackwell babel-l9-16 to dev-a100 babel-y9-08 in Aug), which made a freshly merged tree look broken. Always `nvidia-smi -L` before interpreting a CUDA kernel error. The login shell alias `work` (`~/.bashrc:59`) activates `fastrin_test`, so the dev-node default python is the wrong one on A100; use `/data/user_data/robertre/conda_envs/pwm_blackwell/bin/python` explicitly.
- **scipy-needing CPU stages** (cloth audit stages, `fastrin_test python` for scipy) run with `/home/robertre/miniconda3/envs/fastrin_test/bin/python`; CPU-only is fine there.
- Unpinned `pip` pulls webdataset 1.x / numpy 2.x, which break the loaders (handoff README). pytorch3d rebuild recipe: `CUDA_HOME=/usr/local/cuda-12.9 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0" pip install --no-build-isolation git+...pytorch3d@stable` (~15 min; toolkits `/usr/local/cuda-11.7 .. 13.3` present).
- `~/.condarc` puts new envs under `/data/user_data/robertre/conda_envs` (14 envs there); the older envs (fastrin_test, genesis, sim1, newton, ...) live under `~/miniconda3/envs`.

**sbatch env lines (grep count, files containing the line):** `sbatch_scripts/` 40 fastrin_test, 0 pwm_blackwell (plus 2 `pctrl`); `repro_sbatch/` 31 fastrin_test, 31 pwm_blackwell (3 files contain both, as a commented katef alternative); `archive/` 24 fastrin_test. No katef-partition script activates fastrin_test without a pwm_blackwell line, so the current stale-env hazard is confined to `general`/L40S scripts, where fastrin_test is correct.

### 2. SLURM

Verified with `sinfo`, `sacctmgr`, `scontrol`:

| item | value |
|---|---|
| partition `katef` (TIMELIMIT 3-00:00:00) | `babel-x9-[24,28]` gpu:6000Ada:8, `babel-l9-[16,20]`+`babel-z9-16` gpu:RTX_PRO_6000:8, `babel-y9-[08,12]` gpu:A100_40GB:8 |
| account / QoS | `kfragki2`; QoS `katef_qos` (MaxTRESPU gres/gpu=16, MaxJobsPU 10, MaxWall empty, priority 20000), `katef_cpu_qos` (gres/gpu=0, MaxJobsPU 10); also `preempt_qos`, `preempt_cpu_qos` |
| L40S | not on katef; `--gres=gpu:L40S:{1,2,4}` scripts use `--partition=general` (131 sbatch headers use general, 52 katef) |
| gres strings in use | `gpu:L40S:2` (67), `gpu:L40S:1` (61), `gpu:A100_40GB:1` (31), `gpu:A100_40GB:2` (12), `gpu:A100_40GB:4` (4), `gpu:L40S:4` (4), `gpu:RTX_PRO_6000:1` (3) |
| typical header | `--nodes=1 --ntasks=1 --cpus-per-task=8/16/32 --mem=64G..256G --time=48:00:00 / 2-00:00:00 / 3-00:00:00` |
| logs | `--output=slurm/output/%x_%j.log --error=slurm/error/%x_%j.log` (42 scripts; 61 logs each, 5.5 MB total) or `/data/user_data/robertre/logs/` (59 scripts; 520 files, 1.5 GB) |
| exclusions | `--exclude=babel-q5-20,babel-q5-24,babel-s5-24` in 15 general-partition scripts |
| NCCL | `NCCL_P2P_DISABLE=1` in 113 scripts (general/L40S workaround); deliberately NOT set on katef A100 (NVLink; it halved throughput, `exp_unetstd331_mpmfull_a100.sbatch`) |

Operational rules:
- **`scontrol update` on a pending job**: memory records the user fixing a double-stuck job with `scontrol update jobid=... partition=katef` then `Gres=gpu:A100_40GB:4` ("Job's QOS not permit" on general; BadConstraints for L40S on katef). A `TimeLimit=` update is the same mechanism (unverified in memory). `scontrol release`/`ExcNodeList` also used.
- **Dev node**: the Claude session runs inside the dev allocation, so a bare `srun` inherits `SLURM_JOB_ID` and hangs ("step creation disabled"); use `sbatch` or run directly on the node with `CUDA_VISIBLE_DEVICES=0` (GPU 1 is the user's; the exact "gpu:1" wording is unverified). Dev job has 8 CPUs: never run more than one CPU-heavy job at once (pytest took 42 min contended), and never run torch CPU `cdist` with default threads on the shared node.
- A dev-allocation refresh kills background evals AND wipes `/tmp` task logs and scratchpad; only results under the repo (`reports/`) survive.
- Dev allocations are requested as `dev-a100` / `dev-blackwell` jobs (12 h); check `squeue -u robertre` for the current one.

### 3. Storage

`df -h` today:

| mount | size / used / free | note |
|---|---|---|
| `/data/group_data/katefgroup` (nas10) | 9.9T / 802G / 9.1T (8%) | group root |
| `/data/group_data/katefgroup/VLA` | **23T / 23T / 184G (100%)** | `VLA/3DPCWM` = GROUP root for all unified data + `checkpoints/` handoffs + `stats/unified_v1/`; avoid large writes |
| `/data/group_data/katefgroup/datasets` | 86T / 77T / 9.1T (90%) | |
| `/data/user_data/robertre` (nas9) | 1.0T / 931G / 94G (91%) | mode `drwx------` (700): unreadable to colleagues, so anything shared must be copied to group paths |
| `/tmp` (local NVMe) | 3.5T / 850G / 2.7T | periodically wiped |

- Quota vs space: in Aug the katefgroup NFS returned ENOSPC on `touch` while `df` showed 17T free (quota, not disk); user_data hit 100% on 08-12 and snapshot ladders were pruned. `quota -s` fails here (permission denied on an unrelated mountpoint).
- Repo footprint: `runs/` **313 GB** (94 run dirs), `inference_results/` 31 GB, `wandb/` 2.7 GB, `slurm/` 5.5 MB, `runs_smoke/` 335 KB (the 5 GB smoke checkpoints are already gone), `website/data` 29 MB.
- **NFS contention + eager_load** (`project_nfs_eager_load.md`): user_data is ~10 MB/s under contention; 16 worker processes on the same `.npz` files stalled the first batch 4+ min. Fix: `data.eager_load: true` (+ `num_workers: 2`, `num_collate: 4`; 29 s uncached on NVMe, 5+ min on contended NFS) or pre-copy to local NVMe. `np.load(mmap_mode)` does not work on `.npz`.
- **/tmp policy**: never use `/tmp` for outputs (wiped twice, losing a multi-hour session's REPORT/plots); write under `/data/user_data/robertre/...`. Cross-node scratch: `/data/user_data/robertre/scratchpad/` (exists, currently `joint_smoke/`). The Claude session scratchpad is itself under `/tmp/claude-*` and gets wiped by the dev refresh; keep artifact HTML sources under `reports/`.
- `/data/group_data/katefgroup/VLA/3DPCWM/stats/unified_v1/` holds the 7 unified-v2 baseline stats + `blocklists/` + README; only 4 of the 7 (`unified_v1_pose_delta_{pickup,cloth,pushing}`, `robert_mpm_train_1to4obj_v3_p9999`) are tracked in git, so the group copy is the only way colleagues get the other 3.

### 4. Git practice

- Remote `git@github.com:RobertRen1122/particle_world_model.git`, **PRIVATE** (`gh repo view`: `isPrivate true`, default branch `main`, last push 2026-09-04 23:11Z). Colleagues need access to be granted.
- Current branch `robert/unified-unet-combined` at `7091f3a`, in sync with origin (pushed 2026-09-04, 52 commits). `main` tip `6a1fa6a` 2026-07-24. 44 local / 52 remote branches; several sibling worktrees share this `.git` (`git worktree list`), some of them private research branches that exist only locally and must never be pushed.
- `.gitignore` blanket rules: `*.pt *.pth *.ckpt checkpoints/ results/ logs/ wandb/ *.tar *.zip *.npz *.gif *.png *.txt *.json runs/ docs/ inference_results/ scripts/analysis/ paper_figures/ CLAUDE.md .claude`, plus archived scratch dirs. Consequences: `.npz`/`.json` need `git add -f` (345 force-added files tracked: 161 under `reports/unified/norm_ab`, 57 `stats/`, ...); `docs/` is ignored so only 4 docs are tracked (`loss_viz_rules.md`, `tier2_v2_design.md`, `unified_dataset_spec.md`, `unified_unet_design.md`); **`CLAUDE.md` is NOT tracked** despite its own header claiming it is checked in; `website/` and `.github/` are untracked (`??`).
- Commit trailers in use (`git log -5 --format=%B`): `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` on all 5, `Claude-Session: https://claude.ai/code/session_...` on 3 of 5. Subjects use a `scope(area): summary` style (`handoff(unified):`, `report(unified):`, `norm(unified):`, `audit(data):`).

### 5. Experiment hygiene rules (feedback memories + CLAUDE.md)

1. **Mandatory pre-flight before any training launch or model comparison** (CLAUDE.md, `feedback_confirm_training_setup.md`, origin 2026-08-11 robert_mpm v1/v2 mismatch): confirm (a) full realpath'd data paths incl. vintage and size, (b) eval protocol with provenance of every comparison number, (c) train-corpus/split/sampling consistency vs baselines, (d) code-baked design constants (`DOMAIN_FUTURE_LEN`, weights, split rules). Explicit user OK before `sbatch`.
2. **Confirm the plan before autonomous stretches** (`feedback_confirm_plan_before_autonomous.md`, 2026-09-02): list jobs with "overfit" vs "training" labels, GPUs, outputs, page contents; deliver only the literal ask.
3. **Never edit a running bash script** (bash reads lazily; an in-place rewrite killed two re-scoring runs with "unexpected EOF", 2026-08-26). Check `pgrep -f`, write a new path.
4. **No /tmp** for anything that must survive.
5. **Canonical group paths when sharing**: realpath to `/data/group_data/...`, `cd "$SLURM_SUBMIT_DIR"` instead of personal checkout, flag env names, `git add -f` for npz/json.
6. **Loss plotting** (`docs/loss_viz_rules.md`): name the estimator on the axis; gate on a deterministic number (`scripts/unified/lossprobe.py`: fixed batch, fixed t grid, fixed noise seeds); condition on t (FM loss spans ~370x across t); raw faint + estimator bold; print scatter next to effect; read JSON not stdout; guards fail loud; shared batch/axes/steps; look at the image.
7. **Paper/rebuttal prose**: no em dashes, colons, or semicolons (header colons OK).
8. **Blender render default** (`feedback_default_render_style.md`): across-the-table camera pos (2.2, 0, 1.8) target (-0.2, 0, 0.7) 50 mm, white background (HDRI disconnected, backdrop meshes >5 m hidden, film_transparent + composite), Cycles (`MeshRasterizer` = workbench grey, no textures), >=1280x768.
9. `docs/coding_guideline.md`: think before coding, simplicity first, surgical changes, goal-driven verification; `WANDB_MODE=disabled` for any test training.

### 6. Tooling around the repo

- **wandb**: `train.py` logs via `utils/logging.py` (`init_wandb`, `log_with="wandb"` in Accelerate), project `particle_world_model`, entity null (configs `log:` block). 19 scripts set `WANDB_MODE=disabled`, 2 `offline`, 1 `online`.
- **website/** (untracked, 2026-08-24/25): static WebGL2 site (`index.html`, `js/viewer.js`, `js/ui.js`, `css/site.css`), packer `tools/pack_data.py` (npz to `data/<id>.bin.gz`, int16 delta-coded, 29 MB packed), `tools/shoot.py` (Playwright screenshots, needs pwm_blackwell python; never full-page captures >8192 px), `tools/export_place.py`, `_shots*/` dirs, `.github/workflows/pages.yml`, own `.gitignore` that un-ignores json/png/gz.
- **Blender sandbox**: `/data/user_data/robertre/blender_render_v2/` **does not exist** (deleted 2026-08-24 15:30; scripts backed up in `_cleanup_backups_2026-08-24/`, per `project_website.md`). The render-style memory still points at `blender_render_v2/scripts/render_custom_camera.py`.
- **Teleop deploy Phase A** (2026-06-30): scripts and `lightweight_report.html` in `/data/user_data/robertre/particle_world_model_archive/lightweight_audit/` (phaseA_eval.py, phaseA_streaming_pose.py, phaseA_latency_export.py, ort_parity_check.py, `pickup_export_scan.onnx`). Server-side streaming DF pyramid 73 ms/frame on L40S; ONNX export not numerically faithful; nothing in the main repo references `stream_pyramid`.
- **Artifact pages** (verified via artifact listing): "Unified Overfit" (0063d527), "Corpus Audit" (9a0e936c), "One Scale or Four?" (c57a3ba8), "Lever Arm" (bec5b394, 2026-09-03), "Unified UNet, One Fold" report (5f34f3fa), atlas "Unified UNet, Explained" (7395ed05), playground "Cloth Fold Tokenizer" (8e1daf28), "Overfit Grid" (a379718d), "Overfit Round Two", "The Capacity Match", "Overfit Batches", "One Step or Fifty", "Deterministic Head", "Unified Tier-2 v2, Explained", "Unified model — rollout viewer"/"paper eval sets", "Tier-2 v2 — Pose-Native Architecture". Artifact sources live under `reports/unified/...` (e.g. `overfit_v2/joint_artifact.html`, `unet_probe/_build_atlas.py`).
- **Memory directory** `/home/robertre/.claude/projects/-data-user-data-robertre-particle-world-model/memory/` = the project log: `MEMORY.md` index + 47 `project_*.md` + 7 `feedback_*.md` (mode 750, owner robertre).

### 7. Deployed checkpoint handoff dirs (`/data/group_data/katefgroup/VLA/3DPCWM/checkpoints`, setgid `kfragki2_group`)

| dir | owner / date | contents (verified `ls`) | what it is (memory) |
|---|---|---|---|
| `v9_4success_cfg` | robertre, Jun 29 | `best.pt` 1.28 GB (redeployed time-fix + noclip), `best_pretimefix_20260523.pt`, config, stats, blocklist, README | pickup pose v9, 4 success datasets, CFG 70/30; gripper 0.48 deg / object 12.1 deg vs TRUE |
| `v8_place_success_cfg` | robertre, May 23 | best.pt, config, stats, blocklist, README | pickup v8 place-only (val 0.0777) |
| `v7_place_cfg_clean` | robertre, May 21 | best.pt + 1 MB blocklist | earlier place model |
| `push_0807_cond` / `push_0807_uncond` | robertre, Aug 11 / Aug 24 | best.pt + final.pt (~0.98 GB), 2 stats npz, README | rebuttal pushing pose models; cond test 25.7 mm, uncond 99.2 mm (expected uncond divergence) |
| `place_0810_uncond` | robertre, Aug 11 | best/final 1.28 GB, place stats, strict_v2 blocklist, README | uncond place; fulldenoise ~50-58 mm |
| `unified_unet_xy_p99_40k` | robertre, Sep 4 | `final.pt` 2.65 GB (step 40001, optimizer stripped), config, `stats/` 14 npz, `blocklists/` 4 json, README | 331M unified UNet, 4 domains, per-domain xy_p99 stats, trained pre-blocklist |
| `production_runs` | mbronars, Jul 30 | 6 MPM stats npz only | colleague's stats for the MPM UNet lineage |
| `push_08_07_unetL_raw256_..._drop50` | mbronars, Sep 3 | config + `final_step100000.pt` 1.8 GB | colleague's pushing UNet |
| `rigid_push_uncond{,_diverse}`, `rigid_gripper_cond` | robertre, May 13-17 | `latest.pt` ~0.98 GB, stats, README | legacy rigid pushing family (archived code) |
| `gripper_uncond{,_0404,_better_stats}` | robertre, Mar-Apr | best/latest.pt 1.3 GB, stats | earliest unconditional pushing models |
| `sim1_stats` | robertre, May 23 | `sim1_cloth_diverse_v1_stride2_stats.npz` | SIM1 v8 stats |

### Cleanup candidates

- **`sbatch_scripts/`: 82 of 119 files write logs to `/data/user_data/mbronars/logs/` and 77 `cd` into `/data/user_data/mbronars/packages/...` with `PATH=.../pctrl_a100/bin`** (colleague's UNet scripts inherited by the merge); unrunnable from this account (`ls /data/user_data/mbronars` = permission denied). Move to `archive/` or rewrite with `$SLURM_SUBMIT_DIR`.
- 95 sbatch files hardcode `cd /data/user_data/robertre/particle_world_model` (only 1 uses `SLURM_SUBMIT_DIR`), violating the canonical-paths rule for anything shared.
- 40 `sbatch_scripts/*` + 24 `archive/*` scripts activate `fastrin_test`; fine for `general`/L40S but they will silently break if retargeted to katef. The three `repro_sbatch/exp_unified_norm_{A,B,C}.sbatch` carry the katef alternative only as a comment.
- `repro_sbatch/exp_unified_tier2v2_c4f16_kq.sbatch` still has `general` + `gpu:L40S:4` although it was run on katef via `scontrol update` (memory line 425).
- `/data/user_data/robertre/logs/` (520 files, 1.5 GB) and `slurm/{output,error}` (61 + 61 logs) accumulate with no rotation; `wandb/` 2.7 GB local cache; `runs/` 313 GB on a 91%-full volume (previous prune on 08-12 freed 83 GB).
- `.gitignore` ignores `docs/` and `CLAUDE.md`; CLAUDE.md claims to be checked in but is not, and 30+ docs are untracked. Decide whether to un-ignore.
- Stale pointers in memory: `feedback_default_render_style.md` and `project_v8_render_and_v9_plan.md` reference the deleted `blender_render_v2/` scripts.
- `website/` and `.github/workflows/pages.yml` are untracked despite being a deploy target.

---

# 6. Codebase inventory and cleanup map

All numbers below come from commands run read-only in the repo. The per-file raw tables (every script/config/launcher with date, referrers, docstring) (`docs/cleanup_2026-09/{scripts_full,configs_refs,launch_repro,launch_sbatch}.tsv`) are in git history, last present at `79a3c84`.

Reference set used for "referenced by": `repro_sbatch/ sbatch_scripts/ scripts/ docs/ reports/**/*.md CLAUDE.md README.md tests/ configs/ cleanup_docs/ archive/README.md` + core code (`data models utils diffusion train*.py inference*.py guided_inference*.py`), basename match plus Python import match, excluding the file itself. Referrer abbreviations: **C**=CLAUDE.md, **R**=README.md, **cd**=cleanup_docs, **d**=docs, **sb**=sbatch_scripts, **rp**=repro_sbatch, **t**=tests, **cfg**=configs, **grp**=a README in `/data/group_data/katefgroup/VLA/3DPCWM/checkpoints/*/`.

### 1. Top-level layout

1218 tracked files (~67 MB). The repo directory on disk is dominated by gitignored artifacts.

| Entry | Tracked | Size | Purpose / note |
|---|---|---|---|
| `train.py` (3272 l), `train_pose.py`, `train_pickup_pose.py`, `train_vae.py`, `train_latent_diffusion.py` | 5 | — | training entrypoints; `train.py` last commit 08-30, VAE/latent ones 03-08 |
| `inference.py`, `inference_pose.py`, `inference_pickup_pose.py`, `inference_vae.py`, `inference_latent_diffusion.py` | 5 | — | rollout entrypoints; `inference.py` 08-27, VAE/latent 02-22/03-06 |
| `guided_inference.py` (1788 l, 04-19), `guided_inference_pose.py` (1733 l, 06-21) | 2 | — | guidance; referenced by C/R/cd/scripts (16 and 5 refs) |
| `noise_opt_guidance.py` (1185 l), `noise_opt_guidance_pose.py` (679 l) | 2 | — | DOODL noise-opt variants; refs: 1 (`scripts/test_doodl_noise_opt.py`) and **0** |
| `interactive_pushing.py` (545 l) | 1 | — | interactive pushing demo; 4 refs |
| `README.md` (06-24), `pyproject.toml`, `.gitignore` | 3 | — | README references a missing `docs/DECODE_INVESTIGATION.md` (it lives in `cleanup_docs/`) |
| `CLAUDE.md` (38 KB) | **no** (gitignored) | — | project instructions; local-only, not in git |
| `data_subsets` | tracked **symlink** | — | → `../pwm_production/data_subsets`, **dangling** (target missing); committed 08-01 |
| `configs/` | 153 | 5.2 M | +1 untracked (`cloth_fold_v10_mw.yaml`) |
| `models/` (14), `data/` (14), `utils/` (17), `diffusion/` (4) | 49 | ~6 M | core library; `models/_unet_std_ref.py` is an ignored local file |
| `scripts/` | 223 | 15 M | see §2; 9 ignored `apr_06_*.sh` + 4 ignored `scripts/analysis/*.py` on disk |
| `sbatch_scripts/` | 123 | 4.3 M | historical launchers (§4) |
| `repro_sbatch/` | 61 (+1 untracked) | 2.2 M | current-era launchers (§4) |
| `tests/` | 37 | 15 M | 242 tests collected; 11 M of ignored PNG/GIF outputs in `tests/` |
| `stats/` | 57 (99 on disk) | 4.6 M | 42 untracked `.npz/.png` incl. several that live configs need (§11, Risks) |
| `archive/` | 78 | 4.2 M | previous cleanup's parking lot (§6) |
| `cleanup_docs/` | 5 | 230 K | previous cleanup provenance (§6) |
| `docs/` | 4 tracked / **gitignored dir** | **481 M** | 4 force-added `.md`; 22 ignored `.md` + ~475 M of paper/LaTeX/zip assets (§7) |
| `reports/` | 409 | **1.4 G** | 486 untracked files; some files are LIVE code deps (§8) |
| `runs/` | ignored | **313 G** | 94 entries; a running dev job is writing new run dirs here today — check `squeue` before pruning |
| `inference_results/` | ignored | 31 G | 84 result dirs, oldest from grasp_30k era, newest `pickup_v2`/`rebuttal_gifs` |
| `checkpoints/` | ignored | 3.7 G | two May pose checkpoints (`world_pose_{cond,uncond}_df`) |
| `wandb/` | ignored | 2.7 G | 33 online + 4 offline runs (06-26 → 08-29) |
| `website/` | **untracked** | 99 M | ParticleDiffuser project site; `_shots*` screenshot dirs = 62 M; `.github/workflows/pages.yml` (untracked) deploys it |
| `rebuttal/` | untracked | 27 M | CoRL rebuttal LaTeX + 27 MB paper PDF + zip |
| `paper_figures/` | ignored | 91 M | 153 png/gif, newest 05-23 |
| `debug_vis/` | ignored | 32 M | 87 files, newest 09-04 |
| `slurm/` | untracked | 5.5 M | 122 SLURM logs (08-28 → 09-04) |
| `runs_smoke/`, `sweep_0404/`, `guidance/`, `vis_diffusion/`, `vis_flow_matching/` | untracked/ignored | 0.3–2.3 M each | smoke outputs; Apr guidance sweep launchers (10 sh/sbatch); 6 stale `.pyc`; 9+9 png |
| `__pycache__` ×26 dirs, `.pytest_cache`, `particle_world_model.egg-info`, `.codex` (0 B), `.claude/` | ignored | ~1.2 M | caches |

### 2. scripts/ (223 tracked: 221 `.py/.sh` + `refactor/REPORT.md` + `refactor/.gitignore`)

| Subdir | Tracked | Unreferenced & <07-01 (archive cand.) | Unreferenced & recent |
|---|---|---|---|
| `scripts/` root | 109 | 42 | 23 |
| `scripts/unified/` | 57 (+1 untracked `overfit_denoise.py`) | 0 | 35 |
| `scripts/archive/` | 21 | 12 | 0 |
| `scripts/normalization/` | 13 | 6 | 0 |
| `scripts/sim_vs_model/` | 11 | 0 | 5 |
| `scripts/refactor/` | 5 | 1 | 0 |
| `scripts/inference_mpm/` | 5 | 3 | 0 |
| `scripts/analysis/` | 2 (dir is gitignored; 4 more contact-era files ignored on disk) | 1 | 0 |
| **Total** | 221 | **65** | **58** (98 referenced) |

#### 2a. Referenced scripts (98)

| Script | Last commit | Referenced by | Purpose |
|---|---|---|---|
| `eval_pickup_pose.py` | 06-26 | C, R, cd, d×2, `train_pickup_pose.py` | pickup ckpt eval: val loss + per-body mm/deg |
| `long_rollout.py` | 04-19 | C, R, cd, `long_rollout_{pose,sim1}.py`, ignored `apr_06_*.sh` | long-horizon streaming rollout |
| `long_rollout_pose.py` | 06-20 | cd, **grp** (rigid_push_uncond*) | pose-model streaming rollout |
| `long_rollout_sim1.py` / `long_rollout_sim1_variant.py` | 04-21 / 06-21 | each other, teleop, `block_rollout`, `streaming_*`, cd | SIM1 cloth rollouts |
| `block_rollout_sim1_variant.py`, `streaming_{proper,warm}_pyramid_sim1_variant.py` | 06-20 | cd/SIM1, d/physctrl, `unified/eval_cloth_specialist.py` | SIM1 v8 block/streaming baselines |
| `build_sweep_report.py`, `check_novel_difficulty.py`, `test_all_guidance.sh`, `test_novel_difficulty.sh` | 04-17..06-21 | C, R (+`guided_inference.py`, d/knn_noise) | guidance sweep tooling |
| `run_guidance_diag.sh` → `summarize_guidance_diag.py` | 04-19 | d/improvement_ideas | guidance diagnostic sweep |
| `debug_vae_recon.py` | 02-23 | C, R | VAE recon debug |
| `teleop_sim1.py` | 04-19 | `teleop_sim1_rerun.py`, `test_teleop_sequence.py` | keyboard teleop w/ cloth prediction |
| `sweep_f9_clean.py` | 06-20 | `n20_*.sh`, `sweep_f12_new.sh`, archive/run_overnight, **grp** | F9 clean pose-guidance sweep |
| `classify_phantom_modes.py` | 06-20 | 14 sweep/test scripts (10 in `scripts/archive/`) | phantom-mode classifier |
| `save_utils.py` | 06-20 | `sweep_f9_clean.py`, archive/sweep_tier1 | sweep viz helpers |
| `n20_best_noise.sh` | 06-20 | `n20_novel_easy_perp.sh`, `sweep_f12_new.sh` | n=20 noise-opt runs |
| `test_doodl_noise_opt.py` | 06-21 | cd, `sweep_doodl_*.sh`, **grp** | DOODL smoke (only user of `noise_opt_guidance*.py`) |
| `test_multiwindow_04_18.py`, `test_multiwindow_noise.py` | 06-21 | cd, each other, **grp** | multi-window noise-opt tests |
| `build_4success_combined_blocklist.py` | 06-21 | cd×2 | builds the LIVE v9 blocklist (§8) |
| `check_rigid_pose_head.py`, `test_gripper_kinematics_roundtrip.py`, `test_v8_local.py` | 06-21 | cd only | v8 rigid-head checks |
| `compare_guidance.py`, `debug_action_mask.py`, `unguided_baseline.py`, `visualize_{knn_guidance,pose_delta_distribution,x0_estimate}.py`, `test_teleop_sequence.py` | 06-21 | **cd only** (+d/knn_noise for 2) — the CLEANLINESS report *mentions* them, nothing runs them | guidance-era viz/tests |
| `inspect_pickup_data.py` | 06-21 | `normalization/compute_stats_pickup_pose.py` | pickup data inspection |
| `eval_object_rollout.py` → `eval_object_rollout_unet.py` | 07-28 / 08-27 | sb×3 (`eval_unetstd*`), `make_eval_readme.py` | MPM v2 object-model eval (transformer / unet) |
| `unet_std_launch_snapshot.py` | 08-05 | sb×3, 8 analysis/probe scripts, `calibrate_aux_losses.py` | frozen copy of `models/unet_std.py` at the unetstd launch |
| `eval_firstimpact_error_prod.py`, `per_axis_error.py` | 07-31 | sb/eval_sph64_sweep | first-impact / per-axis error |
| `noise_walk_probe.py`, `noise_walk_census.py` | 07-31 | sb/noise_walk_probe×4, `models/joint_transformer.py` (docstring) | delta-noising random-walk probes |
| `analysis_generation.py` → `analysis_metrics.py` → `analysis_plots.py`; `analysis_sequential.py` ← `analysis_restart.py` | 08-05 | intra-chain only | unetstd failure-analysis harness |
| `calibrate_aux_losses.py` | 08-05 | cfg/unetstd_dfctx_sample_mpmfull_physaux, `utils/phys_losses.py` | aux-loss lambda calibration |
| `eval_pickup_longhorizon.py` | 08-07 | `build_longhorizon_artifact.py` | long-horizon pickup eval |
| `overfit_viz_pose.py` | 08-22 | `overfit_viz_pickup_pose.py` | rigid-pose overfit render |
| `refactor/regression_harness.py`, `refactor/multi_batch_check.py` | 06-21, 05-02 | cd, `refactor/REPORT.md` | TransformerBlock refactor gates |
| `analysis/analyze_vae_latent.py` | 02-22 | `analysis/run_analyze_vae_latent.sh` | VAE latent viz |
| `normalization/compute_latent_stats.py` (+`run_compute_latent_stats.sh`) | 02-23 | 3 `run_*.sh`, `train_latent_diffusion.sh` | latent stats |
| `normalization/compute_stats_rl.py` | 04-27 | `compute_pose_stats.py` | RL stats |
| `normalization/compute_stats_mpm.py`, `compute_stats_pickup_pose.py` | 06-20/21 | cd×2 each | LIVE stats recipes (MPM drag, pickup v9) |
| `normalization/compute_stats_full_isodist.py` ← `compute_stats_local_isodist.py` | 08-05 | cfg/unetstd331_lightctx | isodist stats |
| `inference_mpm/inference_mpm_multi.py` | 06-21 | cd×3, d/flame_viz, `inference_mpm.sh`, `sweep_physics.py`, `unified/eval_paper_mpm_unified.py` | MPM h5 inference (paper Table-1 path) |
| `inference_mpm/sweep_physics.py` | 02-23 | `sweep_physics.sh` | E/nu/rho sweep |
| `archive/sweep_tier1_guidance.py` | 06-20 | C, 11 sibling sweeps, `save_utils.py` | Tier-1 pose guidance sweep |
| `archive/{compare_all_phases,plot_novel_results,sweep_action_opt,sweep_joint_opt,sweep_noise_hparam,sweep_noise_pose,sweep_novel_pose}.py` | 06-20 | **C only, at stale paths** (`scripts/sweep_*.py`) | pose-guidance phases |
| `archive/analyze_phantom_moves_pose.py` | 06-20 | `archive/sweep_phantom_weight.py` | phantom analyzer |
| `sim_vs_model/{compare_sim_vs_model,run_simulator_sweep,run_model_extreme_material,time_sim_vs_dt,accuracy_vs_stable_sim,batch_stable_sims}.py` | 08-22 | intra-package only | rebuttal R3 sim-vs-model study |
| `unified/eval_unified.py` | 08-30 | `data/unified_dataset.py`, HANDOFF, PREFLIGHT, 9 scripts, **grp** (unified_unet_xy_p99_40k) | unified 4-domain eval harness |
| `unified/overfit_prod.py` | 09-02 | rp×8, overnight README | production-batch overfit |
| `unified/lossprobe.py` | 08-29 | C, d/loss_viz_rules, t, 4 overfit scripts | deterministic loss probe |
| `unified/{overfit_smoke,overfit_routed,probe_unet_overfit,diag_pickup_pose}.py` | 08-28/29 | d/loss_viz_rules + siblings | overfit diagnostics |
| `unified/compute_norm_round2_stats.py` | 08-29 | cfg×2, `data/dataset.py`, `data/unified_dataset.py`, t | round-2 stats builder |
| `unified/compute_unified_iso_stats.py`, `materialize_cloth_wds.py`, `verify_unified.py` | 08-07/27 | `data/unified_dataset.py` / `train.py` | unified dataset build chain |
| `unified/eval_norm_arm.sh`, `eval_norm_worker.sh` | 08-27/30 | cfg×3/×2, rp×2/×6, d/unified_unet_design, PREFLIGHT | norm A/B eval protocol |
| `unified/norm_ab_table.py`, `normalization_study.py` | 08-27 | PREFLIGHT / `velocity_study.py` | norm A/B tables |
| `unified/{build_manifest,compute_pose_delta_stats,audit_tier2_recon,decompose_pickup_val,gate_v2,test_s3_pose_native_model,eval_specialist}.py` | 08-07..22 | README_UNIFIED_V1 / cd+d/tier2 / siblings / `eval_push_0807.py` | tier-2 design-phase tooling |

#### 2b. Unreferenced but recent (≥ 2026-07-01) — 58, keep-or-triage, not auto-archive

| Group | Scripts |
|---|---|
| MPM-UNet probe era (07-31..08-05, 14) | `ablate_context_sensitivity`, `eval_mpm_ar_rollout`, `analysis_{crossframe,gifs,plots,restart,streaming}`, `calibrate_f0off_init`, `floor_probe`, `grad_alloc_probe`, `make_eval_readme` (writes into mbronars' eval dir), `per_t_profile`, `procrustes_probe`, `shrinkage_probe` |
| Rebuttal / handoff (08-07, 08-22, 9) | `build_longhorizon_artifact`, `verify_longhorizon_consistency`, **`eval_push_0807.py` (grp: push_0807_{cond,uncond} READMEs)**, `overfit_viz_pickup_pose`, `sim_vs_model/{batch_accuracy,build_blowup_figure,render_stable_vs_ours,run_blowup_sweep,time_model_rollout}` |
| unified design/audit (08-07..08, 13) | `audit_consistency`, `build_{rollout,routing}_viewer`, `build_samples_artifact`, `compute_pickup_particle_stats`, `gate_drift_aux`, `offline_domain_val`, `probe_val_draw_noise`, `t6_interference_exp`, `test_s1_s2_pose_channels`, `test_s5_inference`, `bias_audit_v2` (linked from `progress.html`), `check_regression` |
| unified eval cells (08-22..27, 7) | `eval_cloth_specialist`, `eval_paper_cloth_table3`, `eval_v9_particles`, `overfit_viz`, `eval_paper_mpm_unified`, `probe_unet_forward`, `rescore_v2_norm_ab.sh` |
| unified UNet / norm round 2 (08-28..31, 12) | `bench_unet_domains`, `export_unet_playground`, `unet_mpm_regression` (all 3 linked from `unet_probe/*.html`), `probe_compile_nan`, `pushing_frame_rate_probe`, `velocity_study`, `compute_pd_asinh_stats`, `overfit_floor` (used by `norm_round2/plot_floor.py`), `verify_norm_round2`, `audit_val_vs_sample`, `draw_mpm_hi`, `select_{constrained,overfit}_batches` |
| data audit 09-03 (2) | `audit_cloth_episodes`, `audit_mpm_episodes` — invoked from `reports/unified/data_audit_0903/{flag_*,build_audit_page}.py`; they produced the LIVE blocklists |

#### 2c. Archive candidates — unreferenced AND last commit < 2026-07-01 (65)

| Group | Scripts (last commit) |
|---|---|
| Jan–Apr shell launchers, root (29) | `train.sh`, `train_grasp_debug.sh`, `train_mpm_sand.sh`, `train_stratified{,_windowed}.sh`, `inference.sh` (01); `train_uncond_debug.sh`, `train_multi_gpu.sh`, `train_rl.sh`, `train_mpm_multi.sh`, `train_vae_grasp{,_debug}.sh`, `train_latent_diffusion.sh`, `train_vae_rl.sh`, `train_dense_causal.sh` (02); `train_df{,_ddim}_overfit.sh`, `train_latent_diffusion_rl_df.sh`, `train_rl_df.sh`, `train_vae_rl_large.sh`, `train_rl_diffusion.sh`, `train_mpm_4obj_relative_df.sh`, `train_push_uncond_flow_relative_df.sh`, `train_push_cond_relative_df_v3.sh`, `train_rl_{cond,uncond}_relative_df.sh` (03); `test_push_04_04.sh`, `train_push_uncond_actemb_relative_df_x0.sh`, `train_sim1_cloth_2gpu.sh` (04). None has an existing run under `runs/` except `inference.sh` (points at `grasp_30k_fm_multi_gpu_bs_4x2_stratified`) |
| May pose-model debug (5) | `build_frames_summary.py`, `debug_pose_frames.py`, `debug_translation_invariance.py`, `debug_twist_units.py` (05-02), `teleop_sim1_rerun.py` (04-21) |
| June pose-guidance sweeps, root (8) | `n20_f12_new_train.sh`, `n20_novel_easy_perp.sh`, `sweep_doodl_{long,lr,old_vs_new}.sh`, `sweep_f12_new.sh`, `summarize_doodl.py`, `summarize_n20.py` — all drive `sweep_f9_clean.py`/`test_doodl_noise_opt.py` |
| `scripts/archive/` already-parked (12) | `parse_train_logs`, `run_overnight_f9_to_f14.sh`, `summarize_f9_to_f14`, `sweep_{joint_opt_lr,joint_phantom,k_sample_novel,lr_ablation,phantom_threshold,phantom_weight,pose_guidance,rod_leash,tier1_5_focused}` |
| normalization (6) | `compute_stats.py` (02-22, grasp_30k), `compute_stats_v2.py` (03-10), `compute_stats_sim1.py` (04-18), `compute_pose_stats.py` (06-20), `run_compute_latent_stats_rl{,_large}.sh` |
| inference_mpm (3) | `inference_mpm.sh`, `sweep_physics.sh`, `__init__.py` |
| other (2) | `refactor/compare_runs.py`, `analysis/run_analyze_vae_latent.sh` |

Ignored-on-disk scripts (not in git, deletable or archive-out-of-repo): `scripts/apr_06_*.sh` ×9, `scripts/analysis/{contact_statistics,visualize_contacts,visualize_repulsion,visualize_repulsion_on_samples}.py` (contact-head era), `scripts/refactor/results/`.

### 3. configs/ (153 tracked, 115 referenced / 38 not; by month: Mar 4, Apr 13, Jun 9, **Jul 72, Aug 54**, Sep 1)

83 configs are referenced **only** by their own `sbatch_scripts/` launcher (one-shot experiments).

| Family (count) | Dates | Referenced by | Status |
|---|---|---|---|
| `default`, `grasp_debug`, `grasp_rl`, `grasp_rl_diffusion`, `uncond_debug`, `mpm_sand` (6) | 04 | C, R, sb, `scripts/*.sh` | canonical but their launchers are all archive candidates |
| `vae_grasp`, `vae_rl`, `latent_diffusion_grasp`, `latent_diffusion_rl` (4) | 03 | C, R, sb, scripts | VAE/latent tier; stats files `unified_stats.npz`/`latent_stats*.npz` named in C **do not exist** |
| `push_uncond_*` (5) | 04 | `..._perframe_knn` by R+sb×5+7 sh; `push_uncond_{diffusion,flow,flow_long,flow_relative_df_rope_fix}` **unreferenced** | 4 archive cand. |
| `sim1_cloth`, `sim1_cloth_x0`, `sim1_cloth_fold_v9{,_med}`, `cloth_fold_v10` (5) | 04–08 | d, sb, scripts/unified | `sim1_cloth_fold_v9` unreferenced; `cloth_fold_v10` used by unified materializer |
| `sim1_variant_pose_rigid_v8` (1) | 06 | C, cd, d×2, R, rp×4, sb | **LIVE (v8 deployed)** |
| `robert_mpm_c8_v6_1`, `robert_mpm_drag_v6_1` (2) | 06 | C, cd, R, rp, sb | **LIVE (paper MPM)** |
| `pickup_pose_v9_4success_cfg{,_resume,_6d}` (3), `pickup_pose_uncond_place_0810`, `pickup_holdout_eval`, `pickup_longhorizon_eval` | 06–08 | C, cd, d, rp×9, sb, scripts, **grp** | v9 LIVE (deployed `config.yaml` = `_resume`); `pickup_holdout_eval` unreferenced |
| `pose_push_*` (5) | 06–08 | d, sb, rp, scripts | `_0807` pair deployed (grp) |
| `robert_mpm_object_v2*` (17) | 07 | 9 by sb/rp/scripts; **8 unreferenced** (`v2`, `v2_c2`, `dbg4tar_pointnet*`×3, `fast_lpo_pointnet`, `fast_lpo_pnxL`, `fast_lpo_regdd`) | July MPM-v2 exploration; `v2`/`v2_c2` have runs |
| `sph64_*` (31), `mul4_*` (9), `mul3_*` (4), `mul10_*` (3), `movisphere_*` (6) | 07-25..31 | each by exactly its own sb launcher | one-shot MPM UNet/sphere probes; no run under `runs/` for any |
| `unet_*` (19), `unetstd*` (7), `direct_noreg_*` (3), `bypass_ovf1w_ctx4`, `spikehunt_full_rope`, `sedge166_split108` | 08-01..05 | 6 by sb; **19 unreferenced** | overfit-1-window probes; `unetstd331_lightctx_sample_mpmfull` has the 11 G run |
| `unified_v1_*` (7), `unified_norm_*` (5), `unified_unet_*` (5) | 08-07..09-03 | rp, cfg, d, reports, t, **grp** | **LIVE** unified line; `unified_v1_tier2` unreferenced; `unified_unet_norm_xy_p99` = deployed 40K handoff |
| untracked `cloth_fold_v10_mw.yaml` | — | untracked `repro_sbatch/exp_cloth_v10_mw.sbatch`; smoke run `runs/_smoke_cloth_mw` (08-10) exists | motion-weighted cloth specialist, never launched for real |

Unreferenced list (38): `push_uncond_{diffusion,flow,flow_long,flow_relative_df_rope_fix}`, `robert_mpm_object_v2{,_c2,_dbg4tar_pointnet,_dbg4tar_pointnet_fpdec,_dbg4tar_pointnet_noaux,_fast_lpo_pointnet,_fast_lpo_pnxL,_fast_lpo_regdd}`, `unet_gva_world_ovf1`, `unet_local_world_ovf1`, `bypass_ovf1w_ctx4`, `direct_noreg{_aug,,_50k}_ovf1w_ctx4`×3, `sedge166_split108`, `spikehunt_full_rope`, `unet_dot_world_localall`, `unet_full_rope{,_edge,_pedge,_sedge166,_sedge}_ovf1w_ctx4`×5, `unet_gva_world_{localall,ovf1_localdata}`, `unet_local_pn_ovf1w_ctx4`, `unet_local_world_{localall,ovf1_localdata}`, `unet_sedge166_dfctx_ovf1w_ctx4`, `unet_serial_world_localall`, `unetstd_dfctx_ovf1w_ctx4`, `unetstd_dfctx_sample_mpmfull_physaux`, `pickup_holdout_eval`, `sim1_cloth_fold_v9`, `unified_v1_tier2`.

### 4. Launch files

No launcher references a missing config. Deployed-checkpoint `config.yaml` copies in the group dir depend on: `stats/pushing_actions_stats.npz` (4 ckpts), `pushing_actions_04_04/04_18_stats.npz`, `push_08_07_stats.npz`, `pickup_pose_stats_place_robust.npz`, `pickup_pose_stats_4dataset_success_robust.npz`, `push_raw256_stats_noclip.npz`.

**`repro_sbatch/` (61 tracked, 06-26 → 09-03) — matched to `runs/`:**

| Launcher(s) | Config | Run under `runs/` |
|---|---|---|
| `pickup_v9_repro`, `exp_{ctx4,depth12,noclip,objrotw5,posegeo,pickup_holdout}` | pickup_pose_v9_4success_cfg | yes: `pickup_pose_v9_4success_cfg_repro_timefix`, `pickup_v9_{ctx4,depth12,noclip,objrotw5,posegeo}`, `pickup_v10_assetholdout`; **`exp_dim768` → `pickup_v9_dim768` missing** |
| `pickup_v9_6d` | pickup_pose_v9_6d | yes (`…_6d_4success_cfg{,_timefix}`) |
| `sim1_v8_repro`, `exp_cloth_{anchorid,control,noclip}` | sim1_variant_pose_rigid_v8 | repro yes; **`sim1_v8_{anchorid,control,noclip}` missing** (never ran) |
| `mpm_obj_v6_1_repro`, `mpm_drag_v6_1_repro` | robert_mpm_{c8,drag}_v6_1 | obj yes (`…_c8_repro`); drag `_repro` missing (only original) |
| `exp_mpm_obj_v2_fast6k_{c2,c8}` | robert_mpm_object_v2_fast{_c2,} | yes (5.9/6.1 G) |
| `exp_unified_tier1`, `exp_unified_tier2{,v2_4gpu}`, `exp_unified_tier2v2_c4f16{,_2g,_kq}`, `exp_unified_tier2v2_clean` | unified_v1_* | tier1, tier2v2, c4f16, c4f16_2g yes; **`_kq`, `_clean` missing** |
| `exp_pose_push_{cond,uncond}_0807`, `exp_pickup_uncond_place_0810`, `exp_unetstd331_mpmfull_a100` | as named | yes (all 4, deployed) |
| `exp_unified_norm_{A,B,C}{,_katef}`, `exp_unified_norm_{asinh,xy_p99}_katef`, `eval_norm_ab_chain_{B,C}`, `eval_norm_worker` | unified_norm_* | B, C yes; **A, asinh, xy_p99 (transformer) missing** |
| `exp_unified_unet_c4f16_2g`, `probe_unified_unet_c4f16_preempt`, `exp_unified_unet_c4f16_combined_katef` | unified_unet_c4f16* | **missing** (superseded by norm arms) |
| `exp_unified_unet_norm_{asinh,perdomain,xy_p99}`, `eval_unet_norm_*`, `eval_unet_trained_windows_{a,b}`, `train_unet_xy_p99_resume200k` | unified_unet_norm_* | yes ×3 (9.9 G each) |
| `overfit_v2_{array,array2,smoke,tfcloth,joint_tf,joint_unet,joint_tf_xyg,joint_unet_xyg}` | `scripts/unified/overfit_prod.py` | outputs go to `reports/unified/overfit_v2/`, not `runs/` |
| `train_tf333m_xy_p99_{a100x4,pro6000x2}` (09-03) | unified_v1_tf333m_xy_p99_c4f16_200k | **not launched** (user-gated) |
| untracked `exp_cloth_v10_mw` | untracked cloth_fold_v10_mw | only `_smoke_cloth_mw` |

**`sbatch_scripts/` (123 tracked; Feb 13, Mar 8, Apr 8, May 2, Jun 7, **Jul 68**, Aug 17):** only **12/123** have a matching run: `train_drag_v6_1_2gpu`, `train_robert_mpm_c8_v6_1_2gpu`, `train_sim1_variant_pose_rigid_v8_2gpu_48h`, `train_pose_uncond_04_18_2gpu`, `train_pose_2gpu_flow_uncond_relative_df_rope_fullpoints_world`, `train_v2fast_base_200k`, `unetstd331_lightctx_sample_mpmfull{,_l40s}`, `_mem_probe_a100`, `_speed_a100_probe`, `train_cloth_fold_v10_2gpu`, `train_cloth_fold_v9_med_2gpu`. The other 111 (31 `train_sph64_*`, 11 `train_mul4_*`, 9 `train_v2*`, 7 `train_push_*`, 5 `train_mpm_*`, 4 `train_sphere_*`, 4 `noise_walk_*`, 4 `train_latent_*`, 4 `eval_*`, 3 `vae_overfit/`, 3 `overfit/`, …) are historical. Only 5 are mentioned anywhere else (cd, `scripts/refactor/REPORT.md`).

**`scripts/*.sh` (40):** 33 are the Jan–Apr/June launchers listed in §2c; the 7 referenced ones are `test_all_guidance.sh`, `test_novel_difficulty.sh`, `run_guidance_diag.sh`, `train_push_uncond_actemb_relative_df.sh` (only by its `_x0` sibling), `n20_best_noise.sh`, plus `sweep_doodl_old_vs_new.sh`/`sweep_f12_new.sh` which point at the existing `runs/pose_push_uncond_df_04_18`.

### 5. tests/ — 37 tracked, **242 tests collected** (`pytest --collect-only -q`, 18.7 s)

| Test file | Last | Covers | # |
|---|---|---|---|
| `test_batched_stratified.py` | 07-27 | `models.transformer`, `models.joint_transformer`, `data.dataset` | 36 |
| `test_pose_guidance_v2.py` | 06-20 | `guided_inference_pose`, `data.pose_batch` | 28 |
| `test_diffusion.py` / `test_diffusion_forcing.py` / `test_latent_diffusion_forcing.py` | 01-17 / 03-02 / 03-06 | `diffusion`, `diffusion.scheduling`, `models.latent_transformer`, `train` | 13/11/7 |
| `test_norm_round2_stats.py` | 08-28 | `data.dataset`, `data.unified_dataset` (+`configs/unified_norm_xy_p99.yaml`) | 12 |
| `test_lossprobe.py` | 08-29 | `scripts.unified.lossprobe` | 10 |
| `test_unet.py` / `test_unet_std_pose_native.py` / `test_unet_std_stratified.py` / `test_unet_std_unified_flags.py` / `test_unet_wrapper.py` | 08-01..28 | `models.unet`, `models.unet_std`, `utils.pose_native` | 9/6/6/3/4 |
| `test_pose_batch.py` / `test_pose_dataset.py` / `test_pose_ops.py` / `test_pose_transformer.py` | 06-20 | `data.pose_*`, `utils.pose_ops`, `models.pose_transformer` | 9/3/4/1 |
| `test_attention.py` / `test_embeddings.py` / `test_banded_rope.py` | 03-02 / 01-17 / 06-26 | `models.components.*` | 8/8/4 |
| `test_ar_rollout_latent.py` | 03-06 | `inference`, `data.test_batch` | 8 |
| `test_chunk_ar.py` | 07-31 | imports `tests.test_batched_stratified` | 7 |
| `test_fm_guidance.py` / `test_get_x0_est_bug.py` | 04-19 / 03-23 | `diffusion.flow_matching` / `diffusion.ddim` | 6/4 |
| `test_episode_blocklists.py` | 09-03 | `data.unified_dataset` (LIVE blocklists) | 4 |
| `test_train.py`, `test_transformer.py`, `test_utils.py`, `test_visualization.py`, `test_batch.py` | 01–06 | `train`-side, `models.transformer`, `utils.ema`, `utils.visualization`, `data.batch` | 4/4/3/2/2 |
| `test_diffusion_overfit.py`, `test_flow_matching_overfit.py` | 01 | overfit convergence | 4/4 |
| `test_30k_data.py` | 01-17 | glob over grasp_30k data (data-dependent) | 1 |
| `test_normalization.py`, `vis_random_samples.py`, `vis_rl_data.py`, `visualize_rope.py` | 02–04 | **0 tests** — standalone scripts misfiled under tests/ | 0 |

No pickup-pose (`PickupPoseTransformer`) or `data.unified_dataset` sampler test beyond blocklists/stats. 11 MB of ignored `tests/*.png|gif` outputs on disk.

### 6. Previous cleanup (archive/ + cleanup_docs/) — do not redo

| Item | What it recorded / moved |
|---|---|
| `archive/README.md` (06-20) | `archive/configs/` (21: MPM c8 v5/v6/v7/pfknn/v6_1 C1/C2/ablations, SIM1 v1–v7 lineage), `archive/sbatch/` (24 matching launchers), `archive/pickup_rigid_legacy/` (32: `rigid_transformer*`, `train_rigid*`, `inference_rigid*`, `panda_gripper_kinematics.py`, pickup v2–v8 configs, 4 `PICKUP_*_SUMMARY.md`). Pure scratch was moved **out of repo** to `../particle_world_model_archive/` (`MOVE_MANIFEST.txt`); pickup v3→v9 tree frozen on `robert/pickup_v9_snapshot` |
| `cleanup_docs/CLEANLINESS_REPORT.md` (06-21, 240 l) | 101 findings (11 high). Verified now: `POSE_DIM=7` removed; `inference_pickup_pose.py` passes `rotation_rep`; `noise_opt_guidance.py` uses `getattr(..., "dual_time")`; `compute_stats_sim1_variant_stride.py` deleted; duplicate `interactive_pushing.py` gone. **Still present:** `PickupPoseStats.to()` (`data/pickup_pose_batch.py:58`, now shape-correct but unused) |
| `MPM_TRAINING_MERGE_REPORT.md`, `SIM1_V8_MERGE_REPORT.md` (06-20) | merge handoffs for MPM v6.1 object/drag and SIM1 v8 (what was committed vs. needed merging) |
| `PICKUP_V9_MERGE_AND_REPRO.md` (06-20, 396 l) | v9 merge topology + repro recipe; deployable bundle at group `v9_4success_cfg/` |
| `DECODE_INVESTIGATION.md` (06-24) | time-conditioning bug (fixed) + decoder identity-loss A/B (parked, branch `decoder-identity-ab`) |

`scripts/archive/` (21 pose-guidance sweep scripts) was the June cleanup's script parking lot; CLAUDE.md still cites 8 of them at their pre-move paths `scripts/sweep_*.py`.

### 7. docs/

`docs/` is **gitignored as a directory**; 4 files are force-tracked, 22 `.md` + ~475 MB of assets are local-only.

| Doc | Tracked | Last | Status |
|---|---|---|---|
| `loss_viz_rules.md` (08-29), `unified_unet_design.md` (08-27), `tier2_v2_design.md`, `unified_dataset_spec.md` (08-07) | T | — | **current** (unified line) |
| `pickup_training_handoff.md` (07-27), `public_release_plan.md` (07-18), `sim1_investigation_status.md` (07-10) | ign | — | current-ish handoffs; the first two link into `../particle_world_model_archive/` scripts |
| `per_particle_component_audit_plan.md`, `rigid_model_component_audit_plan.md`, `rigid_model_reasoning_research_plan.md` (06-27..29) | ign | — | June audit plans (done; see memory) |
| `physctrl_baseline.md`, `physctrl_add_dataset.md`, `DATASET_SPEC_FOR_POINTWORLD.md`, `datasets_corl2026.md` (05) | ign | — | CoRL baselines/dataset spec; reference `configs/config_dit_base_*.yaml`, `scripts/inference_mpm/{eval_paper,plot_paper_curves}.py`, `scripts/collect_pickup_3dpcwm.py` — **none exist here** (other repos) |
| `pose_transformer_architecture.md` (04-27), `improvement_ideas.md` (email draft, 04-19), `knn_noise_investigation.md`, `guidance_improvements.md`, `df_inference_analysis.md`, `positional_encoding_analysis.md`, `contact_modeling_analysis.md` (lit review), `architecture.md`, `architecture_diagrams.md`, `data.md`, `coding_guideline.md` (04) | ign | — | historical (Feb–Apr particle/guidance era). None mentions the removed contact head |
| assets: `moreR/` 163 M (3 alphabet RL sweep dirs), `flame_viz/` 100 M + `flame_viz.zip` 96 M (duplicate), `3DPointCloudWorldModelsCoRL2026{,_new}/` 58+54 M (LaTeX builds incl. `0_main.pdf`), `flame_supplementary/` 8.6 M, `FLAME_supplementary.pdf`, `Datasets ….pdf`, `diagrams/` (2 md), `pr_reviews/pr_6.md`, `guided_inference_math.{tex,log}` | ign | — | ~475 MB of non-source |

### 8. reports/ (1.4 G; 409 tracked — 288 json, 49 log, 37 py, 15 html, 10 png, 7 md)

| Subdir | Size | Tracked / untracked | What |
|---|---|---|---|
| `phantom_blocklists/` | 131 K | 2 / 0 | **LIVE**: v9 + place blocklists (configs + `data/unified_dataset.py`) |
| `unified/data_audit_0903/` | 71 M | 18 / 18 | **LIVE**: `mpm_blocklist_train.json`, `cloth_blocklist_{train,val}.json` loaded by `data/unified_dataset.py:618-628`; builders `flag_*.py`, `build_audit_page.py`, `audit_artifact.html`; 39 gif + 29 png untracked |
| `unified/overfit_v2/` | **401 M** | 15 / 4 | 4-number joint overfit: 504 npz + 16 `.pt` + builders `build_joint{,_norm}.py` (HANDOFF cites `build_joint_norm.py`) |
| `unified/overnight_0830/` | 245 M | 10 / 5 | overnight UNet audit: 249 npz, 73 png, 57 log; `plot_overnight.py` modified |
| `unified/norm_round2/` | 66 M | 36 / 42 | round-2 norm study; `PREFLIGHT.md`; `overfit_prod/` 20 M untracked |
| `unified/unet_probe/` | 74 M | 44 / 7 | UNet atlas/playground (`unified_unet_atlas.html` cited by 4 live configs); 10 `.pt` |
| `unified/norm_ab/`, `norm_ab_trained_windows/` | 15 M, 3.6 M | 161+52 / 224+49 | norm A/B per-cell JSONs (tracked) + 453 untracked logs |
| `unified/site_dumps_130k/` | 81 M | 0 / dir | npz dumps for the website |
| `unified/gifs/`, `viz_*` ×22, `renders/`, `ckpt*_push_*`, `specialist_scratch` | 37 M + ~55 M | 0 | GIF/PNG viz (ignored by extension) |
| `unified/*.{json,log,html,md}` root | — | ~70 / ~120 | eval JSONs (paper MPM table1, unet200k stream, spec_v2c2…), `HANDOFF_unified_unet_xy_p99_40k.md`, `README_UNIFIED_{V1,UNET}.md`, `OVERNIGHT_STATUS.md`, `progress.html` (cited by `unified_v1_tier1.yaml`), 120 untracked eval logs |
| `cloth_v8_rollouts/` | 287 M | 0 / dir | 6 npy + 3 json regenerated v8 rollouts (website source) |
| `longhorizon/` | 8.1 M | 0 / dir | long-horizon pickup consistency JSON/npz |
| `place_export/` | 13 M | 0 (ignored) | 2 npz place demos |
| `analysis_unetstd_185k/` | 34 K | 1 | index.html |

**Code/config → reports dependencies (must survive):** `reports/phantom_blocklists/robert_pickup_data_4success_combined.json` (3 configs + `data/unified_dataset.py:609`), `…/robert_pickup_data_place_success_strict_v2.json` (`pickup_pose_uncond_place_0810.yaml`), `reports/unified/data_audit_0903/{mpm_blocklist_train,cloth_blocklist_train,cloth_blocklist_val}.json` (`data/unified_dataset.py`), comment-only cites: `reports/unified/unet_probe/unified_unet_atlas.html` (4 configs), `reports/unified/progress.html` + `MANIFEST.json` (`unified_v1_tier1.yaml`; MANIFEST is untracked/ignored), `reports/unified/audit_report.json` (`data/unified_dataset.py:4` docstring, ignored on disk), `reports/aux_loss_calibration_{unetstd_deploy,unetstd_final,f0off_init}.json` (6 configs + `train.py:167` + `utils/phys_losses.py:4` — **files do not exist**, comment-only).

### 9. Untracked files (`git status --porcelain | grep '^??'`)

| Location | Count | Size | Class |
|---|---|---|---|
| `reports/unified/**/*.log` | 453 | 15 M | logs — safe to delete (eval/probe stdout; the JSONs they produced are tracked) |
| `reports/unified/**` html 17, jsonl 5, jpg 3, pdf 1, md 1 (`velocity_study.md`), py 4 | 31 | small | regenerable pages / study notes; `velocity_study.md` + the 4 `.py` are source-like |
| `reports/unified/norm_round2/overfit_prod/`, `site_dumps_130k/` | 2 dirs | 20 M, 81 M | generated npz |
| `reports/cloth_v8_rollouts/`, `reports/longhorizon/` | 2 dirs | 287 M, 8 M | generated npy/npz/json (website + longhorizon eval inputs) |
| `configs/cloth_fold_v10_mw.yaml`, `repro_sbatch/exp_cloth_v10_mw.sbatch` | 2 | 33 K | **source, never committed** — motion-weighted cloth specialist (only smoke-run) |
| `scripts/unified/overfit_denoise.py` | 1 | 33 K | **source** — sampled-error overfit probe (imports `lossprobe`, `overfit_smoke`, `probe_unet_overfit`) |
| `.github/workflows/pages.yml` | 1 | — | **source** — GitHub Pages deploy of `website/` on push to `main` |
| `website/` | dir | 99 M | **source + data** — site (`index.html`, css/js/data 29 M, `tools/{pack_data,export_place}.py`) + 62 M `_shots*` review screenshots + logs |
| `rebuttal/` | dir | 27 M | LaTeX source (`rebuttal_template.tex`, `references.bib`, 2 fig PDFs) + 27 MB paper PDF + zip + logo |
| `slurm/` | 122 | 5.5 M | SLURM logs 08-28..09-04 — deletable once the referenced jobs are recorded |
| `runs_smoke/` | 8 | 335 K | smoke logs/json from the UNet merge gates |

Modified tracked: `git diff --stat` = 4 files, +207/−2: three `norm_round2/*.log` (appended monitor/probe output) and `overnight_0830/plot_overnight.py` (+4/−2: excludes `_mpm_hi` runs from the mpm grid and rewrites the caption).

### 10. Stale constants / duplicated functionality (observed, not fixed)

| Item | Where | Note |
|---|---|---|
| `DOMAIN_DT = {"mpm": 1/24, "pickup": 0.05, "cloth": 1/30, "pushing": None}` | `data/unified_dataset.py:54` | pushing is ≥100 fps (likely ~450-500), not 24; `pushing: None` is stale |
| `DOMAIN_FUTURE_LEN = {mpm 16, pickup 38, cloth 16, pushing 16}` | `data/unified_dataset.py:692` | design constant baked in code (flagged in CLAUDE.md pre-flight rule) |
| Stats sets `STATS_SETS`/`POSE_STATS_SETS` | `data/unified_dataset.py:657-666` | 43 `stats/unified_*` npz look "unreferenced" by basename but are resolved by template; keep |
| CLAUDE.md dead paths | `CLAUDE.md` | cites `scripts/{compare_all_phases,plot_novel_results,sweep_*}.py` (now `scripts/archive/`), `stats/{unified_stats,latent_stats,latent_stats_rl}.npz` (absent) |
| Group `v9_4success_cfg/README.md` | group dir | cites `scripts/run_uncond_inference.py`, `run_uncond_per_dataset.py` — exist only in the pickup clone (`robert/pickup_v9_snapshot`) |
| 6 eval entrypoints for MPM | `scripts/eval_object_rollout{,_unet}.py`, `eval_mpm_ar_rollout.py`, `eval_firstimpact_error_prod.py`, `inference_mpm/inference_mpm_multi.py`, `unified/eval_paper_mpm_unified.py` | overlapping AR-rollout evals; `inference_mpm_multi.py` is the paper-protocol one |
| 3 cloth eval paths | `unified/eval_cloth_specialist.py`, `eval_paper_cloth_table3.py`, `block_rollout_sim1_variant.py` | |
| 16 stats scripts | `scripts/normalization/` (10) + `scripts/unified/compute_*` (5) + `normalization_study.py` | generational duplicates (`compute_stats` → `_v2` → per-format → isodist → round2) |
| 7 overfit harnesses | `unified/overfit_{smoke,routed,floor,prod,denoise,viz}.py`, `probe_unet_overfit.py`, root `overfit_viz_{pose,pickup_pose}.py` | successive generations |
| Noise-opt duplication | `noise_opt_guidance{,_pose}.py` vs `guided_inference{,_pose}.py --mode noise` | root DOODL scripts have 1 and 0 referrers |
| Frozen model copy | `scripts/unet_std_launch_snapshot.py` (= `models/unet_std.py` at launch) + ignored `models/_unet_std_ref.py` | 11 scripts import the snapshot |
| `models/components/pool_draft.py`, `utils/pose_viz.py` | 0 importers | dead modules |
| `data.test_dataset` needs `stats/gripper_template.pt` (untracked) | `data/test_dataset.py` | |
| `tests/{test_normalization,vis_random_samples,vis_rl_data,visualize_rope}.py` | collect 0 tests | scripts misfiled in `tests/` |
| `.gitignore` vs reality | `docs/` and `scripts/analysis/` ignored but partially tracked; `CLAUDE.md` ignored; `*.json/*.npz/*.log` ignored yet 297 tracked under `reports/` via `-f` | new files under those patterns silently stay untracked |
| Dangling tracked symlink | `data_subsets` | |

### 11. Git state

| Item | Value |
|---|---|
| Branch | `robert/unified-unet-combined` @ `7091f3a` (09-04); in sync with `origin/…` (0 ahead / 0 behind) |
| vs `main` (= `origin/main`, `6a1fa6a`, 07-24) | **144 commits ahead, 0 behind**; merge-base `6a1fa6a` |
| Local branches | 45 (`all_relative_df`, `bronars*`, `decoder-identity-ab`, `refactor/transformer-cleanup`, 39× `robert/*`) |
| Remote branches | 53 incl. mbronars' `unet`, `unet_clean`, `unet_dt`, `flow_map*`, `speed-optimizations`, `particle_overfit` |
| Worktrees | several sibling worktrees share this `.git` (`git worktree list`), incl. `../particle_world_model_pickup` (`robert/pickup_v9_snapshot`), `../pwm_decoder_ab` (`decoder-identity-ab`), `../pwm_rope_decode_wt` (`robert/particle-rope-decode`), `../pwm_sidepose` (`robert/unified-unet-sidepose`), plus private research worktrees that must not be pushed. Branches with a worktree cannot be deleted while it exists |
| Running jobs | a dev allocation is active and writing under `runs/` today; check `squeue -u robertre` before touching `runs/` |

### 12. Proposed cleanup map

#### KEEP (live — with what depends on it)
- Core: `train.py`, `train_pose.py`, `train_pickup_pose.py`, `inference*.py` (all 5), `guided_inference{,_pose}.py`, `models/ data/ utils/ diffusion/` (except the two dead modules below), `tests/` (242 collected), `pyproject.toml`.
- Configs: `sim1_variant_pose_rigid_v8`, `robert_mpm_{c8,drag}_v6_1`, `pickup_pose_v9_*` ×3, `pickup_pose_uncond_place_0810`, `pose_push_{cond,uncond}_df_0807`, `cloth_fold_v10`, `unetstd331_lightctx_sample_mpmfull`, all `unified_*` (17), plus the CLAUDE-documented tier configs (`default`, `grasp_*`, `vae_*`, `latent_diffusion_*`, `mpm_sand`, `push_uncond_flow_relative_df_perframe_knn`, `sim1_cloth*`) — deployed group `config.yaml`s and READMEs point at these.
- `stats/`: all 57 tracked + the **untracked-but-required** `pushing_actions_stats.npz` (23 configs, 4 deployed ckpts), `pushing_actions_04_{04,18}_stats.npz`, `uncond_pushing_stats.npz` (9), `robert_mpm_v2_object{,_6key}_p9999_stats.npz` (21), `cloth_curated_med_stride2_stats.npz` (`cloth_fold_v10`), `cloth_fold_{med,v1}_stride2_stats.npz`, `unified_stats_rl.npz`, `mpm_sand_stats.npz`, `gripper_template.pt` (`data/test_dataset.py`) → `git add -f`.
- `reports/phantom_blocklists/` (2), `reports/unified/data_audit_0903/*.json` + builders, `reports/unified/unet_probe/unified_unet_atlas.html`, `progress.html`, `HANDOFF_*.md`, `README_UNIFIED_*.md`, `norm_round2/PREFLIGHT.md`, tracked eval JSONs (paper/rebuttal provenance).
- Scripts with live consumers: `eval_pickup_pose.py` (C/R/train_pickup_pose), `build_4success_combined_blocklist.py`, `normalization/compute_stats_{mpm,pickup_pose}.py`, `inference_mpm/inference_mpm_multi.py` (+`sweep_physics.py`), `long_rollout{,_pose,_sim1,_sim1_variant}.py`, `block_rollout_sim1_variant.py`, `streaming_*_sim1_variant.py`, `eval_push_0807.py` + `long_rollout_pose.py` + `sweep_f9_clean.py` + `test_doodl_noise_opt.py` + `test_multiwindow_04_18.py` (**group READMEs**), `eval_object_rollout{,_unet}.py` + `unet_std_launch_snapshot.py`, `calibrate_aux_losses.py`, the whole `scripts/unified/` line (`eval_unified.py`, `overfit_prod.py`, `lossprobe.py`, `compute_norm_round2_stats.py`, `compute_unified_iso_stats.py`, `materialize_cloth_wds.py`, `verify_unified.py`, `eval_norm_{arm,worker}.sh`, `audit_{cloth,mpm}_episodes.py`, …), `sim_vs_model/` (rebuttal R3), guidance tooling cited by CLAUDE (`test_all_guidance.sh`, `test_novel_difficulty.sh`, `build_sweep_report.py`, `check_novel_difficulty.py`, `debug_vae_recon.py`).
- Launchers: `repro_sbatch/` (all 61 are reproduction receipts; 12 targets still unlaunched) + the 12 `sbatch_scripts/` with existing runs.
- `archive/`, `cleanup_docs/`, tracked `docs/*.md` ×4.
- Untracked source to commit: `configs/cloth_fold_v10_mw.yaml` + `repro_sbatch/exp_cloth_v10_mw.sbatch` (or drop both together), `scripts/unified/overfit_denoise.py`, `.github/workflows/pages.yml`, `website/` (minus `_shots*`, `*.log`), `rebuttal/*.tex|bib|fig*.pdf`, `reports/unified/velocity_study.md` + 4 untracked `.py` builders.

#### ARCHIVE (move under `archive/`, reason)
- `archive/scripts_2026H1/`: the 65 §2c archive candidates (29 Jan–Apr root `.sh`, 5 May pose-debug, 8 June DOODL/n20 sweeps, 6 old normalization, 3 inference_mpm sh/init, `refactor/compare_runs.py`, `analysis/run_analyze_vae_latent.sh`) — unreferenced, pre-07-01, no matching runs; fold the existing `scripts/archive/` (21) into the same place and fix the 8 stale CLAUDE.md paths.
- `archive/scripts_guidance_2026-06/`: the "cd-only" cluster (`compare_guidance`, `debug_action_mask`, `unguided_baseline`, `visualize_{knn_guidance,pose_delta_distribution,x0_estimate}`, `check_rigid_pose_head`, `test_gripper_kinematics_roundtrip`, `test_v8_local`, `test_teleop_sequence`, `teleop_sim1{,_rerun}`, `test_multiwindow_noise`, `run_guidance_diag.sh`+`summarize_guidance_diag.py`, `n20_best_noise.sh`, `classify_phantom_modes.py`, `save_utils.py`) — only cited by the June cleanliness report; keep the 4 group-README scripts at their current paths.
- `archive/mpm_v2_unet_probes_2026-07/`: 101 configs (`sph64_*`, `mul{3,4,10}_*`, `movisphere_*`, `unet_*`, `unetstd_dfctx*`, `direct_noreg_*`, `bypass_*`, `spikehunt_*`, `sedge166_*`, `robert_mpm_object_v2_*` except `_fast`/`_fast_c2`) + their 111 run-less `sbatch_scripts/` + the 14 §2b probe scripts (`analysis_*`, `*_probe.py`, `ablate_context_sensitivity`, `eval_mpm_ar_rollout`, `calibrate_f0off_init`, `make_eval_readme`, `noise_walk_*`, `eval_firstimpact_error_prod`, `per_axis_error`) — one-shot July exploration, referenced only by each other.
- `archive/configs/`: `push_uncond_{diffusion,flow,flow_long,flow_relative_df_rope_fix}`, `pose_push_{uncond_04_18,*_fullpoints}` (runs exist but superseded by `_0807`), `sim1_cloth_fold_v9{,_med}`, `pickup_holdout_eval`, `unified_v1_tier1`/`tier2`/`tier2v2_clean`, `unified_norm_{A,B,C}` (A/B/C study closed 08-26), `unified_unet_c4f16{,_combined}` (superseded by `_norm_*`).
- `archive/unified_design_2026-08/`: §2b tier-2 design tooling (`audit_consistency`, `build_{rollout,routing}_viewer`, `build_samples_artifact`, `gate_drift_aux`, `gate_v2`, `offline_domain_val`, `probe_val_draw_noise`, `t6_interference_exp`, `test_s{1_s2,3,5}_*`, `bias_audit_v2`, `check_regression`, `decompose_pickup_val`, `audit_tier2_recon`, `compute_pickup_particle_stats`, `compute_pose_delta_stats`, `rescore_v2_norm_ab.sh`, `norm_ab_table`, `normalization_study`, `velocity_study`, `compute_pd_asinh_stats`, `verify_norm_round2`, `probe_compile_nan`, `pushing_frame_rate_probe`) — studies concluded per memory, results already in tracked reports.
- Ignored docs: `docs/*.md` ×22 historical → commit into `archive/docs/` (or leave ignored; currently invisible to git).
- `reports/unified/{norm_ab,norm_ab_trained_windows,norm_round2,overnight_0830,unet_probe}` tracked parts: keep in place (provenance) but stop adding logs.

#### DELETE (logs, generated, duplicates)
- 453 untracked `reports/unified/**/*.log` (15 M) + `slurm/` (122 logs) + `runs_smoke/` + `website/_shots*` (62 M) + `website/*.log` + `rebuttal/*.aux|log|out` + `docs/guided_inference_math.log`.
- Generated/regenerable: `reports/unified/site_dumps_130k/` (81 M), `norm_round2/overfit_prod/` (20 M), `reports/unified/gifs/` + 22 `viz_*` dirs (~90 M), `reports/unified/overfit_v2/*.pt` + 504 npz (400 M, rebuildable from `overfit_prod.py`), `overnight_0830` npz/png (245 M), `docs/flame_viz.zip` (duplicate of `docs/flame_viz/`), `docs/3DPointCloudWorldModelsCoRL2026` (old vs `_new`, 58 M), `paper_figures/` (91 M, 05-23), `debug_vis/` (32 M), `vis_{diffusion,flow_matching}/`, `tests/*.png|gif` (11 M), `guidance/` (6 stale `.pyc`), `sweep_0404/`, `scripts/apr_06_*.sh`, `scripts/analysis/{contact_*,visualize_repulsion*}.py`, `scripts/refactor/results/`, all `__pycache__`/`.pytest_cache`/`egg-info`, `.codex` (0 B), `stats/*.png` (9), `stats/robert_mpm_drag_setA_C2_p9999_stats_OLD.npz`.
- Dead code: `models/components/pool_draft.py`, `utils/pose_viz.py` (0 importers), `noise_opt_guidance_pose.py` (0 refs), `data_subsets` dangling symlink, `PickupPoseStats.to()`.
- Big ignored dirs to prune with the user: `inference_results/` 31 G (84 dirs, mostly Feb–Jun), `wandb/` 2.7 G, `checkpoints/` 3.7 G (May pose ckpts), `runs/` 313 G — candidates: `_smoke_*`/`_overfit_*` (≈6 G), `fit_test`, `v6_smoke_100M`, `smoketest_c8`, `pose_ropesweep_*` ×4 + `pose_fewshot_*` ×2 + `pose_overfit_*` ×3 (≈11 G), `unified_smoke_*`/`unified_t2_smoke*` (13 G), `drag_ovfit_*` ×4 (2.2 G).

#### RISKS
1. **A running dev job writes into `runs/` today** — do not prune `runs/` while it is alive (`squeue -u robertre`).
2. **Deployed checkpoints depend on repo files**: group READMEs cite `scripts/eval_push_0807.py`, `scripts/long_rollout_pose.py`, `scripts/sweep_f9_clean.py`, `scripts/test_doodl_noise_opt.py`, `scripts/test_multiwindow_04_18.py`, `scripts/unified/eval_unified.py`, `configs/unified_unet_norm_xy_p99.yaml`, `configs/pickup_pose_uncond_place_0810.yaml`; their `config.yaml`s resolve `./stats/pushing_actions*.npz`, `push_08_07_stats.npz`, `pickup_pose_stats_*.npz` relative to a repo checkout.
3. **Live data code reads `reports/`**: `data/unified_dataset.py` loads `reports/phantom_blocklists/…4success_combined.json` and `reports/unified/data_audit_0903/*_blocklist_*.json`; `tests/test_episode_blocklists.py` asserts them. Moving `reports/` breaks training.
4. **Config-required stats are untracked** (`pushing_actions_stats.npz`, `robert_mpm_v2_object_6key_p9999_stats.npz`, `cloth_curated_med_stride2_stats.npz`, …) — a fresh clone cannot train `cloth_fold_v10` or any pose-push config; a careless `git clean -x` deletes them.
5. `CLAUDE.md` is gitignored and local-only; `docs/` and `scripts/analysis/` are ignored directories with force-added files — anything new there is silently untracked.
6. `.github/workflows/pages.yml` deploys `website/` on push to `main`; committing it + `website/` changes the public site.
7. `scripts/unet_std_launch_snapshot.py` is imported by 11 scripts as the frozen unetstd model — archiving it breaks the whole Aug analysis chain.
8. `sbatch_scripts/*.sbatch` and `repro_sbatch/*.sbatch` hardcode `conda activate fastrin_test` (L40S-only pytorch3d) and absolute repo paths; moving files changes `cd`/`PYTHONPATH` lines inside them.
9. The worktrees share `.git`; deleting local branches that have a worktree is blocked/unsafe while the worktree exists.

---

# 7. Consolidated action list and open decisions

This chapter merges the per-chapter "cleanup candidates" into one ordered list. Nothing here has been done; it is the proposal.

### A. Fix before the next unified launch (correctness)

| # | Action | Why | Where |
|---|---|---|---|
| A1 | Make the pickup blocklists match the group-path shard URLs (normalize prefixes in `_load_blocklists`, or ship both forms as the mpm/cloth lists do) and add a shipped-list match test | none of the 8,506 pickup entries has matched since 2026-08-26 | `data/unified_dataset.py:91-151`, `tests/test_episode_blocklists.py` |
| A2 | Decide the blocklist asymmetry for the 200K pair: the UNet xy_p99 40K baseline trained unfiltered (mpm/cloth lists did not exist, pickup lists did not match); a fresh transformer run now trains filtered | otherwise the capacity-matched pair is not corpus-matched | `repro_sbatch/train_tf333m_xy_p99_*.sbatch`, `train_unet_xy_p99_resume200k.sbatch` |
| A3 | Rename the three smoke-overwritten `final.pt` files (`robert_mpm_object_v2_c2`, `robert_mpm_joint_v6_1_c8`, `cloth_fold_v10`) to `final_SMOKE_step*.pt` | `best.pt` is the real model in each; a future eval that defaults to `final.pt` would score a step-3 model | `runs/` |
| A4 | `git add -f` the stats files that live configs and deployed `config.yaml`s require | fresh clone cannot train `cloth_fold_v10` or any pose-push config; `git clean -x` hazard | `stats/` (list in chapter 6 §12 KEEP) |
| A5 | Fix `DOMAIN_DT["pushing"]` (≈1/450–1/500 once the writer subsample is confirmed with the collector's owner) or delete `DOMAIN_DT` (nothing consumes it) | stale constant that misled the first normalization study | `data/unified_dataset.py:54` |
| A6 | Either consume or delete `train.spike_grace_steps` in `train.py`; either wire `domain_weights`/`seed` through `train.py` or remove the dead plumbing in `dataset.py:1809-1811` | dead config keys look like decisions | `train.py`, `data/dataset.py` |
| A7 | Correct the handoff README sentence about sampler selection (done in this commit) and re-verify the shared copy | colleague-facing | `reports/unified/HANDOFF_unified_unet_xy_p99_40k.md`, group dir README |

### B. Repository hygiene (no behavior change)

| # | Action | Where |
|---|---|---|
| B1 | Track `CLAUDE.md`; un-ignore `docs/` (or move the 4 tracked docs + this one out of an ignored dir); decide what to do with the 22 local-only docs and ~475 MB of paper assets under `docs/` | `.gitignore`, `docs/` |
| B2 | Commit the never-committed source: `configs/cloth_fold_v10_mw.yaml` + its sbatch (or drop both), `scripts/unified/overfit_denoise.py`, `.github/workflows/pages.yml`, `website/` (minus screenshots and logs), `rebuttal/` sources, `reports/unified/velocity_study.md` + 4 untracked builders | untracked list, chapter 6 §9 |
| B3 | Delete logs and regenerable artifacts: 453 untracked `reports/unified/**/*.log`, `slurm/`, `runs_smoke/`, `website/_shots*`, `site_dumps_130k/`, `norm_round2/overfit_prod/`, `gifs/` + `viz_*`, `overfit_v2/*.pt` + npz (rebuildable), `overnight_0830` npz/png, `docs/flame_viz.zip`, `paper_figures/`, `debug_vis/`, `tests/*.png|gif`, stale `.pyc`, `stats/*.png`, `stats/*_OLD.npz` | chapter 6 §12 DELETE |
| B4 | Remove dead code: `models/components/pool_draft.py`, `utils/pose_viz.py`, `noise_opt_guidance_pose.py`, the dangling `data_subsets` symlink, `PickupPoseStats.to()`, `NormalizationStats.normalize_actions/denormalize_actions` (test-only), `data/rotation_aug.py::xy_isotropic_stats` | chapter 3 §10, chapter 2 cleanup |
| B5 | Fix doc drift: README flag table (`context_adaln`, defaults), CLAUDE.md archived sweep paths and the sample-mode loss rule and the missing stats paths, `docs/tier2_v2_design.md:26` (BODY_POSE scales still exist), the 21 references to the non-existent `docs/scope_unified_unet.md` and `docs/unified_unet_deferred.md`, `docs/data.md` / `datasets_corl2026.md` / `unified_dataset_spec.md` counts and dt | chapters 1, 3 |
| B6 | Rewrite or archive launchers: the 82 `sbatch_scripts/` pointing at a colleague's home; replace hardcoded `cd /data/user_data/robertre/...` with `$SLURM_SUBMIT_DIR`; make the env line explicit per partition | `sbatch_scripts/`, `repro_sbatch/` |
| B7 | Move the four zero-test files out of `tests/` (`test_normalization.py`, `vis_random_samples.py`, `vis_rl_data.py`, `visualize_rope.py`) | `tests/` |

### C. Archive (move under `archive/`, keep history)

| # | What | Count | Reason |
|---|---|---|---|
| C1 | VAE / latent tier: `models/vae.py`, `models/latent_transformer.py`, `train_vae.py`, `train_latent_diffusion.py`, `inference_vae.py`, `inference_latent_diffusion.py`, `data/latent_stats.py`, `VAETransformerBlock`, 2 tests, 12 launchers, 4 configs | ~25 files | no inbound dependency; last content commit 2026-03-08 |
| C2 | Research UNet: `models/unet.py`, `models/_unet_std_ref.py` (ignored), 21 `unet_*`/`sedge166_*`/`spikehunt_*` configs | 23 | superseded by `unet_std`; keep `_unet_std_ref.py` only if the bit-identity gate stays |
| C3 | Guidance stack: `guided_inference.py`, `guided_inference_pose.py`, `noise_opt_guidance.py`, `interactive_pushing.py`, `scripts/*guidance*`, `scripts/archive/sweep_*`, the "cd-only" June cluster | ~40 | pushing-era; unaware of pose-native/UNet; four scripts cited by group READMEs stay in place |
| C4 | Joint-transformer sweep configs (`sph64_*`, `mul*`, `movisphere_*`, `robert_mpm_object_v2_*` except `_fast{,_c2}`, `_c2`, `v2`) + their 111 run-less `sbatch_scripts/` + 14 probe scripts | ~230 | one-shot July exploration, no checkpoints |
| C5 | 65 scripts unreferenced and untouched since before 2026-07-01 (chapter 6 §2c) | 65 | superseded generations of launchers, stats scripts, pose debug |
| C6 | Superseded configs: `unified_v1_tier2`, `unified_norm_A`, `unified_norm_{xy_p99,asinh}` (transformer arms never trained), `unified_unet_c4f16{,_combined}`, `unified_v1_tier2v2_clean`, `push_uncond_{diffusion,flow,flow_long,flow_relative_df_rope_fix}`, `pickup_holdout_eval`, `sim1_cloth_fold_v9` | 13 | studies closed or never run |
| C7 | Superseded stats sets and their generators: `iso_p9999`, `global_iso_p9999`, `unified_asinh`, `pd_asinh`, `xy_p99_global` (+ `compute_unified_iso_stats.py`, `compute_pd_asinh_stats.py`, the asinh branch of `load_all`/`_pose_stats` if `AsinhNormalizationStats` goes); unreferenced npz (`mpm_multi_stats`, `mpm_multi_object_stats`, `robert_mpm_train_1to4obj{,_v2}_stats`, `pushing_actions_filtered_stats`, `pushing_pose_se3_stats_500b`, `pushing_pose_stats`, `sim1_cloth_variant_0002{,_stride2}_stats`, `pickup_3dpcwm_stats`) | 35 npz + 2 scripts | every alternative to `xy_p99` lost its A/B; keep the manifests as provenance |
| C8 | Dead `ModelConfig` flags and their code paths: `per_object_noise` (+ `run_per_object_inference_df`, per-object schedules), `use_windowed_temporal` + `create_sink_window_mask`, `use_cross_object_edge`, `predict_body_poses` + `BODY_POSE_*_SCALE`, `rope_position_scale`, legacy `AdaLayerNorm`, `encoder_pointnet_embedded_in`; opt-in-but-unused `decoder_anchor_identity`, `decoder_query_self_attn`, `banded_rope`, `use_continuous_physics_adaln`; the FM `shift` knob (dead on the DF path) | — | enabled in zero configs; each removal needs the `inference.py` alias table updated |
| C9 | Tier-2 design-phase tooling under `scripts/unified/` (chapter 6 §12 ARCHIVE list) | ~25 | studies concluded, results in tracked reports |

### D. Prune with the user (large, irreversible)

`runs/` 313 GB (smoke dirs ≈6 GB, pose sweeps ≈11 GB, unified smokes 13 GB, `drag_ovfit_*` 2.2 GB, plus the 5.3 GB UNet norm arms once the shared stripped copy is confirmed sufficient), `inference_results/` 31 GB, `wandb/` 2.7 GB, `checkpoints/` 3.7 GB, `/data/user_data/robertre/logs/` 1.5 GB. Never while a job is writing.

### E. Open decisions (the user's call)

1. 200K launch pre-flight (chapter 4 §6): hardware, snapshot cadence, `bucket_max_items` for the transformer, `fps_random_start`, compile on/off parity, and now the blocklist asymmetry (A2) and the pickup fix (A1).
2. Whether the unified UNet xy_p99 40K handoff should be re-issued after A1 (it trained on the held-out pickup assets).
3. Whether to regenerate the cloth corpus with the flat-layer assembly fixed (recovers 472 of 1,223 episodes); nothing to regenerate for pushing.
4. Scale conditioning (per-domain or per-window scale as an input) as a research item; not a prerequisite.
5. Whether `main` gets this branch (144 commits ahead) and when the website workflow may deploy.
6. Which of the colleague's inherited configs and launchers (mbronars' UNet R&D) are theirs to keep.
