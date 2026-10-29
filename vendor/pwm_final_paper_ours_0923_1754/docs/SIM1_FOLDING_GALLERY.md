# Completed SIM1 folding collection — test motion gallery

> **Status note, 2026-09-19.** The GIFs and stills live in the repository's untracked `validation/folding_gallery_20260916/`
> folder on the lab volume (41 MB), so they do not render on GitHub. Training validation has since been done: see the
> folding sections of [DATASETS.md](DATASETS.md).

Rendered September 16, 2026 directly from retained test NPZs. No resimulation or model predictions.

**Collection and format documentation are complete; plug-and-play training validation is not.**

See the [dataset format and training handoff](SIM1_FOLDING_DATASET.md) for exact loading contracts and remaining integration checks.

## Final collection

| Split | Retained clips | Garments with retained clips | Usable states |
|---|---:|---:|---:|
| train | 4,510 | 939 | 4,068,020 |
| test | 236 | 49 | 212,872 |

All 1,000 planned garments were processed. 4,746 of 5,000 planned action slots were retained; 12 garments yielded no retained clips. These are screened candidates, not certified successful folds.

## How to read the GIFs

- Each GIF covers all 902 exported states in time, displayed every sixth state plus the final state (152 displayed frames). About 30 seconds, real-time playback, looping.
- Left panel: top view. Right panel: side view. Axes are metres, with table height at z=0. Cameras stay fixed within each clip; bounds differ across clips.
- Cloth colors follow persistent material points, not separate objects. Green/orange points are the left/right grippers; full arms are omitted.
- Display-only spatial subsampling uses up to 6,000 cloth points and 400 per gripper. Stored data are unchanged and remain 30 Hz with 1,000 points per gripper.
- Selection was fixed by family/action and lexicographic garment ID, not by which motions looked best. Six clips are illustrative, not a population-level quality audit.
- The lift/carry and geometry gates do not guarantee a neat fold. Watch the full approach, grasp, lift/carry, release, and return.

GIFs are local generated artifacts under `validation/folding_gallery_20260916/`, not embedded in the Markdown file or supplied by a fresh checkout. To share this page, include that asset folder and preserve relative paths.

## 1. Tee — left-inward v0

Garment: `tee__tee_1MWCTRN0EU` · action `0` · 13,659 cloth particles.

Collection gate: passed. Peak p99 stretch: **1.120**; stretch warning: **False**. Sampled geometry and lift/carry proxy passed.

![tee__tee_1MWCTRN0EU action 0](../validation/folding_gallery_20260916/tee__tee_1MWCTRN0EU__action-0.gif)

[Open GIF](../validation/folding_gallery_20260916/tee__tee_1MWCTRN0EU__action-0.gif) · [Six-timepoint contact sheet](../validation/folding_gallery_20260916/tee__tee_1MWCTRN0EU__action-0.png) · [Source/hash/render receipt](../validation/folding_gallery_20260916/tee__tee_1MWCTRN0EU__action-0.json)

Source selected NPZ:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-restart-20260915-v3/production/new/test/tee__tee_1MWCTRN0EU/loader-candidates/action-0.npz
```

## 2. Pants — right-diagonal v0

Garment: `pants__pants_straight_sides_A1NQSJ7M9U` · action `1` · 5,050 cloth particles.

Collection gate: passed. Peak p99 stretch: **1.130**; stretch warning: **False**. Sampled geometry and lift/carry proxy passed.

![pants__pants_straight_sides_A1NQSJ7M9U action 1](../validation/folding_gallery_20260916/pants__pants_straight_sides_A1NQSJ7M9U__action-1.gif)

[Open GIF](../validation/folding_gallery_20260916/pants__pants_straight_sides_A1NQSJ7M9U__action-1.gif) · [Six-timepoint contact sheet](../validation/folding_gallery_20260916/pants__pants_straight_sides_A1NQSJ7M9U__action-1.png) · [Source/hash/render receipt](../validation/folding_gallery_20260916/pants__pants_straight_sides_A1NQSJ7M9U__action-1.json)

Source selected NPZ:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-noon-20260916-v4/production/new/test/pants__pants_straight_sides_A1NQSJ7M9U/loader-candidates/action-1.npz
```

## 3. Hooded Jacket — two-hand-half v0

Garment: `hooded_jacket__jacket_hood_55OXILONT9` · action `2` · 13,972 cloth particles.

Collection gate: passed. Peak p99 stretch: **1.163**; stretch warning: **False**. Sampled geometry and lift/carry proxy passed.

![hooded_jacket__jacket_hood_55OXILONT9 action 2](../validation/folding_gallery_20260916/hooded_jacket__jacket_hood_55OXILONT9__action-2.gif)

[Open GIF](../validation/folding_gallery_20260916/hooded_jacket__jacket_hood_55OXILONT9__action-2.gif) · [Six-timepoint contact sheet](../validation/folding_gallery_20260916/hooded_jacket__jacket_hood_55OXILONT9__action-2.png) · [Source/hash/render receipt](../validation/folding_gallery_20260916/hooded_jacket__jacket_hood_55OXILONT9__action-2.json)

Source selected NPZ:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-recovery-20260914-v2/production/new/test/hooded_jacket__jacket_hood_55OXILONT9/loader-candidates/action-2.npz
```

## 4. Dress — left-inward v2

Garment: `dress__dress_sleeveless_72XRXUATVL` · action `3` · 12,468 cloth particles.

Collection gate: passed. Peak p99 stretch: **1.143**; stretch warning: **False**. Sampled geometry and lift/carry proxy passed.

![dress__dress_sleeveless_72XRXUATVL action 3](../validation/folding_gallery_20260916/dress__dress_sleeveless_72XRXUATVL__action-3.gif)

[Open GIF](../validation/folding_gallery_20260916/dress__dress_sleeveless_72XRXUATVL__action-3.gif) · [Six-timepoint contact sheet](../validation/folding_gallery_20260916/dress__dress_sleeveless_72XRXUATVL__action-3.png) · [Source/hash/render receipt](../validation/folding_gallery_20260916/dress__dress_sleeveless_72XRXUATVL__action-3.json)

Source selected NPZ:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-restart-20260915-v3/production/new/test/dress__dress_sleeveless_72XRXUATVL/loader-candidates/action-3.npz
```

## 5. Eight Panel Skirt — right-diagonal v2

Garment: `eight_panel_skirt__skirt_8_panels_6END0302VH` · action `4` · 5,648 cloth particles.

Collection gate: passed. Peak p99 stretch: **1.160**; stretch warning: **False**. Sampled geometry and lift/carry proxy passed.

![eight_panel_skirt__skirt_8_panels_6END0302VH action 4](../validation/folding_gallery_20260916/eight_panel_skirt__skirt_8_panels_6END0302VH__action-4.gif)

[Open GIF](../validation/folding_gallery_20260916/eight_panel_skirt__skirt_8_panels_6END0302VH__action-4.gif) · [Six-timepoint contact sheet](../validation/folding_gallery_20260916/eight_panel_skirt__skirt_8_panels_6END0302VH__action-4.png) · [Source/hash/render receipt](../validation/folding_gallery_20260916/eight_panel_skirt__skirt_8_panels_6END0302VH__action-4.json)

Source selected NPZ:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-30hz-recovery-20260914-v2/production/new/test/eight_panel_skirt__skirt_8_panels_6END0302VH/loader-candidates/action-4.npz
```

## 6. Jumpsuit — two-hand-half v0

Garment: `jumpsuit__jumpsuit_sleeveless_D2K6R1XHPE` · action `2` · 10,317 cloth particles.

Collection gate: passed. Peak p99 stretch: **1.200**; stretch warning: **True**. Sampled geometry and lift/carry proxy passed.

![jumpsuit__jumpsuit_sleeveless_D2K6R1XHPE action 2](../validation/folding_gallery_20260916/jumpsuit__jumpsuit_sleeveless_D2K6R1XHPE__action-2.gif)

[Open GIF](../validation/folding_gallery_20260916/jumpsuit__jumpsuit_sleeveless_D2K6R1XHPE__action-2.gif) · [Six-timepoint contact sheet](../validation/folding_gallery_20260916/jumpsuit__jumpsuit_sleeveless_D2K6R1XHPE__action-2.png) · [Source/hash/render receipt](../validation/folding_gallery_20260916/jumpsuit__jumpsuit_sleeveless_D2K6R1XHPE__action-2.json)

Source selected NPZ:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/folding-final-30hz-20260914-v1/production/new/test/jumpsuit__jumpsuit_sleeveless_D2K6R1XHPE/loader-candidates/action-2.npz
```

## Readiness before training

The six-timepoint contact sheets show localized cloth lifting/carrying followed
by release and settling. Much of each garment remains near its initial position;
these are partial manipulations, not six demonstrations of a neat completed fold.
That distinction matters when choosing training windows and evaluation metrics.
The contact-sheet inspection is not a continuous collision or dynamics audit.

The NPZ example in the format guide is tested. The legacy WDS loader still needs a verified new-data adapter/materialization, new gripper link/reference compatibility, chosen normalization, and a real model forward/backward smoke test. None of those model-level checks is implied by rendering these GIFs.

Final membership comes only from the v4 `production/train-INDEX.json` and `test-INDEX.json`; older generations are inherited by reference, not additive datasets.
