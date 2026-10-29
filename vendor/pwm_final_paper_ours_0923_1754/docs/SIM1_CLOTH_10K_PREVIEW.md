# Cloth particle cap: original versus 10,000 points

> **Status note, 2026-09-19.** This preview of a fixed 10,000-point cap led to a different rule: an 8 mm farthest-point
> spacing target with a 10,000-vertex ceiling, released as `pwm-data-v1/folding_8mm` (see [DATASETS.md](DATASETS.md)).
> The images live in the untracked `validation/cloth_10k_preview_20260918/` folder (33 MB) and do not render on GitHub.

September 18, 2026. **Visualization experiment only. No collection files,
training indices, or model settings were changed.** The originals remain intact.

The examples are a 13,659-point T-shirt, an 18,304-point dress, and the largest
retained garment, a 32,739-point hooded jacket. Each is reduced to exactly
10,000 cloth points. Both grippers retain all 1,000 points each, for a maximum
of **12,000 total model input points**, not 10,000 total.

## Reading the comparisons

- Left is original; right is the capped subset, at exactly the same saved times.
- Static images show the initial exported pose, a side view at peak mean cloth
  height, and the same **14 cm material-panel patch** enlarged to expose spacing.
- Original and capped views use the same camera limits and marker size. Every
  cloth point is drawn in each view; there is no second display downsampling.
- GIFs compare top and side views throughout the full saved clip. They display
  5 frames/second at real-time playback speed; the data remain 30 Hz.
- Cloth colors identify persistent material positions, not different objects.
  GIF grippers are green and orange and are identical in both columns.

## Method and scope

Measured results (millimetres, material-rest **mesh-edge path** distances):

| Example | Original cloth points | Retained points | Median neighbor spacing, before → after | Maximum distance from any original vertex to its nearest retained vertex |
|---|---:|---:|---:|---:|
| T-shirt | 13,659 | 10,000 | 7.99 → 8.83 mm | 8.05 mm |
| Dress | 18,304 | 10,000 | 10.05 → 12.36 mm | 11.13 mm |
| Largest hooded jacket | 32,739 | 10,000 | 7.91 → 13.69 mm | 10.71 mm |

The displayed silhouettes and lifted regions remain recognizable in these
examples. The close-ups show visibly reduced density, especially for the largest
garment: spacing is not identical to the original. The coverage column measures
distance from **original vertices**, not every continuous point on a triangle,
and is not a learned prediction-error bound. Measurements use the source rest
metric; spacing can still change as the simulated cloth stretches or folds.

### Sampler

This prototype selects existing vertices by deterministic farthest-point
sampling on the cloth's **triangle-edge graph**, weighted by material/rest UV
edge lengths. It uses the frozen garment input, not a deformed trajectory frame.
Shared-edge lengths are averaged across incident faces. Seed vertex is 0;
ties use vertex order. The fixed selected indices can be reused for all frames
and all actions of that garment. Mesh connectivity prevents shortcuts merely
because two layers happen to touch in 3D.

This is approximate surface spacing along mesh edges, not exact continuous
geodesic distance or Euclidean nearest-neighbor distance. It does not remesh,
add interpolated points, independently sample each frame, or change physics.
Each connected component receives coverage when the point budget permits it.
There is **no explicit seam/edge quota or semantic landmark guarantee** in this
prototype; boundary coverage is measured separately. These comparisons do not
establish model-quality parity or prove all thin features remain adequate.

The selected source NPZs and mesh inputs are SHA-256 verified. Selection and
spacing were checked against independent SciPy shortest paths on three random
graphs, plus simple line, disconnected-component, and no-reduction cases.
Point indices, metric summaries, and provenance are saved beside the figures.
All three 152-frame GIFs, fixed 10,000-point subsets, unchanged gripper indices,
and document asset links were checked after rendering.

## 1. T-shirt: 13,659 → 10,000 cloth points

![T-shirt original and capped views with spacing close-up](../validation/cloth_10k_preview_20260918/tee__tee_1MWCTRN0EU_comparison.png)

[Synchronized before/after GIF](../validation/cloth_10k_preview_20260918/tee__tee_1MWCTRN0EU.gif) ·
[Spacing distributions](../validation/cloth_10k_preview_20260918/tee__tee_1MWCTRN0EU_spacing.png) ·
[Metrics and source](../validation/cloth_10k_preview_20260918/tee__tee_1MWCTRN0EU.json)

## 2. Dress: 18,304 → 10,000 cloth points

![Dress original and capped views with spacing close-up](../validation/cloth_10k_preview_20260918/dress__dress_sleeveless_BO3SYDJ3KH_comparison.png)

[Synchronized before/after GIF](../validation/cloth_10k_preview_20260918/dress__dress_sleeveless_BO3SYDJ3KH.gif) ·
[Spacing distributions](../validation/cloth_10k_preview_20260918/dress__dress_sleeveless_BO3SYDJ3KH_spacing.png) ·
[Metrics and source](../validation/cloth_10k_preview_20260918/dress__dress_sleeveless_BO3SYDJ3KH.json)

## 3. Largest garment: 32,739 → 10,000 cloth points

![Largest jacket original and capped views with spacing close-up](../validation/cloth_10k_preview_20260918/hooded_jacket__jacket_hood_ST57P6E999_comparison.png)

[Synchronized before/after GIF](../validation/cloth_10k_preview_20260918/hooded_jacket__jacket_hood_ST57P6E999.gif) ·
[Spacing distributions](../validation/cloth_10k_preview_20260918/hooded_jacket__jacket_hood_ST57P6E999_spacing.png) ·
[Metrics and source](../validation/cloth_10k_preview_20260918/hooded_jacket__jacket_hood_ST57P6E999.json)

## Reproduction and artifacts

Script: `validation/preview_cloth_10k.py`. Outputs:
`validation/cloth_10k_preview_20260918/`. The script requires a fresh output
directory and does not overwrite an existing preview. Source clip choice is
fixed: the previously viewed T-shirt, the retained dress nearest 22k vertices
(18,304 in this collection), and the largest retained garment; lowest retained
action slot per garment. These illustrative examples include train and test
garments and are not a held-out model evaluation.

Each `*_subset.npz` contains cloth indices, cloth-plus-gripper indices, and
mesh-distance coverage for original vertices. These are **preview selections,
not a published training export**. `RESULTS.json` aggregates the measurements.

The figures and GIFs are generated local assets, not embedded in this Markdown
file. Share the asset folder along with this document if colleagues need to
view it elsewhere. See the [full-resolution dataset guide](SIM1_FOLDING_DATASET.md)
for the original, unchanged dataset schema.
