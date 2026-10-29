# Local Rerun ablation comparison

[Download/open the Rerun recording](../../validation/guidance_study_2026-09-15/upgrade64/rerun/pick_development_00_asset_5624_terms_visible_v3.rrd)
(about 10 MB). This is the complete four-objective comparison for
`pick_development_00_asset_5624`, exported September 16, 2026.

Copy the `.rrd` file to your laptop and open it with Rerun 0.29.1:

```bash
python -m pip install 'rerun-sdk==0.29.1'
rerun pick_development_00_asset_5624_terms_visible_v3.rrd
```

The recording contains its layout and all point data; the model, simulator,
checkpoint and server connection are not needed to view it.
See [Rerun's file-loading documentation](https://rerun.io/docs/reference/sdk-operating-modes).

- **Guided comparison:** the four objectives in a 2-by-2 view.
- **Replay comparison:** the corresponding four simulator executions.
- **Individual objective tabs:** original unconditional sample, guided prediction,
  and simulator replay side by side.
- **Results and legend:** objective definitions, goal errors, overlap, and failed checks.
- **Terms explained (always visible on the left):** scroll for bullet-point calculations for every active loss term,
  including weights, tolerances, frame averaging, contact gates, noise
  regularization and the RMS cap. The contact variant is explicitly labeled
  **Goal + geometry/joints + contact**.

The explanations are embedded in the recording and also available as
[a standalone Markdown note](../../validation/guidance_study_2026-09-15/upgrade64/rerun/pick_development_00_asset_5624_terms_visible_v3.terms.md).

This version actively selects the saved layout and uses a new application ID.
Earlier exports only supplied a default layout, which Rerun can ignore when an
older layout is already active. Open the file ending in `terms_visible_v3.rrd`.

Press play or scrub the **time** timeline. All views share the same frame times:
the last observed frame plus 68 future frames at 20 Hz. Each trajectory contains
all 1,183 particle points. Orange is the object, gray the hand, and blue/green
the fingers. The red cross is the requested goal; red rings indicate 20 mm.
Every 3D view starts with the same camera and can be rotated and zoomed.

These are the same saved trajectories as the Markdown ablation GIFs. The original
sample is the actual unconditional draw, shared across all four objectives.
Replay points use the original body templates at the saved simulator poses;
finger positions are not clamped for display. Replay is an independent,
approximate simulator diagnostic, not source-simulator or real-robot validation.
All four objectives fail the qualified replay checks on this scene.

This file is a snapshot of one scene and seed. It does not update as later
experiments finish. Source hashes and export settings are in the
[export receipt](../../validation/guidance_study_2026-09-15/upgrade64/rerun/pick_development_00_asset_5624_terms_visible_v3.json).
Rerun's file verifier passed; the saved recording contains 36 body-point streams
with 69 samples each, representing nine trajectories.

The reusable exporter reads a completed comparison without running inference:

```bash
PYTHONPATH=. python scripts/dev/export_guidance_ablation_rrd.py \
  --root validation/guidance_study_2026-09-15/upgrade64 \
  --scene pick_development_00_asset_5624 \
  --out /path/to/new_comparison.rrd
```

Run the exporter in an environment with the repository's PyTorch dependencies and
`rerun-sdk==0.29.1`. It checks source hashes, shared initial samples, matched
budgets, frame alignment, and point materialization. It refuses to overwrite an
existing recording or export an incomplete four-way comparison.
