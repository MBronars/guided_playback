# Dragging interaction collection — 2026-09-15

## Status

**SCALING ON HOLD FOR CONTACT ACCURACY.** The 24-scene targeted GPU pilot retained
20 scenes and rejected four during settling. All four serial/grouped comparisons
passed, but agreement and motion transfer do not establish physically accurate
contact. The user observed interaction across visible gaps. Shared-grid coupling
can act before geometric contact and can cause artificial sticking.

Production has not started. `CONTACT_ACCURACY_HOLD.json` now prevents the local
publication script from issuing a gate or bundle. A separate spatial-resolution
and contact-algorithm study must be reviewed before reconsidering scaling.
The completed controlled comparison, same-state real-object replay GIFs and
remaining limitations are in [DRAG_CONTACT_ACCURACY.md](DRAG_CONTACT_ACCURACY.md).

This is the sand-free, sequence-v3 fixed-resolution method the user chose after
reviewing timestep GIFs. It does **not** establish timestep convergence. The
passive clock, constitutive laws, command actuator and physical quality filters
are unchanged. Only sampling and read-only outcome reporting were added.

## What changes in the dataset

- Training uses the existing 10,000-shape training bank, 1–4 objects, 1,024
  particles/object, homogeneous elastic or plasticine scenes at 10/100 kPa or
  1 MPa. No sand. Held-out banks remain separate and untouched by this pilot.
- For each multi-object count, attempted layouts are 30% separated controls,
  50% close pairs and 20% crowded chains. Single objects are separate controls;
  two-object crowded scenes are close pairs, not three-body chains.
- Starts have no intended overlap. Target-directed horizontal stages seek later
  interaction; ordinary side-wall, ceiling, floor-penetration, collapse,
  nonfinite and settling checks still govern retention. Missed interactions
  remain labelled misses, not secretly resampled scenes.
- Speeds remain continuous 0.05–0.60 m/s. The first aimed horizontal event gets
  enough duration to seek 14 cm of commanded travel where feasible; the 25 cm
  per-event bound is unchanged. All-lift targeted sequences get a final pull;
  pure lifts remain in controls. Reverse has one direction per stage.

See [the full protocol](DRAG_SEQUENCE_PROTOCOL.md) for exact timing, masks,
sampling and units. This is a driven material patch, **not a physical gripper**.

## Contact measurement correction — original result preserved

The first heuristic required two frames with sampled-point distance ≤20 mm,
then at least 5 mm of undriven-object movement. It misses real grid coupling:
the particle-to-grid interpolation has support wider than that cutoff. Several
pair scenes show 10–16 cm of target movement without ever crossing 20 mm.

The replacement diagnostic reconstructs each object's P2G grid mass at saved
frames using the actual 27-node quadratic weights. Its CPU reconstruction is
tested against the actual Warp CPU P2G kernel, as well as mass conservation,
support overlap, out-of-grid rejection and a quiet pre-drive baseline.

A **grid-mediated transfer candidate** requires two frames with at least 0.01%
of the smaller object's mass sharing grid nodes, subsequent target COM motion
of at least 5 mm, and no more than 1 mm of pre-drive target motion. This is not an
exact mesh-contact/impulse measurement and does not observe every native step.

The original distance heuristic and its failed pilot check are kept unchanged.
`METRIC_REVIEW.json` records the observer correction separately. No timestep,
trajectory, material law, wall threshold or physical-retention result was
changed to obtain these measurements. Production reporting records **both**
`INTERACTIONS.json` (distance proxy) and `GRID_COUPLING.json` (kernel-based
coupling). Shared-grid MPM contact can exhibit co-motion; this release does not
claim a separately calibrated multi-body friction/restitution law.

## Visual evidence

These are actual saved states, with fixed equal metric axes and 2× slow
playback. Colours identify objects; black points identify the driven patch.
XY is on the left; XZ is on the right. Keyframes are sampled visual inspection,
not a claim to have manually inspected every native simulation step.

### Four objects: the driven object pushes a neighbour; distractors stay separate

![Pair interaction](../../../runs/mpm-drag-interactions-20260915/object-gifs-v2/drag-000015.gif)

### Three objects: push transfers motion down the chain, then reverse separates

![Crowded pull and reverse](../../../runs/mpm-drag-interactions-20260915/object-gifs-v2/drag-000038.gif)

### Four objects, stiff elastic: lift, release, then targeted push

![Stiff elastic interaction](../../../runs/mpm-drag-interactions-20260915/object-gifs-v2/drag-000115.gif)

[All object-coloured GIFs](../../../runs/mpm-drag-interactions-20260915/object-gifs-v2/GALLERY.md)
and [all 20 retained clips' object-coloured keyframes](../../../runs/mpm-drag-interactions-20260915/contact-sheets/GALLERY.md).

## Evidence and launch layout

Local immutable pilot: `runs/mpm-drag-interactions-20260915/`.
The first proposed collection is bounded to one currently allocated PRO6000,
11 hours or 2,160 attempts, whichever ends first (allocation and shutdown
reserves also apply). This is an attempt cap, **not** a promised retained count.
The launch operator requires a source/bank-bound reviewed gate.

The planned shared bundle is:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/
  assets/mpm/fresh-10k-v1/                   # unchanged split-aware bank
  evidence/mpm-drag-interactions-20260915-v1/ # reviewed certificate + evidence
  staging/mpm-drag-interactions-20260915-v1/  # immutable executable bundle
    source/collection/mpm/drag/
    GATE.json
    BUNDLE.json
    collect.sbatch
    collection/block-*/                     # attempts, raw, rejected, diagnostics
    data/train/train/block-*/diagnostic-export/  # prefiltered training shards
```

The retained export uses the explicit drag adapter (`task: drag`) and both
existing normalization profiles. Passive readers still reject active inputs.
Every new bundle reserves a unique namespace; do not rerun a consumed bundle or
overwrite a previous run. The current certificate will be train-only,
homogeneous, 1–4 objects; mixed-material and held-out/count-evaluation expansion
need their own review. The passive campaign is not modified by this work.
