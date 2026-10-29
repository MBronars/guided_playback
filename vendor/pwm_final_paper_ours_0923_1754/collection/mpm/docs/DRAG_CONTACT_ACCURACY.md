# Dragging contact accuracy — 2026-09-15

**Update:** [The completed g256 extension and updated comparison GIFs](DRAG_CONTACT_256.md)
add a fourth shared-grid resolution to the study below. The original results
are retained; g256 reduces the gap but does not eliminate sticking.

## Decision

**The visible-gap interaction is a real shared-grid artifact, not just GIF
rendering. Separate object grids with a geometry-aware contact condition are
the strongest candidate in the controlled tests. Increasing grid resolution
alone does not solve both premature interaction and artificial sticking.**

This is a diagnostic result, **not approval to scale** the new interaction-rich
dragging dataset. The explicit-contact implementation tested here is restricted
to two aligned blocks with a known normal. It has not replaced the production
collector and must not be applied unchanged to arbitrary object meshes.

The passive campaign, its frozen sources, timestep and existing data are not
changed. The interaction pilot's `CONTACT_ACCURACY_HOLD.json` remains in force.

## What was compared

The final controlled study contains 18 completed trajectories: six method/grid
settings, each tested on three cases. All use the same 8,192 volumetric material
points, particle volumes, density, elastic coefficients, initial velocities and
50 µs native timestep. There are no walls or gravity in these block tests.

- Two solid 10 cm cubes; 16³ cell-centred points per cube; E=100 kPa,
  nu=0.2, density=1,000 kg/m³. This is volumetric quadrature, **not** the
  production shape bank's 1,024 surface-sampled points per object.
- **Near miss:** initially 20 mm apart in X, the first cube slides in Y at
  0.15 m/s. The second cube should not move. Duration 0.6 s.
- **Push/withdraw:** initially 40 mm apart; command the first cube at +0.2 m/s
  for 0.3 s, then −0.2 m/s for 0.3 s, then release for 0.1 s. The entire
  first cube is commanded for this analytic test, not the production local
  particle patch. Velocity is overridden before P2G; position is not kinematic.
- **Free impact:** initially 40 mm apart; give the first cube 0.2 m/s initial
  velocity with no subsequent commands. Duration 0.6 s. This is the
  closed-system linear-momentum check.

Each state is saved every 5 ms. Contact onset below means the first saved state
where the target COM speed exceeds 1 mm/s. The surface estimate uses deformed
material-cell X extents, not nearest particle-centre distance.

Methods:

1. Unchanged shared velocity grid at dx=15.625, 7.8125 and 5.2083 mm.
2. Independent object grids with a frictionless, unilateral normal impulse at
   shared nodes, dx=15.625 mm. There is **no geometry gate** in this variant.
3. The same separate grids and impulse, additionally gated by the estimated
   material-cell face gap, at dx=15.625 and 7.8125 mm.

The third method uses a known +X normal and a global X-envelope for these
aligned blocks. It is a geometry oracle for this benchmark, not a general
surface-contact implementation or a full reproduction of an established method.

## Measured outcome

Positive onset gap means motion starts before the estimated surfaces meet.
Negative target velocity during withdrawal means the uncommanded target moves
backward. These are measured diagnostic values, not newly invented pass gates.

| Method | dx (mm) | Near-miss target excursion (mm) | Push onset gap (mm) | Minimum target vx during/after withdrawal (m/s) |
|---|---:|---:|---:|---:|
| Shared grid | 15.625 | 17.750 | +36.001 | −0.260 |
| Shared grid | 7.8125 | 0 | +12.056 | +0.160 |
| Shared grid | 5.2083 | 0 | +8.002 | −0.100 |
| Separate grids, no geometry gate | 15.625 | 0 | +36.001 | +0.291 |
| Separate grids, geometry gate | 15.625 | 0 | −0.0034 | +0.338 |
| Separate grids, geometry gate | 7.8125 | 0 | −0.0213 | +0.356 |

The geometry-gated push cases' maximum estimated X-envelope penetration is
0.0194 and 0.0305 mm, respectively. This does **not** establish micrometre
accuracy on arbitrary shapes; the benchmark has known, aligned faces and a
cell-envelope approximation. Neither a monotonic convergence claim nor a
friction/restitution calibration follows from these two resolutions.

The first shared-grid setting visibly rotates/drags the untouched cube during
the near miss. Separating velocity fields removes that spurious tangential
coupling, but without a geometry condition the normal interaction still starts
36 mm early. That is why “separate grids” alone is not the complete fix.

All methods retain the existing native finite-state/deformation/stencil/speed
guards. Linear momentum is measured on the free-impact case. The reported
energy proxy includes translational kinetic and elastic strain energy but
**omits affine APIC kinetic energy**; it is not an exact full-energy certificate.
The commanded push case receives external work and is not a closed-energy test.

### Controlled comparison GIFs

Actual simulated states, common axes and object colours, a fixed central material
slice, no interpolation, 5× slow playback. All 8,192 points are simulated and
included in measurements, even though only the slice is drawn.

![Near miss](../../../runs/mpm-contact-accuracy-20260915-padded/review/near_miss.gif)

![Push and withdrawal](../../../runs/mpm-contact-accuracy-20260915-padded/review/push_withdraw.gif)

![Free impact](../../../runs/mpm-contact-accuracy-20260915-padded/review/free_impact.gif)

[Measured curves and all results](../../../runs/mpm-contact-accuracy-20260915-padded/review/REPORT.md).

## Real shape replays

Three retained pilot scenes were replayed at grid sizes 100, 160 and 192:
`drag-000013`, `drag-000017`, and `drag-000038`. All nine completed.

Each replay restores the **exact full settled checkpoint**: positions, FP64
master positions, velocities, F/F_trial, affine C, plastic history, material
coefficients and particle weights. It uses the exact original selected patch
and command arrays, dt=50 µs, physical box and wall planes. No re-settling or
patch reselection is performed. Grid spacing is 15.625, 9.765625 and 8.1380 mm.

At the original resolution, trajectory RMS versus the original recorded pilot
is 0.0028, 0.0010 and 0.0010 mm respectively. The replay is therefore a close
baseline reproduction, not a newly sampled scene. Refined-grid RMS differences
range from 10.98 to 38.04 mm; these reflect changed spatial discretization and
are not claimed to be errors against ground truth.

![Two-object replay](../../../runs/mpm-contact-accuracy-20260915-v4/replays/review/drag-000013.gif)

![Stiff elastic replay](../../../runs/mpm-contact-accuracy-20260915-v4/replays/review/drag-000017.gif)

![Three-object replay](../../../runs/mpm-contact-accuracy-20260915-v4/replays/review/drag-000038.gif)

[Real-scene measurements](../../../runs/mpm-contact-accuracy-20260915-v4/replays/review/REPORT.md).

These replay images show that changing grid size changes interaction distances
and trajectories. They do not prove that the finer grid is physically correct.
In particular, a state settled on one grid can move when transferred to another
grid. All nine matched **no-command controls** also completed. Maximum target
drift without commands is 0.006–0.575 mm on the original grid, 3.674–8.739 mm
at grid 160, and 6.852–9.812 mm at grid 192. Thus raw target movement alone
would falsely attribute some re-equilibration to the drag.

Comparing commanded and matched no-command target trajectories, their COMs
first differ by more than 1 mm at particle gaps of 22.81–30.63 mm on grid 100,
14.87–16.54 mm on grid 160, and 12.24–13.88 mm on grid 192. This supports the
same conclusion as the blocks: finer resolution reduces the interaction gap,
but does not remove it.

[Control measurements and curves](../../../runs/mpm-contact-accuracy-20260915-padded/replays/control-review/REPORT.md).

Nearest particle distance is not a signed mesh
surface distance, and subtracting nonlinear trajectories is not a contact-force
measurement.

## Test-harness corrections, preserved rather than hidden

The first 0.5 m free-space box was too small for successful target rebounds.
Three separate-grid push cases stopped at the stencil guard around
0.483–0.551 s. Failure-state capture showed the target at the high-X grid edge,
with finite velocities below 0.48 m/s and no reported nonpositive deformation.
This was **not evidence of a numerical explosion**.

The final study doubles the empty box and grid count while retaining dx,
grid origin, particle positions and all physical inputs. All 18 cases then
complete. Across the 15 original completed cases, padding changes trajectory
RMS by at most 0.000957 mm. The failed original runs and rejected states remain
available in `runs/mpm-contact-accuracy-20260915-failure-diagnostics/`.

In run folder names, `g32/g64/g96` identify the reference 0.5 m resolution.
The padded runs actually use 64/128/192 cells in a 1 m box. Always compare
**dx**, not raw grid count across these two boxes.

A regression test caught a padding-only CFL reporting bug: the frozen padded
study used the old cell count when computing its reported CFL, understating it
by a factor of two. The actual simulator dx and dt were correct. Recalculation
from all 18 saved configurations gives CFL=0.05933, 0.11866 or 0.17799, all
below the unchanged 0.4 budget. The working source is fixed; archived requests
are not overwritten. See
[the explicit correction receipt](../../../runs/mpm-contact-accuracy-20260915-padded/CFL_REPORTING_CORRECTION.json).

Negative X-envelope gaps in rotating near-miss bodies mean overlapping X
projections; they do not by themselves prove 3D penetration. That limitation
is distinct from the near-miss target moving when it should remain stationary.

## What remains before production contact changes

1. Generalize geometry-aware contact to the actual irregular shapes: local
   surface separation and normals, not a hardcoded +X envelope.
2. Specify and test inter-object friction and separation/rebound behavior.
   Cover oblique, grazing, unequal-mass and simultaneous multi-object contacts.
3. Repeat the same-state real-scene comparisons with the generalized solver,
   including no-command controls, wall checks and collapse/deformation guards.
4. Check joint particle/grid resolution sensitivity. Refining the grid at fixed
   particle count reduces particles per cell; it is not a complete convergence
   study, especially for surface-sampled production assets.
5. Only then qualify execution parity, the active-data training contract and
   throughput for the new solver, before issuing a new immutable scale bundle.

The per-body-grid approach is motivated by established contact work, but our
prototype is deliberately narrower. See
[Bardenhagen et al. (2001)](https://www.sci.utah.edu/~guilkey/MPMPubs/Bardenhagen-ImprovedContact-01.pdf)
for separate velocity fields, contact/separation logic and associated caveats.

## Code and execution

Diagnostic code lives in `collection/mpm/drag/contact_study.py`,
`contact_replay.py`, and `study_device.py`. Production stepping kernels and
production GPU exclusivity checks were not modified by this study.

GPU sharing was explicitly authorized. Diagnostics allow the identified
co-tenant, stop for unapproved processes, enforce a 16 GiB own-memory ceiling
and require 8 GiB free. These shared-mode runs are **not throughput benchmarks**.
The identified co-tenant had exited before the successful study started; it was
not killed to make room. No new production data collection job was launched.

Working-tree collection regression after the diagnostic fixes: **368 tests passed**.
Frozen run snapshots, requests, completion/failure receipts and raw trajectories
are retained alongside the visual reviews. Numeric completion and visual review
are separate from any future scientific accuracy or production scale gate.
