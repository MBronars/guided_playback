# Contact comparison extended to grid 256

Update: the user chose g256 as the practical shared-grid approximation.
[Native-g256 collection qualification](DRAG_G256_NATIVE_RELEASE.md) now tests
fresh settling, export and grouped execution on that grid. This does not change
the contact limitations or the historical measurements below.

## Controlled result

**256 improves the premature-interaction distance but does not eliminate
artificial sticking.** In the same-state block push/withdraw experiment:

| Method | Physical cell size | Gap at first target motion | Most negative target vx during withdrawal |
|---|---:|---:|---:|
| Shared grid, g192 | 5.208 mm | +8.002 mm | −0.1001 m/s |
| Shared grid, g256 | 3.906 mm | +2.304 mm | −0.0657 m/s |
| Geometry-gated separate grids, g64 | 15.625 mm | −0.0034 mm | No backward motion; minimum +0.3378 m/s |

Positive gap means interaction before the estimated material-cell surfaces
meet. The g256 near-miss target remains effectively stationary (less than
0.000001 mm COM excursion), and the free-impact onset gap is 2.348 mm.
The geometry-gated result remains a restricted aligned-block prototype, not a
general contact solver for irregular dataset shapes.

### Updated block comparisons

The top row now compares shared grids 64, 128, 192 and 256 in the **same 1 m
box**. The bottom row retains the separate-grid contact prototypes. All use
identical particles, material coefficients, initial states and 50 µs timestep.
The GIFs show actual saved material-point slices at 5× slow playback.

![Push and withdrawal](../../../runs/mpm-contact-256-20260915-v2/combined-blocks/review/push_withdraw.gif)

[Near-miss GIF](../../../runs/mpm-contact-256-20260915-v2/combined-blocks/review/near_miss.gif) ·
[Free-impact GIF](../../../runs/mpm-contact-256-20260915-v2/combined-blocks/review/free_impact.gif) ·
[All measured curves](../../../runs/mpm-contact-256-20260915-v2/combined-blocks/review/REPORT.md)

## Updated real-object comparisons

The same three scenes are compared at grids **100, 160, 192 and 256**. Their
box is 1.5625 m, so g256 here means **6.104 mm cells**, not the 3.906 mm cells
in the block benchmark. Each grid restores the exact full settled state and
replays the original particle-patch commands. There is no new settling, particle
resampling, material change or timestep change.

Matched no-command trajectories at every resolution distinguish residual
settling from differences caused by the commands. A refined grid can disturb
the original grid's settled state; raw target displacement alone is not a
reliable contact-onset metric. Particle-to-particle gaps are not signed mesh
surface distances, and driven-minus-control displacement is not a contact force.

At the first saved frame where commanded and no-command target COMs differ by
more than 1 mm, the measured nearest-particle gaps are:

| Scene / pair | g192 gap (mm) | g256 gap (mm) |
|---|---:|---:|
| 000013, 1 → 0 | 13.329 | 9.242 |
| 000017, 0 → 1 | 13.882 | 8.629 |
| 000038, 0 → 1 | 12.886 | 9.378 |
| 000038, 1 → 2 | 12.242 | 8.439 |

The real scenes therefore also show smaller gaps at g256, not zero gaps.
Maximum no-command target drift at g256 ranges from 9.326 to 17.352 mm, versus
6.852–9.812 mm at g192. These saved-frame, particle-distance observations must
not be interpreted as an exact native-step geometric contact distance.

The g256 three-object replay also changes deformation visibly, not just the gap.
Its position RMS versus the original g100 trajectory is 55.743 mm, compared
with 38.041 mm for g192. This is a discretization-dependent change, not an error
against known ground truth; the smaller gap alone is insufficient to rank the
entire trajectory as more accurate.

### Two objects, scene 000013

![Scene 000013](../../../runs/mpm-contact-256-20260915-v2/combined-driven/review/drag-000013.gif)

### Stiff elastic objects, scene 000017

![Scene 000017](../../../runs/mpm-contact-256-20260915-v2/combined-driven/review/drag-000017.gif)

### Three objects, scene 000038

![Scene 000038](../../../runs/mpm-contact-256-20260915-v2/combined-driven/review/drag-000038.gif)

[Real-scene measurements](../../../runs/mpm-contact-256-20260915-v2/combined-driven/review/REPORT.md) ·
[Matched no-command controls](../../../runs/mpm-contact-256-20260915-v2/combined-controls/control-review/REPORT.md)

## Reproduction and limits

- All nine new simulations completed: three block cases, three
  commanded real scenes and three no-command real-scene controls. Previous
  results are reused by explicit, hashed references, not rerun or overwritten.
- The native timestep remains 50 µs. Real-scene CFL values at g256 are 0.15189,
  0.33860 and 0.15189, below the unchanged 0.4 budget. The block CFL is 0.23732.
- The first attempted launch found the backend's software admission ceiling
  of 192. The only change to `physics.py` extends that ceiling to 256. Transfer,
  constitutive and native guard kernels remain unchanged. Regression tests
  admit 192/256 and reject 257 before allocation. The intentional source change
  is recorded separately from the historical source-parity hashes.
- Collection regression: **371 passed**. This is not a claim of scientific
  contact accuracy or permission to scale.
- The original failed setup attempt is retained under
  `runs/mpm-contact-256-20260915/`. The successful execution snapshot and outputs
  are under `runs/mpm-contact-256-20260915-v2/`.
- GPU sharing remains enabled under the same memory/process checks. These are
  diagnostic comparisons, not qualified throughput measurements. No production
  campaign configuration or frozen production source is changed.
- At fixed particle count, finer cells reduce particles per cell. This grid-only
  experiment is not joint particle/grid convergence, and a smaller visual gap
  does not by itself establish physically correct contact.

For original protocol details, failed-test-box diagnosis, geometry limitations
and the remaining production-contact work, see
[the original contact study](DRAG_CONTACT_ACCURACY.md).
