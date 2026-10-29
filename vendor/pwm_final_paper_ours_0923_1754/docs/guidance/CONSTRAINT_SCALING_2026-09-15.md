# Tolerance-scaled physical constraints

Unscaled augmented-Lagrangian residuals use metres, including angle residuals
converted to metres. This leaves a large initial goal miss numerically much
larger than contact violations. On the first can's development AL24 trial,
the initial goal residual is roughly140mm while collision/contact residuals are
around10mm. Early updates reduce the goal miss while worsening collisions.
This is a nonconvex optimization failure, not a guarantee from the AL formulation.

An optional `--constraint-scaling tolerance` divides each signed residual by
its declared tolerance before the AL calculation and dual update. Acceptance
thresholds remain unchanged. A contact residual of4mm at a2mm tolerance becomes2;
a goal residual of40mm at a20mm tolerance also becomes2. Raw metre residuals
are retained alongside scaled values, and the complete scale map is saved in
`protocol.json`. Exact-zero/tolerance boundaries are not relaxed.

The penalty's units change with this option, so the CLI requires an explicit
`--constraint-penalty`. The first development recipe uses0.02 with dual updates
every8 optimizer updates, versus the unscaled2000/every2 recipe. These settings
are an experiment, not a theoretical convergence guarantee or an exact paper
reimplementation. Candidate selection still uses the base objective plus any
explicit feasibility-first `--screen`; augmented values from changing duals
are not compared across iterations for final ranking.

```bash
# Add to a qualified proposal-free physical noise-optimization command:
--constraints --constraint-scaling tolerance --constraint-penalty .02 \
--dual-every 8 --screen
```

With scaling disabled, the existing AL calculation is preserved. Tests cover
raw-versus-scaled units, penalty values, dual updates, physical gradients,
unchanged screen thresholds and exact coverage of the active constraint keys.
The affected32 tests pass, and real H16 one-update pick/all-noise and
place/actuator-only GPU smoke runs pass at source641a016:
`validation/guidance_noise_2026-09-15/scaled_constraint_gpu_gate_v1/report.json`.
Matched H36 development comparisons are running. Functional qualification does
not establish that this option improves physical results.
