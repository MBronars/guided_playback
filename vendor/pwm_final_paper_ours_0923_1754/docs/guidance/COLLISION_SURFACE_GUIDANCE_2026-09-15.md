# Collision-surface guidance and release screening

Sparse point distances can falsely suggest release. In the first 16-scene
pick/place experiment, the bowl placement from physical6, seed8312, passed the
sampled-point screen while its two finger collision surfaces still overlapped
the bowl by approximately1.0/1.1mm at the final frame. This is a limitation of
the old geometric screen, not evidence of executed placing success.

`--collision-backend fcl` adds optional signed distances between convex collision
components. It works with plain goal evaluation or proposal-free physical costs.
The default remains `sampled`, preserving the running baseline. FCL results
retain `sampled_validation` and the original sampled-point traces alongside
the mesh screen. Do not compare acceptance counts across different screens
without rescoring every trajectory under the same criteria.

Contact/release and penetration costs use full convex surfaces; joint limits,
support, rest/hold, goal and rotation definitions remain explicit task criteria.
For this pickup diagnostic, mesh contact tolerance is2mm, mesh penetration
tolerance2mm, release clearance5mm. The2mm penetration setting accommodates
the development recordings' proxy-depth maximum1.77mm; it is not a universal
simulator tolerance. Source geometry and preparation receipts remain required.
Cup decompositions are retained. Each component is reconstructed using a20µm
half-space membership tolerance; this is not exact source USD cooking.

`guidance/collision_distance.py` uses FCL witness points to obtain local
first derivatives with respect to body poses. It freezes closest-feature
selection; derivatives can change at feature switches and exact contact uses
a zero subgradient. This neither differentiates contact dynamics nor constrains
contact forces. Model weights, future-action hiding and sampler semantics are
unaffected. The optional dependencies are `python-fcl` and `scipy`.
`--collision-vendor PATH` can be repeated to append isolated import directories
after the model's NumPy/PyTorch imports.

Validation receipts under `validation/guidance_noise_2026-09-15/`:

- `tests_mesh_physical_v1.xml`:34 tests pass, including existing optimizer tests.
- `tests_mesh_release_v1.xml`:8 FCL tests pass. Analytic positive/negative box
  distances, translation/rotation derivatives, frame ordering, task-screen
  consistency, and a sparse-sampling false-release counterexample are covered.
- `collision_gradient_v1.json`:finite directional derivatives agree on all16
  real recorded scenes. Largest absolute derivative discrepancy0.000422; global
  translation gradients cancel. These are geometry checks, not model success.
- `mesh_audit_v2/`:independent FCL trajectory audits. Initial MuJoCo distance
  diagnostics in `mesh_audit_v1/` are **not qualified**: some returned zero
  distances with centimetres between witnesses. Its validity notice preserves
  that failure. MuJoCo replay controls use a different contact-query path and
  still require their separate positive-control/timestep qualification.

The interface follows the [Python-FCL proximity API](https://github.com/BerkeleyAutomation/python-fcl).
GPU smoke qualification passed on the first development pick and place scenes:
`g1v6_*_mesh_physical_al1_h16_smoke` completed one optimization update with
mesh constraints, finite gradients, and the independent mesh screen. Both still
fail task completion; these checks establish execution, not guidance efficacy.
The campaign rescores every completed baseline with the same mesh screen in
`mesh_validation_summary.json`. All64 six-update trials fail that shared screen.

Generated-outcome comparisons must accompany any claim that the new costs
improve the model's trajectories. The primary checkpoint and fixed scene
inventory are documented in `NOISE_OPT_OVERNIGHT_2026-09-15.md`.

## Gripper support-plane collision

An additional audit found that support costs/screens checked only the object.
The first can's goal-through24 trajectory put one finger24.3mm below the table.
All16 recorded control clips have nonnegative gripper collision-mesh clearance;
five of the first100 generated trajectories audited exceed3mm gripper intrusion.
This explains some failures, not all: the corresponding physical-through24
can has no gripper-table intrusion and still fails the independent replay.

Physical pick/place objectives now include `gripper_support`, using every
hand/finger collision vertex transformed by its predicted pose. It uses the
same3mm support tolerance and support weight as the object, with mean-plus-peak
penalty. Pick/place validators and proposal-free AL constraints include the
matching residual. `gripper_bottom` traces and maximum gripper support intrusion
remain separately visible. A stationary released object with its entire gripper
below the table is a regression case; pose-column reordering and upward gradients
are checked. No grasp proposal or prescribed closure schedule is introduced.

Existing source_v5/v6 campaign snapshots retain their earlier behavior. New
geometry-screen versions must rescore both baselines and new methods before
comparing acceptance rates. Independent replay outputs stay separately labeled
and require recorded-control and timestep qualification.
