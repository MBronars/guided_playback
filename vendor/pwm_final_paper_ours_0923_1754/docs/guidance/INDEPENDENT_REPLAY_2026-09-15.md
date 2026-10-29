# Independent pickup replay diagnostic

`scripts/dev/replay_pickup_mujoco.py` replays a saved generated gripper trajectory
against a dynamic object. It covers this repository's four-body Panda pick/place
fixtures: passive object, hand, and two prismatic fingers. It uses the observed
context to initialize poses and estimate velocities. It never prescribes the
object's future pose or loads source joint commands for a generated replay.

This is a CPU MuJoCo diagnostic. It differs from source PhysX in contact solving,
inertia/COM approximation, collision cooking and controller realization. The
floating hand follows a dynamic weld to a mocap target; arm reachability, arm
torques and robot execution are not tested. Table height is the fixture's declared
support reference, which is inferred for placing. Qualified recorded controls
and timestep comparisons are required before interpreting generated outcomes.

Optional dependencies used here: MuJoCo3.3.7, Python-FCL0.7.0.10 and SciPy.
They are not imported by training. `--dependency-path` can append isolated
dependency directories after NumPy/PyTorch load, avoiding accidental replacement
of the model environment's NumPy. Standard installed packages also work.

```bash
PYTHONPATH=. python scripts/dev/replay_pickup_mujoco.py \
  --scene <prepared-scene-directory> \
  --trajectory <generated-seed.npz> --frames 36 \
  --finger-mode mesh_preload --step .0005 --out <new-output-directory>
```

The NPZ must contain `R` and `T`, including observed context, in the four-body
order above. Its context must match the fixture. The script writes the scene
XML, complete protocol/hashes, executed body poses and finger positions, contact
counts, tracking/divergence metrics and simulator warnings. It refuses existing
output directories. Outputs are diagnostics, not automatically task-success labels.

Finger policies are explicit. `achieved` directly commands the generated finger
positions. `mesh_preload` commands an additional6mm closure while both planned
finger/object collision-surface gaps are at most2mm. These defaults are fixed
exploratory controller choices, not universal robot control parameters. `recorded`
uses source commands only for positive controls and is rejected with `--trajectory`.
Under contact load, source finger commands can differ substantially from achieved
finger poses; directly commanding achieved poses loses the recorded picks here.

The first matched control study covers all16 fixed scenes at0.5ms; the fixed
mesh-preload controller is also repeated at0.25ms. Qualification requires finite
outputs, no simulator warnings, hand tracking within3mm/3°, final object agreement
within20mm/20°, and maximum timestep-to-timestep object differences within5mm/5°.
Eleven scenes pass. Five fail and remain reported: pick6079,5173,6778 and
place5423,11421. Controller investigation included reserved control clips, so it
is exploratory rather than a blind simulator-validation benchmark. Guidance
recipe reservation is a separate designation.

Receipts under `validation/guidance_noise_2026-09-15/`:

- `mujoco_controls_matched_v2_summary.json`: all control/timestep measurements.
- `replay_generated_gate_v1/parity.json`: replacing recorded future states/actions
  with NaNs leaves every generated replay array bitwise identical.
- `replay_integrated_gate_v1/parity.json`: the integrated script reproduces the
  qualified prototype bitwise for a generated pick and a recorded placing control.
- `generated_replay_summary_v2.json`: shared mesh task checks on both integration
  steps, tracking checks, and recorded-control qualification for every replay.
- `generated_replay_report_v2/index.html`: synchronized model prediction above
  independent replay, with every scene retained.

Generated plans can be more timestep-sensitive than their recorded controls.
For the first can's physical100 plan,0.25/0.125ms replays lift roughly77/78mm,
whereas0.5/0.0625ms runs lose the object. This fails numerical robustness and is
not a successful execution result. Favor robust scene-level outcomes over one
favorable timestep, seed or video.
# Optional velocity feedforward (06:49 EDT continuation)

The replay CLI accepts `--finger-velocity-feedforward` for achieved-position
controllers. A MuJoCo position actuator applies `kp*(target-q)-kv*qdot`;
adding `(kv/kp)*desired_velocity` to its position target compensates damping
along the planned motion. This does not supply acceleration compensation or
grip force. The existing contact preload, force limits and joint travel limits
remain in effect. Velocity comes from planned achieved finger positions, so
switching preload on does not introduce an artificial velocity impulse.
Recorded-command controls reject this option to preserve their semantics.

The default remains disabled. In `replay_feedforward_gate_v1/report.json`,
both a generated can trajectory and a recorded bowl control reproduce every
legacy trajectory field bitwise with the option disabled. The recorded-command
guard also passes. `damping_feedforward_analytic_v1.json` verifies force
cancellation in MuJoCo 3.3.7 and reduces isolated slider tracking RMS from
1.024 to 0.011 mm; this isolated test does not establish grasp success.
Both receipts are under `validation/guidance_noise_2026-09-15/`.

This is a separately labeled controller ablation. Qualify it on all 16 recorded
controls at 0.5 and 0.25 ms before interpreting generated-plan replay outcomes.
Existing controller results and model optimization recipes are unchanged.

The completed fixed feedforward ablation qualifies 10/16 recorded controls,
compared with 11/16 without feedforward; pick asset 8043 loses timestep
qualification. It is not an established controller improvement. Replaying 73
saved seed-8311 plans/ordinary samples gives zero qualified task passes under
this controller. This list includes different objectives/horizons and repeated
scenes, so 73 is a diagnostic count, not independent task trials. Both region
plans retain their failed lifts. The first can physical100 result changes with
controller/timestep, reinforcing numerical sensitivity rather than establishing
success. Receipts: `mujoco_feedforward_controls_v1_summary.json` and
`mujoco_feedforward_generated_v1_summary.json` under the campaign root.
