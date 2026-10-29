# Finger travel in the independent replay

The baseline floating-hand MuJoCo replay inherits the simulator's soft joint
limit parameters. It can exceed the Panda finger travel range despite clipping
motor targets. The first generated can region plan exceeds the upper limit by
3.28 mm during its prefix; later feedback cannot undo that violation.

`replay_pickup_mujoco.py --joint-limit-time` optionally changes only the finger
limit time constant. Omission preserves the original defaults and all legacy
arrays remain bitwise equal on generated-can and recorded-place fixtures.
The value must be finite and at least twice the integration step, following
[MuJoCo's solver parameter guidance](https://mujoco.readthedocs.io/en/3.3.7/modeling.html#solver-parameters).
The feedback runner rejects altered limit dynamics until a separate experiment
qualifies them. Existing queued feedback trials use the original controller.

An isolated 0.1 kg slider with 20 N outward load, the same kp2000/kv100 servo,
and a 40 mm travel limit demonstrates the difference at 0.5/0.25 ms steps:

| Limit time | Maximum transient overshoot, mm | Recorded controls qualified |
| --- | --- | --- |
| Default 20 ms | 2.858 / 2.857 | 11/16 |
| 2 ms | 0.181 / 0.182 | 10/16 |
| 1 ms | 0.065 / 0.081 | 11/16 |

The original 2 ms isolated check failed its declared 0.1 mm maximum-overshoot
criterion; its record is retained. A second fixed candidate, 1 ms, is the
minimum allowed by the coarser integration step and passes that criterion.
All 16 recorded controls were replayed at both steps for each candidate before
testing generated plans. The 1 ms setting preserves exactly the same 11 scenes
under the original pose/timestep qualification, rather than changing the
qualification threshold. Finger overshoot is reported separately: the remaining
unqualified asset6778 still exceeds the 0.1 mm allowance.

Receipts are `joint_limit_response_probe_v1.json`,
`joint_limit_response_probe_v2.json`, `joint_limit_response_gate_v2.json`,
`replay_limit_default_gate_v1/report.json`, and
`mujoco_limit_controls_v1_summary.json` / `mujoco_limit_controls_v2_summary.json`
under `validation/guidance_noise_2026-09-15`. This qualifies a simulator ablation,
not a model improvement or robot result. It does not alter any original replay,
model-generated trajectory, material, collision geometry, or task criterion.

The frozen generated-plan comparison contains 86 diagnostics, each at both
steps: selected seed8311 outputs plus one ordinary sample per scene. Every
desired hand rotation/translation and commanded finger target is bitwise equal
between controllers. Both controllers yield zero qualified complete-task passes.
The first region can lifts -1.24/-0.04 mm with firmer limits; the second region
pick moves downward 23.57/24.39 mm. The 1 ms setting reduces over-travel but does
not remove it in all generated motions. The two region failures therefore are
not explained solely by the default limit softness. These are correlated
diagnostics, not 86 independent benchmark trials.

`scripts/dev/render_replay_limit_ablation.py` plots recorded-control over-travel
and generated object height/joint travel, with exact paired-command checks.
The main report links `replay_limit_report_v1/index.html`; its summary retains
every paired result and the separate recorded-control qualification.
