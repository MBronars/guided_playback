# Contact geometry and executed forces

Artifacts live under `validation/guidance_noise_2026-09-15/`.
Start with `grasp_execution_report_v1/index.html` and the feedback movies linked
there. These diagnostics do not change the active optimizer, controller, or task
criteria.

`grasp_execution_report_v1/optimized_prediction_vs_execution.gif` directly pairs
the selected can continuation's predicted body motion with its independent
0.5 ms execution. Both share the same executed prefix. The model predicts a
201.3 mm final lift; execution ends at -1.64 mm after a transient lift and drop.
The predicted continuation itself narrowly misses the contact screen by 0.047 mm;
neither panel is a complete task success.

The checkpoint remains the EMA unconditional bidirectional pickup model whose
SHA256 is `99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba`.
The scene set and registered exclusions are unchanged (`scenes_v2.json`).
Generated base plans use seed 8311; the new feedback draw uses seed 6381.
The recorded references are dataset trajectories, not generated successes.

## Closing direction is not the explanation for the first two region failures

`audit_grasp_closing_direction.py` freezes the completed diagnostic set when
invoked: all 16 recorded references, the first two region24 plans, and the first
pick/place physical100 plans. It measures signed collision-component gap versus
each finger's calibrated joint coordinate, using central differences of 0.05 and
0.1 mm. Positive derivative means closing locally reduces the gap. A nearby
observation has gap <=2 mm; the two estimates must differ by <=0.05, and direction
is classified using magnitude 0.1. These are diagnostic bins, not new acceptance
thresholds.

Analytic outside squeeze, inside squeeze requiring opening, and tangential
motion cases pass. For the first can region plan, all 40 nearby finger/frame
observations have the expected closing direction. For asset 8026, 33 of 34 do;
one changes with probe scale. Recorded picks have the same general pattern.
This local measurement does not establish force closure or grasp stability.
It does not justify another guidance penalty for these failures.

The first audit attempt read the wrong exported translation field before
processing generated examples; it is preserved as `grasp_closing_direction_v1`.
The corrected complete audit is `grasp_closing_direction_v2`, with source hashes
and all per-frame arrays.

## Greater lift can accompany a worse collision

For `g0v9_pick_development_00_asset_5624_feedback_observed24_f16_h20`, the original
plan executes 16 frames before a fresh 20-frame continuation. Both end at the
same original 36-frame deadline. Standard noise optimization lowers its selected
objective from 0.2469209 to 0.0007945 (best iteration 18 of 24).

The ordinary fresh continuation reaches final lift 144.26 / 149.77 mm at physics
steps 0.5 / 0.25 ms. The optimized continuation finishes at -1.64 / -0.34 mm.
However, the fresh sample first crushes the object between the hand and table.
Peak vertical contact forces are about 32.6 kN, compared with object weight
3.43 N. The stiff floating-hand controller is capable of unphysical loads; this
is not a real robot force prediction. The optimized continuation removes the
extreme crushing, briefly lifts, then drops the object. All complete task checks
remain failures. Final height alone cannot rank overall physical quality.

`audit_replay_contact_forces.py` records `mj_contactForce` after each integration
step. It transforms contact-frame forces to world coordinates with the sign
applied to the passive object. Their sum agrees with the object's translational
`qfrc_constraint` at every substep. Logging preserves positions, joints, and full
integration states bitwise on 16 recorded plans x 2 steps plus the three feedback
variants x 2 steps: 38 checks. Recorded-plan replay uses the previously qualified
fixed mesh-preload controller. Forces correspond to the solver evaluation before
position integration; plotted times are step ends.

The force results are in `replay_contact_forces_v1/report.json`. No force-based
success threshold or force-based optimization cost was introduced. The existing
penetration, joint-limit, and complete-task checks already reject these cases.

API reference: [MuJoCo 3.3.7 contact-force extraction](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-contactforce).

## Contact timing across all eight completed lift-workspace plans

`scripts/dev/audit_region_contact_timing.py` reads the completed selected region24
plans and both existing executions. It compares object height, nearest-link gap,
farther-finger gap, predicted opening, preload motor target and achieved opening.
The two replay steps must have identical desired poses and commands; independently
queried mesh gaps reproduce the saved controller proximity gate bitwise on all
eight scenes. The audit changes no dynamics or task threshold. Results and plots:
`validation/guidance_noise_2026-09-15/region_contact_timing_v1/index.html`.

Five of eight model plans pass the region plus geometry screen; none passes the
qualified execution screen. Reserved asset8270 is a particularly clear example:
the model predicts211.96mm lift, while both physics steps leave the object within
0.002mm of its initial height. Its recorded controller qualifies and its generated
replay is timestep-stable. The hand tracks within1.51mm and0.73degrees at0.5ms,
but both fingers never simultaneously enter the2mm proximity tolerance in the
saved executed frames. The model predicts two-finger proximity from frame26,
by which time its object has already risen39.21mm. Earlier single-link proximity
starts at frame19 in both prediction and execution. This is a concrete mismatch
in predicted interaction, not evidence that the object rose before every possible
contact or that closing in the wrong direction is the cause.

Reserved asset6839 instead predicts51.24mm rise before its first saved frame with
any hand/finger surface within2mm. Its complete model screen already fails held
relative-motion checks. Seven other predictions have0–3.19mm rise by that measure,
so this is not a universal explanation of failure. Measurements are sampled at
50ms output frames: they neither exclude inter-frame contact nor measure forces.
One finger or the palm may move an object; absence of a two-finger grasp alone
does not establish physically impossible motion. Source hashes and every scene,
including failures and unqualified controller cases, remain in the report.

`scripts/dev/render_qualified_control_comparison.py` makes a four-panel movie of
asset8270: the recorded trajectory and its replay above the generated model plan
and its replay. Both right panels use the same fixed preload controller and0.5ms
step; only gripper motion is commanded. The recorded trajectory lifts136.75mm,
and its replay lifts131.82mm with6.32mm final object-position disagreement. The
generated replay barely lifts. All four panels share the observed anchor and
coordinate limits. `qualified_control_report_v1/index.html` retains the source
hashes and limitations. This reference helps rule out a blanket inability to
lift the object in this replay setup; it does not prove that achieved-opening
to motor-command conversion is equally faithful for every generated grasp.
The recorded control is not a generated success or the requested held-workspace
target. The GIF and its final frame were visually inspected.
