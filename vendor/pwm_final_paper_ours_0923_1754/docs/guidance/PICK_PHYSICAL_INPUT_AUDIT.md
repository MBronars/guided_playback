# Physical inputs available to pickup guidance

Audit date: 2026-09-14. This describes the existing pickup adapter and backbones;
it does not establish a cause of the current phantom-motion failures.

| Information | Current path | Implication |
|---|---|---|
| Young's modulus, Poisson ratio, density | `UnifiedPickupDataset` takes material columns 0–2. Both backbones embed `log10(E), nu, rho/1000`, with optional normalization. | Density sensitivity can be tested directly without changing checkpoint structure. |
| Friction | The collector exports material column 3; `TrajectoryBatch` and the pickup adapter do not expose it to either backbone. | Changing friction requires a new input contract and trained weights, or must be evaluated as unobserved physical variation in replay. |
| Observed hand and finger geometry/motion | Rigid particle templates, observed poses, temporal pose deltas and articulated joint geometry. | Current opening is visible through geometry; the model is not missing all opening information. |
| Absolute joint coordinates and limits | Raw shards contain observed joint positions/velocities, but the core batch has no explicit absolute-q or limit fields. Guidance calibration separately recovers q and reads limits from the URDF. | Joint geometry preserves the articulation; it does not enforce travel or speed limits. |
| Commanded joint targets | Stored in source shards. The model's action interface uses achieved motion, and this unconditional checkpoint hides future actions. | A generated achieved finger position is not automatically the command needed to sustain a grip force. Qualify the command conversion during replay. |
| Support geometry | No explicit table mesh or plane in the pickup batch. The guidance diagnostic infers a support plane from observed object geometry. | The present fixed-environment setup can learn implicit support; a changing boundary needs an explicit contract and separate experiments. |

Code: [pickup adapter](../../data/unified_dataset.py:350),
[batch fields](../../data/batch.py:58),
[UNet physics features](../../models/unet_std.py:1477),
[Transformer embedding](../../models/transformer.py:460),
[joint representation](../../utils/pose_native.py:85), and
[replay preparation](../../scripts/dev/prepare_pick_replay.py:35).
The collector documents and writes `(E, nu, rho, friction)` in
`/data/user_data/robertre/RLSkills_IsaacLab/scripts/collect_pickup_3dpcwm.py`.
Those exports use asset metadata; they are not a saved query of every live
PhysX material setting.

The four development fixtures have recorded object densities 789.90, 860.34,
1555.51 and 1634.06 kg/m³. Their exported friction value is 1.0 in each case;
the regression jar's is 0.9. This small inspection does not measure the full
training distribution. Raw source identities and values are recorded in
`validation/guidance_pick_pilot_2026-09-14/material_channel_audit.json`.

The current sensitivity packet varies the planner's density estimate and
coherently biases observed object poses while retaining the true reference and
fixed goal. It measures input sensitivity. Robust execution, uncertainty
calibration and recovery under missing observations require additional tests.
Adding friction, absolute joint state or boundary inputs remains a versioned
training experiment after the current generation and replay checks.
