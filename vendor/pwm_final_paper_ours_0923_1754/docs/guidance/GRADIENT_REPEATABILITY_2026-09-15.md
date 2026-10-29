# GPU gradient repeatability in noise optimization

Saved best-through6 prefixes of the100-update runs differ slightly from the
independent6-update runs, despite identical initial draws. On the first pickup
development scene, seed8311, maximum position differences were1.15µm for goal
optimization and0.48µm for physical optimization. Noise updates already differ
after the first backward pass. This is not a bitwise reproduction claim.

`validation/guidance_noise_2026-09-15/gpu_gradient_repeat_v1/report.json` isolates
repeatability on the RTX PRO6000, PyTorch2.11.0+cu128, existing clean-context
unconditional bidirectional checkpoint. Three H16 rollouts with the same initial
noise and replayed sampler RNG have identical forward outputs. Repeated terminal
X gradients differ by at most2.33e-10 per entry (relative norm difference1.2e-7).
With deterministic algorithms enabled, those gradients are bitwise identical.
Model weights stay frozen. This is one scene and one gradient functional.

The actual UNet `gather_neighbors` helper also reproduces nondeterministic CUDA
backward accumulation: four repeats differ by up to1.53e-5 in a synthetic
overlapping-neighbor case, versus zero with deterministic algorithms. The KNN
routine supplies indices here; its custom distance backward is not used by this
path. Deterministic mode removes the observed repeat variation, but the probe
does not isolate every contributing backward kernel in the full model.

`guidance/run_unconditional.py --deterministic` requests deterministic PyTorch
algorithms and sets a default `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization. Unsupported nondeterministic operations raise. Both settings
are recorded in `protocol.json`; the default remains unchanged for the pinned
campaign. This does not guarantee identical results across hardware, releases,
different precision or different code. The
[PyTorch2.11 deterministic-algorithms reference](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html)
documents deterministic CUDA gather backward and cuBLAS workspace requirements.

The probe ran alongside an existing GPU worker, so its44.6/48.5second timings
are not a controlled performance comparison. No timing-overhead claim follows.

`deterministic_cli_repeat_v1/report.json` also passes: two independent CLI runs
of one H16 physical-mesh AL update produce bitwise-identical initial and selected
positions, rotations, translations and noise. The CLI itself initializes and
records the deterministic/cuBLAS settings. This validates the exposed switch,
still within the pinned checkpoint, fixture and environment.

The later six-versus24-update can feedback comparison likewise has bitwise-equal
initial forward trajectories and fresh-control executions, but not bitwise-equal
optimizer traces. Across the first six trace rows of its first replan, maximum
loss difference is1.35e-7 and target-distance difference is0.000090mm. The stricter
bitwise-prefix assertion fails; `mpc_can_budget_prefix_audit_v1.json` records the
measured differences rather than converting this into a bitwise pass. This is
consistent with the previously isolated backward repeatability limitation;
the feedback run itself did not isolate its contributing kernels again.
