# GNS baseline — inference only

Roll out a trained GNS checkpoint. **Requires torch and numpy, nothing else.**
No training code, no dataset build, no `torch_geometric`.

```bash
python generate.py \
    --checkpoint  <run>/model-1140000.pt \
    --metadata    <run>/metadata.json \
    --mpm_fresh_test /data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_fresh/test \
    --case 0 --out out.npz
```

Prints the error against ground truth and writes `out.npz` with `predicted`
`[steps, N, 3]` in metres, `ground_truth`, `per_frame_mm`, `particle_type` and
the `context` frames.

Your own scene instead of the release:

```bash
python generate.py --checkpoint ... --metadata ... --input scene.npz --out out.npz
```

`scene.npz` holds `positions` `[T, N, 3]` in metres, `particle_type` `[N]`, and
`material_property` `[N, K]` if the checkpoint uses it (see below). The first
6 frames are context; the rest are predicted autoregressively.

## Checkpoints

| | `metadata.json` | material channels |
|---|---|---|
| `mpm_fresh` | `/data/group_data/katefgroup/datasets/gns_mpm_fresh_npz/metadata.json` | 3: log10 Young's modulus, mat_type, yield stress |
| three-domain (yikai + pickup + rl_push) | `/data/group_data/katefgroup/datasets/gns_robotics3_lazy/metadata.json` | none |

A checkpoint and its `metadata.json` go together: the metadata holds the
normalization the model was trained with, and `generate.py` never refits it.
`--mpm_fresh_test` builds the material channels from the release shards using the
recipe recorded in the metadata's `material_norm`, so it stays consistent.

## What is in this folder

| | |
|---|---|
| `generate.py` | the CLI |
| `learned_simulator.py`, `graph_network.py` | **copies of the GNS model, unchanged** except that their two `torch_geometric` imports point at `pyg_shim` |
| `pyg_shim.py` | `radius_graph` and `MessagePassing` in plain torch |
| `check_against_reference.py` | developer check against the real `torch_geometric` build |

The model itself is not reimplemented — only the two graph utilities are, so
that installing `torch_geometric` + `torch_scatter` + `torch_cluster` (a
compiled, CUDA-version-matched install) is not required.

## How closely it matches

`check_against_reference.py` runs the same checkpoint on the same scenes through
both implementations. On four mpm_fresh episodes at step 1,140,000:

| particles | one step | reference vs its own rerun |
|---:|---:|---:|
| 1,024 | 5.96e-08 m | 5.96e-08 m |
| 2,048 | 1.19e-07 m | 1.19e-07 m |
| 3,072 | 5.96e-08 m | 5.96e-08 m |
| 4,096 | 5.96e-08 m | 5.96e-08 m |

One step agrees to float32 round-off, ~0.1 µm on metre-scale positions.

**A long rollout does not reproduce exactly, and cannot.** Over 43 steps the two
differ by up to 5e-2 m — but the reference differs from *its own rerun* by the
same amount, because CUDA reductions are not bit-reproducible and the rollout is
chaotic. Compare any multi-step number against that baseline, which the check
prints beside it. Run on CPU for a reproducible rollout.

Two details worth knowing if you touch `pyg_shim.py`:

- `radius_graph` must use `cdist(..., compute_mode="donot_use_mm_for_euclid_dist")`.
  The default matrix-multiply form flips pairs sitting near the radius: on one
  1,024-point scene that gave 22 wrong edges out of 28,898 and a 0.6 mm
  single-step error.
- `max_num_neighbors` keeps the *closest* neighbours, where `torch_cluster` keeps
  an implementation-defined subset. They can only differ for a point with more
  than 128 neighbours within the radius; the check prints the observed maximum,
  which is 92 on the densest mpm_fresh scene.
