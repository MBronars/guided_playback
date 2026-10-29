# PointWorld inference

Run one of our trained **PointWorld** checkpoints from this repo. Inference only — no training,
no evaluator, and nothing here imports `pwm_final_paper`.

PointWorld ("PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation", NVlabs)
is the baseline the PWM runs are compared against. This folder is a vendored inference-only copy
of it: 32 files, Apache-2.0. `PROVENANCE.md` records the upstream commit and our changes.

## What a checkpoint does

In goes the scene at one anchor frame and the actuator's whole trajectory; out come the scene's
next H frames. **The actuator is given** — the model reads its trajectory and never predicts it.
The scene's future is not an input.

```
scene0  (Ns, 3)        metres, the anchor frame
robot   (1 + H, Nr, 3) metres, anchor frame then every given frame
    ->  (H, Ns, 3)     metres, row for row with scene0
```

H comes from the checkpoint itself (its model contract), so the same command runs a 10-frame and
a 16-frame model.

## Environment

**Use a separate env.** PointWorld's PTv3 backbone needs four compiled packages, two of which
(`torch_scatter`, `flash_attn`) are torch C++ extensions locked to an exact torch build, so they
cannot load under this repo's env — and installing them carelessly lets pip swap torch out, which
is how an env breaks. They are not optional: the model has **56 `spconv.SubMConv3d` layers holding
284.5M of its 409.8M parameters**, attention calls `flash_attn_varlen_qkvpacked_func` with no
fallback, and pooling makes 7 `torch_scatter` calls.

The env we train and run in, exactly:

```bash
conda create -n pointworld-infer python=3.10 -y && conda activate pointworld-infer
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124==2.3.8 timm==1.0.19 scipy==1.11.2 numpy==1.26.4 \
            opencv-python==4.11.0.86 pyyaml==6.0.2 addict==2.4.0 einops==0.8.2
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install flash_attn==2.7.4.post1 --no-build-isolation
```

Match the CUDA build to your machine. Off the wheel index, `torch_scatter` and `flash_attn` build
from source (needs `nvcc`, tens of minutes). `spconv` ships its own kernels and is the least
torch-sensitive. A GPU is required: flash-attention has no CPU path.

If a separate env is impossible, the packages can go in a directory instead:
`pip install --target <dir> --no-deps <package>` and put `<dir>` on `PYTHONPATH`. `--no-deps` is
what stops pip pulling another torch, and deleting the directory undoes it — but the wheels still
have to match that env's torch.

## Run it

```bash
python baselines/pointworld/infer.py \
    --checkpoint /data/group_data/katefgroup/runs_pw_mbronars/three_pointworld_ga4_cont/matched_step_060000.pt \
    --stats three_domain_h10 --selftest --report
```

`--selftest` predicts a synthetic window and prints the per-frame motion: the quickest check that
the environment works. On a real window:

```bash
python baselines/pointworld/infer.py --checkpoint <ckpt> --stats <ruler> \
    --window window.npz --out pred.npz
```

`window.npz` holds `scene` (Ns, 3) and `robot` (1 + H, Nr, 3) in metres; `pred.npz` gets `pred`
(H, Ns, 3). As a library:

```python
import sys; sys.path.insert(0, "baselines/pointworld")
from infer import PointWorldInference
model = PointWorldInference(checkpoint, "baselines/pointworld/norm_stats/three_domain_h10", device="cuda")
pred = model.predict(scene0, robot)          # (H, Ns, 3) metres
```

## Checkpoints and their rulers

A checkpoint must be run with the normalization statistics it trained with — the model divides by
them. Those are bundled here in `norm_stats/`, so only the checkpoint has to be fetched. Passing
the wrong one gives wrong numbers without any error.

| checkpoint (shared, on group storage) | `--stats` | horizon |
|---|---|---|
| `runs_pw_mbronars/three_pointworld_ga4_cont/matched_step_*.pt` (and `three_pointworld/` for the first two) — pickup + rl_push + yikai_small | `three_domain_h10` | 10 |
| `runs_pw_mbronars/pointworld_dex/matched_step_*.pt` — dex_v2_filtered2 | `dex_h16` | 16 |
| `runs_pw_mbronars/pointworld_two/matched_step_*.pt` — pickup + yikai_small_fs0 | `two_domain_h16` | 16 |

Each checkpoint is 4.9 GB. `--norm_stats <folder>` takes any folder with a `norm_stats.json`
instead of a bundled name.

## Two properties worth knowing

- **Rows come back in the order they went in**, but the model voxel-downsamples the scene at
  `grid_size` (1 mm in our runs), so points sharing a voxel share a trajectory. `--report` prints
  how many rows that affected and the largest mapping distance.
- **The model is stochastic at inference.** PTv3 reshuffles its serialization order on every
  forward pass, eval included. Seed torch before a call for reproducible output; on our data,
  unseeded repeats differ by a few millimetres at most.

## Checked on 2026-09-23

From this repo's root, in the env above, on one A100:

| checkpoint | horizon | result |
|---|---|---|
| `three_pointworld_ga4_cont/matched_step_060000.pt` | 10 | (10, 512, 3), motion from the anchor 2.8 → 20.9 mm |
| `pointworld_dex/matched_step_010000.pt` | 16 | (16, 512, 3) |

Also checked that importing this repo's `utils` package in the same process still resolves to
`pwm_final_paper/utils/` — PointWorld's own top-level `utils.py` is `pw_utils.py` here.

## If you cannot build the environment

The evaluation protocol's receipts already carry predictions: every headline case set writes
`<case set>.predictions/<k>.npz` with predicted positions in metres, their row indices, and which
rows the model predicted. That is enough to score or visualize our results without running
anything — see `EVAL.md` and the PointWorld protocol runner that produced them.
