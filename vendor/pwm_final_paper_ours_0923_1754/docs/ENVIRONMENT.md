# Environment

Everything on this branch was verified in one environment, recorded here as built.
Nothing else was tested. The record is exact; the build steps are the ones that
produced it, not a from-scratch reinstall that this branch has re-run.

| component | version |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.11.0+cu128 (CUDA 12.8, cuDNN 9.19), `torch.cuda.get_arch_list()` = sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| PyTorch3D | 0.7.8, built from source at `facebookresearch/pytorch3d@75ebeeaea0908c5527e7b1e305fbc7681382db47` |
| accelerate | 1.12.0 |
| webdataset | 0.2.111 (pinned; loader parity was established on it) |
| numpy | 1.26.4 (pinned with webdataset) |
| diffusers | 0.35.2 |
| omegaconf | 2.3.1 |

The complete `pip freeze` is in [`env/pwm_blackwell-2026-09-09.freeze.txt`](../env/pwm_blackwell-2026-09-09.freeze.txt).

## Rebuilding it

```bash
conda create -n pwm python=3.10.20 && conda activate pwm
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
# PyTorch3D has no wheel for this torch; build it for every GPU you will use:
export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"   # A100, L40S/A6000, H100, B200, RTX PRO 6000
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
pip install -r env/pwm_blackwell-2026-09-09.freeze.txt   # remaining pins (torch/pytorch3d lines already satisfied)
pip install --no-deps -e .
PYTHONPATH=. python -m pytest tests -q -m 'not integration'   # expect all pass; 8 corpus tests deselected
```

GPUs verified with this environment on this branch: NVIDIA A100-SXM4-40GB and RTX PRO 6000
Blackwell. Numerical receipts are GPU-model specific: the same code gives the same bits
on one GPU model and different bits across models (documented in the parity audit of 2026-09-09, `docs/cleanup/PARITY_AUDIT_2026-09-09.md` in git history, last present at `79a3c84`).

## Data access

The registered corpus lives under `/data/group_data/katefgroup/VLA/3DPCWM` with group
`kfragki2_group` read access (directories `rwxrws---`, shards `rw-rw----`); members of
that group can train without copying anything. The asset-holdout exclusion list and the
cloth sidecar live under the same root. Users outside the group need a copy plus
`data.roots` (see [docs/SUPPORTED.md](SUPPORTED.md#explicit-dataset-roots)).

## Collection

The MPM collection package needs its own environment with Warp 1.10.1
(`pip install -e '.[collection-mpm]'`); training never imports it. See
[collection/README.md](../collection/README.md).
