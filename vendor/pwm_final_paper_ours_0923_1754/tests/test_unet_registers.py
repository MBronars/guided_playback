"""Register rows in the classic UNet core (model.unet_register_mode).
Standalone runner, house style:

    python tests/test_unet_registers.py

CPU, a tiny 3-level ladder on synthetic particles. Covers what the two modes promise:
  1. per_block (historical): every block owns its set, nothing extra rides the stream.
  2. per_level: one learned set per level; a run of blocks carries it from block to block,
     so block i+1 reads what block i wrote into the register rows -- and the LEARNED init
     is what block 0 reads (the control that makes 'carried' non-vacuous).
  3. Registers never cross a seam: the pool and the unpool at every level see exactly the
     particle rows, and a level's set is re-born for its decoder run.
  4. Registers are load-bearing: perturbing a level's init changes the output, and the
     init collects gradient.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.transformer import ModelConfig
from models.unet_std import UNetWrapper

B, T, N = 2, 3, 16
NREG = 4


def _cfg(mode="per_level", **kw):
    base = dict(all_relative=True, context_len=2, num_mat_types=6, pose_native=False,
                unet_points=0, unet_strides="1,2", unet_widths="16,32,64",
                unet_enc_depths="2,2,2", unet_dec_depths="2,2", unet_head_dim=8,
                unet_t_dim=32, unet_l1_by_mat="0:8", unet_pool_mode="pointnet",
                unet_unpool_mode="pointnet", unet_pool_k=4, unet_interp_k=3,
                unet_geo_r_m=0.1, unet_fps_random_start=False,
                unet_register_from_level=1, unet_register_objs=1,
                unet_register_mode=mode, unet_register_count=NREG)
    base.update(kw)
    return ModelConfig(**base)


def _model(mode="per_level", seed=0, wake=True, **kw):
    torch.manual_seed(seed)
    m = UNetWrapper(_cfg(mode, **kw)).model.eval()
    if wake:
        # Every block is an exact identity at construction (zero-init adaLN gates), which
        # would make "the registers were carried" hold trivially. Wake the gates first.
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, torch.nn.Linear) and mod.bias is not None and float(mod.weight.abs().max()) == 0.0:
                    mod.bias.normal_(0.0, 0.2)
                    mod.weight.normal_(0.0, 0.02)
    return m


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, T, N, UNetWrapper.IN_DIM, generator=g)
    pos = torch.rand(B, T, N, 3, generator=g)
    return dict(feats=feats, pos=pos, pos0=pos[:, 0], mat_id=torch.zeros(B, N, dtype=torch.long),
                obj_id=torch.zeros(B, N, dtype=torch.long), t=torch.zeros(B, T), n_context=T - 1)


def test_mode_shapes():
    """per_block keeps its rows inside the blocks; per_level holds one init per level."""
    per_block = _model("per_block")
    assert not hasattr(per_block, "reg_init")
    assert all(b.registers is not None for l in (1, 2) for b in per_block.enc[l])
    assert all(b.registers is None for b in per_block.enc[0])

    per_level = _model("per_level")
    assert all(b.registers is None for lvl in per_level.enc for b in lvl)
    assert set(per_level.reg_init.keys()) == {"1", "2"}
    assert per_level.reg_init["1"].shape == (NREG, 32)
    assert per_level.reg_init["2"].shape == (NREG, 64)
    print("ok  mode shapes: per_block rows live in the blocks, per_level one init per level")


def test_registers_are_carried_through_a_run_and_stripped_at_the_seams():
    """Each block's input carries the register rows the PREVIOUS block returned; the seams
    and the model output never see them."""
    m = _model("per_level")
    seen = {}

    def watch(tag):
        def hook(_mod, args, _kw=None):
            seen.setdefault(tag, []).append(args[0].detach().clone())
        return hook

    handles = []
    for l in (1, 2):
        for i, blk in enumerate(m.enc[l]):
            handles.append(blk.register_forward_pre_hook(watch(f"enc{l}.{i}")))
    for i, blk in enumerate(m.dec[1]):
        handles.append(blk.register_forward_pre_hook(watch(f"dec1.{i}")))
    for l in (0, 1):
        handles.append(m.pools[l].register_forward_pre_hook(watch(f"pool{l}")))
        handles.append(m.unpools[l].register_forward_pre_hook(watch(f"unpool{l}")))
    out = m(**_inputs())
    for h in handles:
        h.remove()

    # rows entering each level's blocks = particle rows + the register set
    n_l = {0: N, 1: 8, 2: 4}
    for l in (1, 2):
        for i in range(2):
            assert seen[f"enc{l}.{i}"][0].shape[2] == n_l[l] + NREG, (l, i)
    # block 0 reads the learned init; block 1 reads what block 0 wrote (carried, not re-born)
    for tag, l in (("enc1", 1), ("enc2", 2), ("dec1", 1)):
        first = seen[f"{tag}.0"][0][:, :, -NREG:]
        second = seen[f"{tag}.1"][0][:, :, -NREG:]
        init = m.reg_init[str(l)][None, None].expand(B, T, NREG, -1)
        assert torch.allclose(first, init, atol=1e-6), f"{tag}: block 0 did not read the learned init"
        assert not torch.allclose(first, second, atol=1e-5), f"{tag}: registers were re-born, not carried"
    # the seams see particle rows only — nothing crosses a level boundary
    assert seen["pool0"][0].shape[2] == N
    assert seen["pool1"][0].shape[2] == n_l[1]
    assert seen["unpool0"][0].shape[2] == N            # skip argument, level 0 resolution
    assert out.shape == (B, T, N, 3)
    print("ok  registers ride a run of blocks, are stripped at every seam, re-born per run")


def test_registers_are_load_bearing():
    """A control for the isolation above: the register rows change the answer, and their
    learned init collects gradient."""
    m = _model("per_level")
    x = _inputs()
    base = m(**x)

    with torch.no_grad():
        m.reg_init["1"].add_(1.0)
    moved = m(**x)
    assert not torch.allclose(base, moved, atol=1e-6), "perturbing a register init changed nothing"
    m.zero_grad()
    m(**x).square().mean().backward()
    for l in ("1", "2"):
        g = m.reg_init[l].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().max() > 0, l
    print("ok  registers are load-bearing (output moves, init collects gradient)")


def test_per_block_stream_is_unchanged():
    """The historical mode is untouched: blocks see particle rows only."""
    m = _model("per_block")
    seen = []
    h = m.enc[1][1].register_forward_pre_hook(lambda _m, args: seen.append(args[0].shape[2]))
    m(**_inputs())
    h.remove()
    assert seen == [8], seen
    print("ok  per_block: the stream between blocks carries no register rows")


if __name__ == "__main__":
    test_mode_shapes()
    test_registers_are_carried_through_a_run_and_stripped_at_the_seams()
    test_registers_are_load_bearing()
    test_per_block_stream_is_unchanged()
    print("ALL OK")
