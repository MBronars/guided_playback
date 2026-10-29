"""Pinned pre-drag model implementation vs current, identical weights/inputs.

Requires local git history. This tests backward compatibility, not retraining.
"""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import types
import pytest
import torch
from tests.test_mpm_drag_adapter import adapter,sample

REFERENCE='7c2aa5b748f8ed47088fe432e581fcab06fb8c04'


@pytest.mark.parametrize('architecture',['transformer','unet'])
def test_predrag_identical_weights_forward(architecture):
    from models.transformer import ModelConfig,TrajectoryTransformer
    from models.unet_std import UNetWrapper
    root=Path(__file__).resolve().parents[1]
    path='models/transformer.py' if architecture=='transformer' else 'models/unet_std.py'
    result=subprocess.run(['git','show',f'{REFERENCE}:{path}'],cwd=root,text=True,capture_output=True)
    if result.returncode:pytest.skip('Pinned pre-drag git object not present')
    module=types.ModuleType('models._predrag_'+architecture);module.__file__=str(root/path)
    module.__package__='models';sys.modules[module.__name__]=module
    try:
        exec(compile(result.stdout,f'{REFERENCE}:{path}','exec'),module.__dict__)
        cfg=ModelConfig(dim=32,depth=1,heads=4,num_latents=8,k_neighbors=4,dropout=0.,context_len=4,
            unet_actions=True,all_relative=True,unet_points=0,unet_strides='1,2',unet_widths='16,32,64',
            unet_enc_depths='1,1,1',unet_dec_depths='1,1',unet_head_dim=8,unet_t_dim=32,
            unet_l1_by_mat='0:16',unet_pool_k=4,unet_interp_k=3,unet_geo_r_m=.1,unet_fps_random_start=False)
        new=(TrajectoryTransformer(cfg) if architecture=='transformer' else UNetWrapper(cfg)).eval()
        old=(module.TrajectoryTransformer(cfg) if architecture=='transformer' else module.UNetWrapper(cfg)).eval()
        # Open zero-initialized gates/heads so a trivial zero output cannot pass.
        with torch.no_grad():
            for p in new.parameters():
                if not p.any():p.normal_(std=.01)
        old.load_state_dict(new.state_dict(),strict=True)
        b=adapter()._process_sample(sample()).slice(1)
        b=replace(b,action_active=None,context_action_active=None)
        noisy=torch.randn_like(b.actions);t=torch.full((1,16),.5)
        with torch.no_grad():
            torch.manual_seed(8);left=old(batch=b,noisy_future=noisy,timestep=t)
            torch.manual_seed(8);right=new(batch=b,noisy_future=noisy,timestep=t)
        assert right.abs().sum()>0
        torch.testing.assert_close(left,right,atol=0,rtol=0)
    finally:sys.modules.pop(module.__name__,None)
