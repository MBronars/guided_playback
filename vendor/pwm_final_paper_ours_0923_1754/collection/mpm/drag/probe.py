"""Real-GPU command buffer isolation and pre-transfer corruption checks."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import numpy as np
from collection.common import digest,write_once
from collection.lease import gpu_lease
from .cli import wait_idle


def probe(initial,manifest,output,uuid):
    import warp as wp
    from ..physics import CollectionBackend,install_original_walls_in_padded_grid
    from ..contracts import Clock
    from .. import runtime
    from .simulate import DragWorld
    from .grouped import CommandGroup
    m=json.loads(Path(manifest).read_text())
    with np.load(initial,allow_pickle=False) as file:a={k:file[k].copy() for k in file.files}
    def world():
        b=CollectionBackend(a['x'],{k:a[k] for k in runtime.STATIC},m['solver_config'],a['v'],device='cuda:0')
        install_original_walls_in_padded_grid(b)
        return DragWorld(b,Clock(**m['clock']))
    with gpu_lease(uuid):
        wait_idle(uuid);wp.config.kernel_cache_dir=str(Path(output).parent/'probe-cache')
        worlds=[world(),world()];reference=world();group=CommandGroup(worlds)
        n=len(a['x']);zero=np.zeros((n,3),np.float32);off=np.zeros(n,np.int32)
        cmd=zero.copy();cmd[0,0]=.1;on=off.copy();on[0]=1
        group.set_commands([cmd,zero],[on,off])
        before=[(x.numpy().copy(),y.numpy().copy()) for x,y in group.command_buffers]
        bad=zero.copy();bad[0,0]=np.nan
        try:group.set_commands([zero,bad],[off,on])
        except ValueError:pass
        else:raise AssertionError('NaN command not rejected')
        for (left,right),(x,y) in zip(before,group.command_buffers):
            np.testing.assert_array_equal(left,x.numpy());np.testing.assert_array_equal(right,y.numpy())
        group.step();reference.step(reference.clock.substep_dt_s,None,off)
        delta=worlds[1].read()['x'].astype(float)-reference.read()['x'].astype(float)
        rms=float(np.sqrt(np.mean(np.sum(delta**2,axis=-1))))
        if rms>1e-7:raise ValueError('Command in world 0 affected independent world 1')
        healthy=worlds[1].read();x=worlds[0].backend.solver.mpm_state.particle_x
        corrupted=x.numpy();corrupted[0,0]=np.nan
        wp.copy(x,wp.array(corrupted,dtype=wp.vec3,device='cuda:0'))
        try:group.step()
        except FloatingPointError as error:
            if 'nonfinite_state' not in str(error):raise
        else:raise AssertionError('Corrupted state reached transfer')
        assert all(w.steps==1 for w in worlds)
        for key,value in healthy.items():np.testing.assert_array_equal(value,worlds[1].read()[key])
        result=dict(passed=True,invalid_command_all_buffers_unchanged=True,nonfinite_stopped_before_transfer=True,
            healthy_world_unchanged_after_fault=True,independent_world_one_step_position_rms_m=rms,
            initial_sha256=digest(initial),manifest_sha256=digest(manifest),sources=runtime.sources())
        write_once(output,result);print(json.dumps({k:v for k,v in result.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--initial',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--uuid',required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.uuid:raise ValueError('Select the explicit GPU UUID')
    probe(a.initial,a.manifest,a.output,a.uuid)
