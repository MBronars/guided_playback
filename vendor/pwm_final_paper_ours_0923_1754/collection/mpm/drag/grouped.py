"""Command graph on the unchanged passive guarded stepping infrastructure.

This is a new execution path, NOT covered by the passive graph certificate.
"""
import numpy as np
import warp as wp
from ..grouped import Group
from .. import physics


class CommandGroup(Group):
    def __init__(self,worlds,*,block=64,fused=True,max_command_speed=.3):
        if max_command_speed not in (.3,.6):raise ValueError('Unrecognized command speed contract')
        self.max_command_speed=max_command_speed
        super().__init__(worlds,block=block,fused=fused)
        self.command_buffers=[(wp.zeros(w.backend.n,dtype=wp.vec3,device=self.device),
                               wp.zeros(w.backend.n,dtype=int,device=self.device)) for w in worlds]
        # Allocated here, never accepted from callers or aliased with state.
        # Replace the post-transfer graph with the same operations and one
        # pre-P2G velocity override. Guards still surround every native step.
        with wp.ScopedCapture(device=self.device,force_module_load=True) as post:
            for i,w in enumerate(worlds):
                b=w.backend;s=w.solver;cmd,sel=self.command_buffers[i]
                wp.launch(physics._command,dim=b.n,inputs=[s.mpm_state.particle_v,cmd,sel],device=self.device)
                s.p2g2p(0,self.clock.substep_dt_s,device=self.device)
                wp.launch(physics._accumulate,dim=b.n,inputs=[b.master,s.mpm_state.particle_x,
                    s.mpm_state.particle_v,self.clock.substep_dt_s],device=self.device)
                self.guard(w,i);w._particles()
        self.post=post.graph
        for w in worlds:w.solver.time=0.

    def set_commands(self,commands,selections):
        if len(commands)!=len(self.worlds) or len(selections)!=len(self.worlds):
            raise ValueError('One command and selection per world required')
        checked=[]
        for w,cmd,sel in zip(self.worlds,commands,selections):
            cmd=np.asarray(cmd);sel=np.asarray(sel)
            if (cmd.shape!=(w.backend.n,3) or sel.shape!=(w.backend.n,)
                    or not np.isfinite(cmd).all() or not np.isin(sel,[0,1]).all()
                    or np.any(cmd[sel==0]) or np.linalg.norm(cmd,axis=-1).max()>self.max_command_speed+1e-6):
                raise ValueError('Invalid group command; no world advanced')
            checked.append((cmd.astype(np.float32),sel.astype(np.int32)))
        for (cmd,sel),(dst,mask) in zip(checked,self.command_buffers):
            wp.copy(dst,wp.array(cmd,dtype=wp.vec3,device=self.device))
            wp.copy(mask,wp.array(sel,dtype=int,device=self.device))
