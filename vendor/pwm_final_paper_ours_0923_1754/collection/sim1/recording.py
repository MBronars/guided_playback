"""Isolated actual-state recorder and integer-index USD clock.

New raw contract: S states, S-1 outgoing commands, initial state explicitly saved.
Actual particle/link/joint states are not interchangeable with command targets.
No integration into live SIM1 is performed by importing this module.
"""
from __future__ import annotations
import numpy as np
from .contracts import finite_array, integer_state_times, unit_quaternions


class StateRecorder:
    def __init__(self, fps, n_particles, n_bodies, n_joints, max_states=2000, max_bytes=1024**3):
        integer_state_times(np.array([0],dtype=np.int64),fps)
        for n in (n_particles,n_bodies,n_joints,max_states,max_bytes):
            if type(n) is not int or n <= 0: raise ValueError('Positive explicit dimensions/budgets required')
        self.fps=float(fps); self.shape=(n_particles,n_bodies,n_joints)
        self.max_states=max_states
        estimate=max_states*(6*n_particles+7*n_bodies+n_joints)*4
        estimate+=(max_states-1)*(n_joints+2)*4
        estimate=2*estimate  # retained frames plus stacking into output arrays
        if estimate>max_bytes: raise ValueError('Recorder exceeds explicit RAM budget; use smaller clips/chunks')
        self.estimated_bytes=estimate
        self.states=[]; self.commands=[]

    def append_initial(self, *, particle_q, particle_qd, body_q, joint_q):
        if self.states: raise ValueError('Initial state already recorded')
        self._state(0,particle_q,particle_qd,body_q,joint_q)

    def append_transition(self, transition_index, *, command_joint_q, command_openness,
                          particle_q, particle_qd, body_q, joint_q):
        if not self.states or type(transition_index) is not int or transition_index!=len(self.commands):
            raise ValueError('Record initial state then every outgoing transition exactly once')
        command=finite_array(command_joint_q,'command_joint_q',(self.shape[2],))
        openness=finite_array(command_openness,'command_openness',(2,))
        if np.any((openness<0)|(openness>1)): raise ValueError('Command openness outside [0,1]')
        command,openness=self._float32(command),self._float32(openness)
        self._state(transition_index+1,particle_q,particle_qd,body_q,joint_q)
        self.commands.append((command,openness))

    def _state(self,index,q,v,b,j):
        if index!=len(self.states) or len(self.states)>=self.max_states:
            raise ValueError('State sequence/index/budget violated')
        N,K,J=self.shape
        q=finite_array(q,'particle_q',(N,3)); v=finite_array(v,'particle_qd',(N,3))
        b=finite_array(b,'body_q',(K,7)); j=finite_array(j,'actual_joint_q',(J,))
        unit_quaternions(b[:,3:],'body_q')
        self.states.append(tuple(self._float32(a) for a in (q,v,b,j)))

    @staticmethod
    def _float32(value):
        with np.errstate(over='ignore'):
            converted=value.astype(np.float32,copy=True)
        if not np.isfinite(converted).all():
            raise ValueError('Raw state/command cannot be represented as finite float32')
        return converted

    def arrays(self, expected_transitions):
        if type(expected_transitions) is not int or expected_transitions < 1:
            raise ValueError('Explicit positive planned transition count required')
        if len(self.commands)!=expected_transitions or len(self.states)!=expected_transitions+1:
            raise ValueError('Incomplete replay must not be exported as a completed episode')
        result={k:np.stack([s[i] for s in self.states]) for i,k in enumerate(
            ('particle_q','particle_qd','body_q','actual_joint_q'))}
        result.update(command_joint_q=np.stack([c[0] for c in self.commands]),
                      command_openness=np.stack([c[1] for c in self.commands]),
                      state_index=np.arange(len(self.states),dtype=np.int64),
                      transition_index=np.arange(len(self.commands),dtype=np.int64),
                      fps=np.array(self.fps,dtype=np.float64))
        result['state_time_s']=integer_state_times(result['state_index'],self.fps)
        return result


def integer_frame_usd_class():
    """Lazy factory: no Newton/Warp import for array-contract tests.

    Uses the actually exported ViewerUSD and retains its rendering implementation.
    Only the clock interface changes: callers must provide the state index.
    """
    import newton
    class IntegerFrameUSD(newton.viewer.ViewerUSD):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.stage.SetTimeCodesPerSecond(self.fps)
            self._last_state_index=-1

        def begin_state(self,state_index):
            if type(state_index) is not int or state_index!=self._last_state_index+1:
                raise ValueError('USD requires complete consecutive state indices starting at zero')
            super().begin_frame(state_index/self.fps)
            # Do not let int(time * fps) truncate exact integer state identity.
            self._frame_index=state_index
            self.stage.SetEndTimeCode(state_index)
            self._last_state_index=state_index

        def begin_frame(self,time):
            raise RuntimeError('Use begin_state(integer_index); float-time recording is ambiguous')
    return IntegerFrameUSD
