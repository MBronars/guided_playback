"""Continue the qualified floating-gripper diagnostic without resetting physics.

MuJoCo is optional and imported only when constructing this replay. The executed
interface contains hand targets and finger commands; it cannot prescribe object
motion. This is still an independent-engine diagnostic, not robot execution.
"""
import hashlib
from pathlib import Path

import numpy as np


def planned_finger_commands(rotations, translations, calibration, collision, *,
                            preload=.006, contact_tolerance=.002):
    """Same fixed mesh-preload rule as the standalone generated-plan replay.

    Poses are in body order [object, hand, finger0, finger1], including the
    observed anchor. No recorded future command is an input.
    """
    R, t = np.asarray(rotations, dtype=np.float64), np.asarray(translations, dtype=np.float64)
    if (R.ndim != 4 or R.shape[1:] != (4, 3, 3) or t.shape != R.shape[:-1]
            or not np.isfinite(R).all() or not np.isfinite(t).all()
            or not np.isfinite(preload) or preload < 0
            or not np.isfinite(contact_tolerance) or contact_tolerance <= 0):
        raise ValueError('Invalid generated finger-command inputs')
    c = {k: np.asarray(v, dtype=np.float64) for k, v in calibration.items()}
    local = np.einsum('tji,tfj->tfi', R[:, 1], t[:, 2:]-t[:, 1, None])
    achieved = c['q_anchor'] + ((local-c['finger_anchor'])*c['axis']).sum(-1)
    gaps = np.array([collision.query(rr, tt)[0][1:] for rr, tt in zip(R, t)])
    contact = gaps.max(-1) <= contact_tolerance
    return dict(achieved=achieved, commanded=achieved-preload*contact[:, None],
                planned_contact=contact)


class PersistentPickupReplay:
    def __init__(self, directory, *, frame=0):
        import mujoco
        self.mujoco = mujoco
        directory = Path(directory)
        with np.load(directory/'integration_states.npz') as saved:
            if str(saved['mujoco_version']) != mujoco.__version__:
                raise ValueError('Physics continuation requires the same MuJoCo version')
            model_path = directory/'scene.mjb'
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != str(saved['model_sha256']):
                raise ValueError('Compiled replay model does not match the saved integration state')
            if not isinstance(frame, int) or not 0 <= frame < len(saved['states']):
                raise ValueError('Invalid replay checkpoint frame')
            self.signature = int(saved['signature'])
            if self.signature != int(mujoco.mjtState.mjSTATE_INTEGRATION):
                raise ValueError('Replay checkpoint must contain the complete integration state')
            self.model = mujoco.MjModel.from_binary_path(str(model_path))
            self.data = mujoco.MjData(self.model)
            mujoco.mj_setState(self.model, self.data, saved['states'][frame], self.signature)
        self.ids = [self.model.body(name).id for name in ('object', 'hand', 'finger0', 'finger1')]
        self.jq = [self.model.joint(f'finger_q{i}').qposadr[0] for i in range(2)]
        if self.model.nu != 2 or self.model.nmocap != 1:
            raise ValueError('Expected one floating hand target and two finger motors')
        # Populate geometry without solving forces again or changing warmstart.
        mujoco.mj_fwdPosition(self.model, self.data)

    def state(self):
        state = np.empty(self.mujoco.mj_stateSize(self.model, self.signature))
        self.mujoco.mj_getState(self.model, self.data, state, self.signature)
        return state

    def observation(self):
        return dict(R=self.data.xmat[self.ids].reshape(4, 3, 3).copy(),
                    T=self.data.xpos[self.ids].copy(), q=self.data.qpos[self.jq].copy(),
                    time=float(self.data.time))

    def advance(self, hand_rotations, hand_translations, finger_targets, *, dt, joint_limits):
        """Apply a sequence of interpolated commands, retaining the simulation state.

        Arrays include the command at the segment's start and at least one new
        frame. A plan may start from the latest achieved pose; unchanged command
        segments can also be replayed exactly for continuation qualification.
        """
        from scipy.spatial.transform import Rotation, Slerp
        R, t, q = [np.asarray(v, dtype=np.float64) for v in
                   (hand_rotations, hand_translations, finger_targets)]
        limits = np.asarray(joint_limits, dtype=np.float64)
        if (R.ndim != 3 or R.shape[1:] != (3, 3) or len(R) < 2
                or t.shape != (len(R), 3) or q.shape != (len(R), 2)
                or limits.shape != (2, 2) or not (limits[:, 0] < limits[:, 1]).all()
                or not all(np.isfinite(v).all() for v in (R, t, q, limits))
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError('Invalid persistent replay command sequence')
        nsub = round(dt/self.model.opt.timestep)
        if nsub < 1 or abs(nsub*self.model.opt.timestep-dt) > 1e-9:
            raise ValueError('Command interval must be divisible by the integration step')
        observations, states = [self.observation()], [self.state()]
        for i in range(len(R)-1):
            slerp = Slerp([0, 1], Rotation.from_matrix(R[[i, i+1]]))
            for sub in range(nsub):
                fraction = (sub+1)/nsub
                self.data.mocap_pos[0] = (1-fraction)*t[i]+fraction*t[i+1]
                self.data.mocap_quat[0] = slerp(fraction).as_quat()[[3, 0, 1, 2]]
                self.data.ctrl[:] = np.clip((1-fraction)*q[i]+fraction*q[i+1], limits[:, 0], limits[:, 1])
                self.mujoco.mj_step(self.model, self.data)
            self.mujoco.mj_forward(self.model, self.data)
            observations.append(self.observation()); states.append(self.state())
        return dict(R=np.array([o['R'] for o in observations]), T=np.array([o['T'] for o in observations]),
                    q=np.array([o['q'] for o in observations]), time=np.array([o['time'] for o in observations]),
                    integration_states=np.array(states), warning_counts=self.data.warning.number.copy())
