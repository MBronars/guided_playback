"""Geometry-targeted folding PREVIEWS, not certified manipulation policies.

Uses the actual source-rest robot for IK/FK, never resets cloth to the targeting
snapshot, and emits the existing numeric replay contract. All phases, including
return, use smooth bounded commands. Numerical replay is still required.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from collection.common import digest, write_once

FAMILIES = ('left-inward', 'right-diagonal', 'two-hand-half')


def select_targets(points, family):
    """Choose observed surface vertices, not empty bounding-box corners."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Finite nonempty XYZ targeting snapshot required')
    if family not in FAMILIES:
        raise ValueError('Unknown fold family')
    center = np.median(p[:, :2], axis=0)
    # Stay in the robot's near, reachable half of the garment. This deliberately
    # differs from selecting far hem corners outside this robot's workspace.
    x = float(np.clip(np.quantile(p[:, 0], .12), .48, .56))
    half = float(np.clip(np.quantile(np.abs(p[:, 1] - center[1]), .7), .13, .25))
    active = [0] if family == 'left-inward' else [1] if family == 'right-diagonal' else [0, 1]
    targets = {}
    for side in active:
        guess = np.array([x, center[1] + (1 if side == 0 else -1) * half])
        d = np.linalg.norm(p[:, :2] - guess, axis=1)
        nearby = np.flatnonzero(d <= d.min() + .015)
        # Highest available surface in this local patch, with explicit vertex ID.
        vid = int(nearby[np.argmax(p[nearby, 2])])
        grasp = p[vid].copy()
        destination = grasp.copy()
        if family == 'two-hand-half':
            destination[0] += .22
        elif family == 'left-inward':
            destination[1] = center[1] - .02
        else:
            destination[0] += .13
            destination[1] = center[1] + .02
        destination[2] = max(.815, grasp[2]) + .015
        targets[side] = dict(vertex=vid, grasp=grasp, destination=destination)
    if len(targets) == 2 and abs(targets[0]['grasp'][1] - targets[1]['grasp'][1]) < .20:
        raise ValueError('Insufficient two-hand separation')
    return targets


def quintic(u):
    u = np.asarray(u)
    return 10 * u**3 - 15 * u**4 + 6 * u**5


class RobotIK:
    def __init__(self, env, model, home):
        import warp as wp
        import newton
        self.wp, self.newton = wp, newton
        self.model, self.state = model, model.state()
        self.q = wp.array(home, dtype=wp.float32, device='cpu')
        self.qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device='cpu')
        self.lower, self.upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        self.ee = [env.lee_index, env.ree_index]
        self.arms = [np.array(env.controllable_joint_indices[:6]), np.array(env.controllable_joint_indices[6:])]
        self.jaws = [np.array(env.left_gripper_joint_indices), np.array(env.right_gripper_joint_indices)]

    def fk(self, q):
        self.q.assign(np.asarray(q, dtype=np.float32))
        self.newton.eval_fk(self.model, self.q, self.qd, self.state)
        return self.state.body_q.numpy()[self.ee].astype(float)

    def solve(self, q, side, target):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        ids = self.arms[side]
        desired = Rotation.from_quat(target[3:])
        def residual(x):
            command = q.copy(); command[ids] = x
            pose = self.fk(command)[side]
            return np.r_[10 * (pose[:3] - target[:3]),
                         (desired.inv() * Rotation.from_quat(pose[3:])).as_rotvec()]
        def jacobian(x):
            h=1e-3
            return np.stack([(residual(x+np.eye(len(x))[i]*h)-residual(x-np.eye(len(x))[i]*h))/(2*h)
                             for i in range(len(x))],axis=1)
        # FK is float32: explicit finite-difference size avoids zero numerical
        # Jacobian columns at SciPy's default sqrt(double-epsilon) step size.
        result = least_squares(residual, np.clip(q[ids], self.lower[ids]+1e-6, self.upper[ids]-1e-6),
            bounds=(self.lower[ids], self.upper[ids]), jac=jacobian, max_nfev=120,
            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        out = q.copy(); out[ids] = result.x
        error = residual(result.x)
        if np.linalg.norm(error[:3]) > .003 or np.linalg.norm(error[3:]) > .003:
            raise ValueError(f'IK unreachable side={side}: position={np.linalg.norm(error[:3])/10:.6f} m, rotation={np.linalg.norm(error[3:]):.6f} rad')
        return out


def make_plan(robot, home, targets, *, pad_offset):
    """Smooth full home-to-home transit; other arm remains at home in single folds."""
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp

    active = sorted(targets)
    home = np.clip(home.astype(float), robot.lower, robot.upper)
    home_poses = robot.fk(home)
    poses = home_poses.copy()
    orientations = {}
    for side in active:
        candidates, failures = [], []
        for yaw in (0., np.pi/2, np.pi, -np.pi/2):
            x = np.array([0., 0., -1.]); y = np.array([np.cos(yaw), np.sin(yaw), 0.])
            rot = Rotation.from_matrix(np.stack([x, y, np.cross(x, y)], axis=1))
            target = np.r_[targets[side]['grasp'] - rot.apply(pad_offset), rot.as_quat()]
            try:
                solution = robot.solve(home, side, target)
                # A reachable grasp alone is insufficient: reject orientations
                # that cannot carry to, lower at, and retreat from the fold target.
                check=solution.copy()
                for key,height in [('grasp',.12),('destination',.12),('destination',0.)]:
                    check=robot.solve(check,side,np.r_[targets[side][key]+[0,0,height]-rot.apply(pad_offset),rot.as_quat()])
                candidates.append((np.linalg.norm(solution[robot.arms[side]]-home[robot.arms[side]]), rot))
            except ValueError as error:
                failures.append(dict(yaw=yaw,error=str(error)))
                continue
        if not candidates:
            raise ValueError(f'No reachable top-down path for side {side}: {failures}')
        orientations[side] = min(candidates, key=lambda item:item[0])[1]
    q = home.copy()
    op = np.array([(q[ids[0]]-robot.lower[ids[0]])/(robot.upper[ids[0]]-robot.lower[ids[0]]) for ids in robot.jaws])
    commands, openness, phases = [home.copy(), home.copy()], [op.copy(), op.copy()], ['discarded', 'home']
    records = []

    def append_motion(name, end, seconds):
        nonlocal q, poses
        begin = robot.fk(q)
        fractions = np.linspace(0, 1, 13)
        curves = {s:Slerp([0,1], Rotation.from_quat(np.stack([begin[s,3:], end[s,3:]]))) for s in active}
        knots, seed = [q.copy()], q.copy()
        for u in fractions[1:]:
            for side in active:
                target = np.r_[(1-u)*begin[side,:3] + u*end[side,:3], curves[side](u).as_quat()]
                seed = robot.solve(seed, side, target)
            knots.append(seed.copy())
        spline = CubicSpline(fractions, np.array(knots), axis=0)
        # Uniform dilation preserves the path. Check both arms on the resulting
        # sampled command clock, not just requested Cartesian waypoint distance.
        frames = int(np.ceil(seconds * 60))
        for _ in range(6):
            values = spline(quintic(np.arange(frames + 1) / frames))
            if np.any(values < robot.lower - 1e-5) or np.any(values > robot.upper + 1e-5):
                raise ValueError('Interpolated IK path leaves actual joint limits')
            values = np.clip(values, robot.lower, robot.upper)
            fk = np.stack([robot.fk(row) for row in values])
            v = np.diff(fk[:,:,:3], axis=0)*60
            acc = np.diff(np.concatenate([np.zeros_like(v[:1]),v,np.zeros_like(v[:1])]),axis=0)*60
            jv = np.diff(values,axis=0)*60
            ja = np.diff(np.concatenate([np.zeros_like(jv[:1]),jv,np.zeros_like(jv[:1])]),axis=0)*60
            speed = float(np.linalg.norm(v,axis=-1).max())
            acceleration = float(np.linalg.norm(acc,axis=-1).max())
            dilation = max(speed/.28, np.sqrt(acceleration/2.), np.abs(jv).max()/1.5, np.sqrt(np.abs(ja).max()/20.))
            if dilation <= 1.001:
                break
            frames = int(np.ceil(frames*dilation*1.03))
        else:
            raise ValueError('Motion cannot meet sampled command bounds')
        records.append(dict(name=name,start=len(commands),stop=len(commands)+frames,
            sampled_wrist_speed_m_s=speed,sampled_wrist_acceleration_m_s2=acceleration))
        commands.extend(values[1:]); openness.extend([op.copy() for _ in range(frames)]); phases.extend([name]*frames)
        q = values[-1].copy(); poses = robot.fk(q)

    def jaw_phase(name, final, frames=90):
        nonlocal q, op
        start = op.copy()
        for u in quintic(np.arange(1,frames+1)/frames):
            op = (1-u)*start+u*final
            for side in (0,1):
                ids=robot.jaws[side];q[ids]=robot.lower[ids]+op[side]*(robot.upper[ids]-robot.lower[ids])
            commands.append(q.copy());openness.append(op.copy());phases.append(name)
        records.append(dict(name=name,start=len(commands)-frames,stop=len(commands)))

    def target_poses(kind, height):
        end = poses.copy()
        for side in active:
            rot = orientations[side]
            point = targets[side][kind] + np.array([0.,0.,height])
            end[side] = np.r_[point-rot.apply(pad_offset),rot.as_quat()]
        return end

    jaw_phase('open-at-home', np.ones(2), frames=60)
    append_motion('transit-above-grasp', target_poses('grasp', .12), 5.)
    append_motion('approach', target_poses('grasp', 0.), 2.)
    closed = np.ones(2); closed[active] = 0.
    jaw_phase('close', closed)
    append_motion('lift', target_poses('grasp', .10), 2.)
    append_motion('fold-carry', target_poses('destination', .10), 3.)
    append_motion('place', target_poses('destination', 0.), 2.)
    jaw_phase('release', np.ones(2))
    append_motion('retreat-above-fold', target_poses('destination', .12), 2.)
    append_motion('return-home', home_poses, 5.)
    # Same fixed 36-second source clock across the gallery, plus existing replay
    # 1s initial hold and 2s terminal hold. Padding is an explicit home hold.
    length = 2161
    if len(commands) > length:
        raise ValueError(f'Plan exceeds common 36s source budget: {len(commands)} frames')
    padding = length-len(commands)
    commands.extend([q.copy() for _ in range(padding)]);openness.extend([op.copy() for _ in range(padding)]);phases.extend(['home-hold']*padding)
    return dict(joint_q=np.asarray(commands,dtype=np.float32),openness=np.asarray(openness,dtype=np.float32),
        base_transform=np.tile(np.array([0,0,.17,0,0,0,1],np.float32),(length,1)), phase=np.array(phases)), records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','home','reference','sim1-source','assets-root','pad-audit','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--family',choices=FAMILIES,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    import warp as wp
    wp.config.kernel_cache_dir=str(args.out.resolve()/'kernel-cache')
    from .environment import build_environment
    from .robot import read_home
    from .replay import command_sequence
    env, model, _ = build_environment(args.input,device='cpu',sim1_source=args.sim1_source,assets_root=args.assets_root)
    home, op=read_home(args.home,env,model)
    if np.any(op <= .6):
        raise ValueError('Preview planner expects both hands initially contact-disabled')
    protocol=json.loads((args.reference/'PROTOCOL.json').read_text())
    if protocol['source_hashes'][str(args.input.resolve())] != digest(args.input):
        raise ValueError('Targeting reference uses a different garment input')
    with np.load(args.reference/'frames-000000-000120.npz',allow_pickle=False) as z:
        points=z['particle_q'][60].copy()
        np.testing.assert_allclose(z['actual_joint_q'][0],home,atol=1e-5,rtol=0)
    # Prior measured actual pad triangles, not a guessed fixed wrist height.
    audit=json.loads(args.pad_audit.read_text())
    targets=select_targets(points,args.family)
    robot=RobotIK(env,model,home)
    arrays,phases=make_plan(robot,home,targets,pad_offset=np.array(audit['measured_pad_center_wrist_q0_m']))
    command_sequence(arrays,home,robot.lower,robot.upper,np.array(robot.jaws))
    np.savez_compressed(args.out/'commands.npz',**arrays)
    def serial(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
        return value
    write_once(args.out/'PLAN.json',dict(family=args.family,targets=serial(targets),phases=phases,
        target_reference_state=60,target_reference_seconds=1.,targeting_only_no_state_reset=True,
        source_sha256={str(x.resolve()):digest(x) for x in [args.input,args.home,args.pad_audit,
            args.reference/'PROTOCOL.json',args.reference/'frames-000000-000120.npz',Path(__file__)]},
        command_sha256=digest(args.out/'commands.npz'),accepted_for_training=False,
        limitations=['IK and sampled wrist limits are not full arm/self collision certification',
            'Frozen reference target may move in a fresh rollout', 'Physical grasp retention and fold outcome require actual replay review',
            'Uses existing one-way coupling and openness-dependent contact policy']))
    print(json.dumps(dict(family=args.family,frames=len(arrays['joint_q']),targets=serial(targets))),flush=True)


if __name__=='__main__':main()
