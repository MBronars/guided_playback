"""Independent engine diagnostic, explicitly not a source-PhysX execution test.

Only the gripper is commanded. Object state is initialized from observed context
and subsequently advanced by MuJoCo. Qualify on recorded controls first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tarfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch


def numbers(x):
    return ' '.join(format(float(v), '.10g') for v in np.asarray(x).ravel())


def quat(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def component_vertices(body, tolerance=2e-5):
    from scipy.spatial import ConvexHull
    vertices = body['vertices'].double().numpy()
    result = []
    for planes in body['planes']:
        p = planes.double().numpy()
        v = np.unique(vertices[(vertices @ p[:, :3].T + p[:, 3]).max(-1) <= tolerance], axis=0)
        h = ConvexHull(v)
        result.append(v[h.vertices])
    return result


def add_obstacles(world, spec):
    """Add the planner's exact box dimensions/poses and enable all body contacts."""
    from guidance.obstacles import validate_obstacles
    for i, box in enumerate(validate_obstacles(spec)['boxes']):
        ET.SubElement(world, 'geom', name=f'obstacle_{i}', type='box',
            size=numbers(np.asarray(box['size'])/2), pos=numbers(box['centre']),
            quat=numbers(quat(np.asarray(box['rotation']))), contype='8', conaffinity='3',
            friction='1 .005 .0001')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--trajectory')
    p.add_argument('--obstacles', help='Same supplied static-box JSON used by the planner, when applicable')
    p.add_argument('--frames', type=int, default=36)
    p.add_argument('--step', type=float, default=.001)
    p.add_argument('--finger-mode', choices=['achieved', 'recorded', 'contact_preload', 'mesh_preload'], default='achieved')
    p.add_argument('--finger-velocity-feedforward',action='store_true',
                   help='Compensate the position servo damping using the planned achieved finger velocity')
    p.add_argument('--save-frame-states',action='store_true',
                   help='Save the compiled model and complete integration state at each output frame for exact continuation checks')
    p.add_argument('--preload', type=float, default=.006)
    p.add_argument('--preload-policy',choices=['instant','latched'],default='instant')
    p.add_argument('--preload-release-mm',type=float,default=2.,help='For latched preload: planned opening/closing stroke that releases/rearms the latch')
    p.add_argument('--contact-tolerance', type=float, default=.005)
    p.add_argument('--mesh-contact-tolerance', type=float, default=.002)
    p.add_argument('--weld-time', type=float, default=.002)
    p.add_argument('--contact-time', type=float, default=.005)
    p.add_argument('--joint-limit-time',type=float,
                   help='Optional finger limit time constant; omission preserves the MuJoCo default')
    p.add_argument('--friction-scale', type=float, default=1.)
    p.add_argument('--table-offset', type=float, default=0.)
    p.add_argument('--dependency-path', action='append', default=[],
                   help='Append an optional dependency directory after NumPy/PyTorch import; repeat as needed')
    args = p.parse_args()
    if args.trajectory and args.finger_mode == 'recorded':
        raise ValueError('Recorded future commands are only for recorded positive controls')
    if args.finger_velocity_feedforward and args.finger_mode=='recorded':
        raise ValueError('Recorded commands must retain their original controller semantics')
    if args.preload_policy!='instant' and args.finger_mode!='mesh_preload':
        raise ValueError('Latched preload currently supports mesh-preload replay only')
    for path in args.dependency_path:
        sys.path.append(str(Path(path).resolve()))
    import mujoco
    from scipy.spatial.transform import Rotation, Slerp
    if args.frames < 1 or not np.isfinite(args.step) or args.step <= 0:
        raise ValueError('Frames and integration step must be positive')
    if not np.isfinite(args.preload) or args.preload < 0 or args.mesh_contact_tolerance <= 0:
        raise ValueError('Preload must be nonnegative and contact tolerance positive')
    if min(args.weld_time,args.contact_time,args.friction_scale) <= 0:
        raise ValueError('Controller/contact times and friction scale must be positive')
    if args.joint_limit_time is not None and (not np.isfinite(args.joint_limit_time)
            or args.joint_limit_time < 2*args.step):
        raise ValueError('Finger limit time must be finite and at least twice the integration step')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    scene = Path(args.scene)
    b = torch.load(scene/'input.pt', weights_only=False, map_location='cpu')
    g = torch.load(scene/'collision_geometry.pt', weights_only=False, map_location='cpu')
    c = torch.load(scene/'calibration.pt', weights_only=False, map_location='cpu')['calibration']
    c = {k: v.double().numpy() if torch.is_tensor(v) else v for k, v in c.items()}
    episode = json.loads((scene/'source_episode.json').read_text())
    with tarfile.open(episode['shard']) as tar:
        raw = {k: np.asarray(pickle.load(tar.extractfile(episode['key']+'.'+k+'.pyd')))[episode['elem']]
               for k in (('material_properties', 'joint_pos_target') if args.finger_mode == 'recorded' else ('material_properties',))}
    C = b.context_len
    if b.batch_size != 1 or C < 2 or b.body_type.shape[1] != 4:
        raise ValueError('Expected one object, Panda hand, two fingers and at least two context frames')
    if b.pose_action_mask[0].tolist() != [False,True,True,True]:
        raise ValueError('Expected passive object, actuated hand and two actuated fingers')
    poses = b.body_poses[0, :C if args.trajectory else None].double().numpy()
    desired = poses.copy()
    if args.trajectory:
        planning_protocol = Path(args.trajectory).parent/'protocol.json'
        if planning_protocol.exists():
            planned = json.loads(planning_protocol.read_text())
            if planned.get('obstacles_sha256'):
                if not args.obstacles or hashlib.sha256(Path(args.obstacles).read_bytes()).hexdigest()!=planned['obstacles_sha256']:
                    raise ValueError('Replay must include the exact obstacles supplied to the planner')
        z = np.load(args.trajectory)
        desired = np.broadcast_to(np.eye(4), (*z['T'][0].shape[:2], 4, 4)).copy()
        desired[..., :3, :3], desired[..., :3, 3] = z['R'][0], z['T'][0]
        np.testing.assert_allclose(desired[:C], poses[:C], atol=2e-6, rtol=0)
        if not np.isfinite(desired).all():
            raise ValueError('Nonfinite generated trajectory')
    H = min(args.frames, len(desired)-C)
    if H < 1:
        raise ValueError('Trajectory must include future frames after observed context')
    dt = float(c['dt'])
    nsub = round(dt/args.step)
    if nsub < 1 or abs(nsub*args.step-dt) > 1e-9:
        raise ValueError('Frame interval must be divisible by integration step')
    anchor = poses[C-1]
    relative_R = anchor[1, :3, :3].T @ anchor[2:, :3, :3]
    offsets = c['finger_anchor'] - c['q_anchor'][:, None]*c['axis']
    axes = np.einsum('fji,fj->fi', relative_R, c['axis'])
    local = np.einsum('tji,tfj->tfi', desired[:, 1, :3, :3],
                      desired[:, 2:, :3, 3]-desired[:, 1, None, :3, 3])
    q_desired = c['q_anchor'] + ((local-c['finger_anchor'])*c['axis']).sum(-1)
    contact_gate = np.zeros(len(desired), dtype=bool)
    if args.finger_mode == 'contact_preload':
        if args.preload < 0 or args.contact_tolerance <= 0:
            raise ValueError('Invalid contact-preload settings')
        local_points = b.body_templates[0].double().numpy()
        mask = b.body_template_mask[0].numpy()
        world_points = [np.einsum('tij,nj->tni', desired[:,k,:3,:3], local_points[k,mask[k]]) + desired[:,k,None,:3,3] for k in range(4)]
        from scipy.spatial.distance import cdist
        gaps = np.array([[cdist(world_points[0][t],world_points[k][t]).min() for k in (2,3)] for t in range(len(desired))])
        contact_gate = gaps.max(-1) <= args.contact_tolerance
    if args.finger_mode == 'mesh_preload':
        from guidance.collision_distance import ConvexCollisionDistance
        probe=ConvexCollisionDistance(g)
        gap=np.array([probe.query(RR[:,:3,:3],RR[:,:3,3])[0][1:] for RR in desired])
        contact_gate=gap.max(-1)<=args.mesh_contact_tolerance
    from guidance.replay_control import mesh_preload_commands
    q_command,preload_active=mesh_preload_commands(q_desired,contact_gate,preload=args.preload,
        policy=args.preload_policy,release_delta=args.preload_release_mm/1000)

    xml = ET.Element('mujoco', model='observed pickup geometry, independent engine')
    ET.SubElement(xml, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(xml, 'option', timestep=str(args.step), gravity='0 0 -9.81',
                  integrator='implicitfast', cone='elliptic', iterations='100', tolerance='1e-10')
    default = ET.SubElement(xml, 'default')
    ET.SubElement(default, 'geom', solref=f'{args.contact_time} 1',
                  solimp='.95 .99 .001', margin='0', condim='4')
    asset = ET.SubElement(xml, 'asset')
    world = ET.SubElement(xml, 'worldbody')
    obstacle_spec = None
    if args.obstacles:
        from guidance.obstacles import load_obstacles
        obstacle_spec = load_obstacles(args.obstacles)
        add_obstacles(world, obstacle_spec)
    support_z = float(g['metadata']['support_z']) + args.table_offset
    ET.SubElement(world, 'geom', name='table', type='plane', size='2 2 .1',
                  pos=f'0 0 {support_z}', contype='4', conaffinity='3', friction='1 .005 .0001')
    object_node = ET.SubElement(world, 'body', name='object', pos=numbers(anchor[0, :3, 3]),
                               quat=numbers(quat(anchor[0, :3, :3])))
    ET.SubElement(object_node, 'freejoint', name='object_free')
    mocap = ET.SubElement(world, 'body', name='hand_target', mocap='true',
                         pos=numbers(anchor[1, :3, 3]), quat=numbers(quat(anchor[1, :3, :3])))
    hand = ET.SubElement(world, 'body', name='hand', pos=numbers(anchor[1, :3, 3]),
                        quat=numbers(quat(anchor[1, :3, :3])), gravcomp='1')
    ET.SubElement(hand, 'freejoint', name='hand_free')
    nodes = [object_node, hand]
    for i in range(2):
        finger = ET.SubElement(hand, 'body', name=f'finger{i}', pos=numbers(offsets[i]),
                              quat=numbers(quat(relative_R[i])), gravcomp='1')
        limit={} if args.joint_limit_time is None else dict(solreflimit=f'{args.joint_limit_time} 1')
        ET.SubElement(finger, 'joint', name=f'finger_q{i}', type='slide', axis=numbers(axes[i]),
                      range=numbers(c['limits'][i, :2]), damping='0',**limit)
        nodes.append(finger)
    component_counts = []
    for k, node in enumerate(nodes):
        components = component_vertices(g['bodies'][k])
        component_counts.append(len(components))
        rho = float(b.rho[0, b.particle_body[0] == k][0])
        friction = float(raw['material_properties'][-1 if k == 0 else 0, 3])*args.friction_scale
        for j, vertices in enumerate(components):
            name = f'body{k}_component{j}'
            ET.SubElement(asset, 'mesh', name=name, vertex=numbers(vertices))
            ET.SubElement(node, 'geom', name=name, mesh=name, type='mesh', density=str(rho),
                          contype='1' if k == 0 else '2', conaffinity='6' if k == 0 else '1',
                          friction=f'{friction} .005 .0001')
    equality = ET.SubElement(xml, 'equality')
    ET.SubElement(equality, 'weld', name='hand_tracking', body1='hand_target', body2='hand',
                  solref=f'{args.weld_time} 1', solimp='.99 .999 .001')
    act = ET.SubElement(xml, 'actuator')
    for i in range(2):
        ET.SubElement(act, 'position', joint=f'finger_q{i}', kp='2000', kv='100',
                      forcerange='-200 200', ctrllimited='false')
    xml_text = ET.tostring(xml, encoding='unicode')
    (out/'scene.xml').write_text(xml_text)
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    velocity_offset=None
    if args.finger_velocity_feedforward:
        from guidance.replay_control import damping_compensation
        velocity_offset=damping_compensation(torch.from_numpy(np.diff(q_desired,axis=0)/dt),
            stiffness=torch.from_numpy(model.actuator_gainprm[:,0].copy()),
            damping=torch.from_numpy(-model.actuator_biasprm[:,2].copy())).numpy()
    ids = [model.body(name).id for name in ('object', 'hand', 'finger0', 'finger1')]
    asset_dir = Path(g['metadata']['asset'])
    object_properties = json.loads((asset_dir/(asset_dir.name+'.json')).read_text())
    mass = float(object_properties['mass'])
    mass_scale = mass/model.body_mass[ids[0]]
    model.body_mass[ids[0]] *= mass_scale
    model.body_inertia[ids[0]] *= mass_scale
    mujoco.mj_setConst(model, data)
    free = [model.joint(name).qposadr[0] for name in ('object_free', 'hand_free')]
    jq = [model.joint(f'finger_q{i}').qposadr[0] for i in range(2)]
    def qpos_at(frame):
        q = model.qpos0.copy()
        for k, adr in enumerate(free):
            q[adr:adr+3] = poses[frame, k, :3, 3]
            q[adr+3:adr+7] = quat(poses[frame, k, :3, :3])
        relative = (poses[frame, 2:, :3, 3]-poses[frame, 1, :3, 3]) @ poses[frame, 1, :3, :3]
        q[jq] = c['q_anchor'] + ((relative-c['finger_anchor'])*c['axis']).sum(-1)
        return q
    data.qpos[:] = qpos_at(C-1)
    mujoco.mj_differentiatePos(model, data.qvel, dt, qpos_at(C-2), data.qpos)
    data.ctrl[:] = data.qpos[jq]
    mujoco.mj_forward(model, data)
    def capture():
        return data.xmat[ids].reshape(4, 3, 3).copy(), data.xpos[ids].copy()
    R0, T0 = capture()
    fk_error = float(np.linalg.norm(T0-anchor[:, :3, 3], axis=-1).max())
    fk_rotation = float(np.max(np.abs(R0-anchor[:, :3, :3])))
    if fk_error > 2e-5 or fk_rotation > 2e-5:
        raise ValueError(f'Initial pose mismatch: {fk_error} m, {fk_rotation} matrix entry')
    protocol = dict(arguments=vars(args), mujoco=mujoco.__version__, source=episode,
                    obstacle_spec=obstacle_spec,
                    initial_fk_error_m=fk_error, initial_fk_rotation_entry_error=fk_rotation,
                    component_counts=component_counts, body_mass_kg=model.body_mass[ids].tolist(),
                    body_inertia_kg_m2=model.body_inertia[ids].tolist(),
                    body_com_local=model.body_ipos[ids].tolist(), object_json=object_properties,
                    support_z=support_z, support_provenance=g['metadata']['support_kind'],
                    contact_friction=model.geom_friction.tolist(),
                    finger_servo_stiffness=model.actuator_gainprm[:,0].tolist(),
                    finger_servo_damping=(-model.actuator_biasprm[:,2]).tolist(),
                    limitations=['Independent MuJoCo engine, not source PhysX.',
                     'Floating dynamic hand follows a mocap weld; arm reachability and torques are not checked.',
                     'Object mass from asset JSON; inertia and COM approximated from convex collision components.',
                     'Initial velocities estimated from two observed frames; no complete simulator snapshot.',
                     'Collision component vertices recovered using 20 micrometre half-space tolerance.',
                     'Contact-preload gate, when enabled, uses planned geometry, not recorded future commands.',
                     'Generated replays read only observed body poses and never load source joint commands.',
                     'Torsional and rolling friction are explicit MuJoCo defaults, not measured source values.'])
    files = [Path(__file__), scene/'input.pt', scene/'collision_geometry.pt', scene/'calibration.pt']
    if args.trajectory:
        files.append(Path(args.trajectory))
    if args.obstacles:
        files.append(Path(args.obstacles))
    protocol['sha256'] = {str(f.resolve()): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    R_list, T_list, q_list, contact_list = [R0], [T0], [data.qpos[jq].copy()], [data.ncon]
    applied_targets=[data.ctrl.copy()]
    states=[]
    if args.save_frame_states:
        signature=mujoco.mjtState.mjSTATE_INTEGRATION
        def capture_state():
            state=np.empty(mujoco.mj_stateSize(model,signature))
            mujoco.mj_getState(model,data,state,signature)
            return state
        states.append(capture_state())
        mujoco.mj_saveModel(model,str(out/'scene.mjb'),None)
    started = time.perf_counter()
    for i in range(H):
        a, z = C-1+i, C+i
        slerp = Slerp([0, 1], Rotation.from_matrix(desired[[a, z], 1, :3, :3]))
        for sub in range(nsub):
            f = (sub+1)/nsub
            data.mocap_pos[0] = (1-f)*desired[a, 1, :3, 3]+f*desired[z, 1, :3, 3]
            data.mocap_quat[0] = slerp([f]).as_quat()[0, [3, 0, 1, 2]]
            if args.finger_mode == 'recorded':
                data.ctrl[:] = raw['joint_pos_target'][a, -2:]
            else:
                target_q=(1-f)*q_command[a]+f*q_command[z]
                if velocity_offset is not None:target_q=target_q+velocity_offset[a]
                data.ctrl[:] = np.clip(target_q, c['limits'][:, 0], c['limits'][:, 1])
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        rr, tt = capture()
        R_list.append(rr); T_list.append(tt); q_list.append(data.qpos[jq].copy()); contact_list.append(data.ncon)
        applied_targets.append(data.ctrl.copy())
        if args.save_frame_states:states.append(capture_state())
    R, T = np.array(R_list), np.array(T_list)
    requested = desired[C-1:C+H]
    distance = np.linalg.norm(T-requested[..., :3, 3], axis=-1)
    Rerr = Rotation.from_matrix((R @ requested[..., :3, :3].swapaxes(-1, -2)).reshape(-1, 3, 3)).magnitude().reshape(H+1, 4)
    result = dict(seconds=time.perf_counter()-started, frames=H, finite=bool(np.isfinite(T).all()),
                  hand_tracking_mean_mm=float(distance[1:, 1].mean()*1000),
                  hand_tracking_max_mm=float(distance[1:, 1].max()*1000),
                  hand_rotation_max_deg=float(np.rad2deg(Rerr[1:, 1].max())),
                  finger_achieved_tracking_max_mm=float(np.max(np.abs(np.array(q_list)-q_desired[C-1:C+H]))*1000),
                  object_prediction_final_error_mm=float(distance[-1, 0]*1000),
                  object_prediction_mean_error_mm=float(distance[1:, 0].mean()*1000),
                  object_rotation_final_error_deg=float(np.rad2deg(Rerr[-1, 0])),
                  object_lift_mm=float((T[-1, 0, 2]-T[0, 0, 2])*1000),
                  requested_lift_mm=float((requested[-1, 0, 2, 3]-requested[0, 0, 2, 3])*1000),
                  warning_counts=data.warning.number.tolist())
    np.savez_compressed(out/'execution.npz', model_R=R, model_T=T, joint_q=np.array(q_list),
                        desired_R=requested[..., :3, :3], desired_T=requested[..., :3, 3],
                        contact_count=np.array(contact_list), dt=dt, requested_q=q_desired[C-1:C+H], commanded_q=q_command[C-1:C+H], planned_contact=contact_gate[C-1:C+H],
                        preload_active=preload_active[C-1:C+H],
                        applied_joint_targets_at_frame_end=np.array(applied_targets),
                        velocity_feedforward_offset=np.zeros((H,2)) if velocity_offset is None else velocity_offset[C-1:C+H-1])
    if args.save_frame_states:
        np.savez_compressed(out/'integration_states.npz',states=np.array(states),signature=int(signature),
                            mujoco_version=mujoco.__version__,model_sha256=hashlib.sha256((out/'scene.mjb').read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
