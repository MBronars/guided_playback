"""Bounded matched-home execution diagnostic, not a grasp/fold collector.

Compare fresh-process eager and captured runs with identical finalized inputs.
Capture covers exactly one 60-Hz physical frame. Per-substep cloth states are
copied into fixed GPU buffers inside either execution mode and checked on CPU
after each frame. Collision distance is an explicit one-variable experiment;
the native 2 mm value remains the default. No damping/rest/controller change.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from collection.common import digest, write_once
from collection.lease import gpu_lease
from .environment import build_environment
from .inputs import REST_FIELDS
from .recording import StateRecorder
from .robot import GripperCollisionFilter, gripper_filter, read_home
from .settling import SettlingGate
from .provenance import snapshot_sources, robot_assets


def save_arrays(path, arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())


def run(args):
    validate_options(args)
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    def timeout(signum, frame):
        raise TimeoutError('Owned diagnostic wall budget or signal')
    for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT): signal.signal(sig, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.wall_seconds)
    result = dict(status='initializing', accepted_for_training=False)
    try:
        result.update(execute(args, began))
    except Exception as exc:
        result.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['wall_seconds'] = time.monotonic() - began
        write_once(args.out / 'RESULT.json', result)


def validate_options(args):
    replaying = getattr(args, 'commands', None) is not None
    maximum_frames, maximum_wall = (2400, 3600) if replaying else (720, 1200)
    if not 1 <= args.frames <= maximum_frames or args.substeps != 24 or not 0 < args.wall_seconds <= maximum_wall:
        raise ValueError('Diagnostic exceeds frame/wall budget or changes native 24 substeps')
    args.self_contact_distance = getattr(args, 'self_contact_distance', .002)
    args.probe_vertex = getattr(args, 'probe_vertex', [])
    if args.self_contact_distance not in (.0005, .001, .002):
        raise ValueError('Only explicitly declared self-contact distance controls allowed')
    if len(args.probe_vertex) > 8 or len(set(args.probe_vertex)) != len(args.probe_vertex) or any(
            type(i) is not int or i < 0 for i in args.probe_vertex):
        raise ValueError('At most eight distinct nonnegative probe vertex IDs required')


def execute(args, began):
    replaying = getattr(args, 'commands', None) is not None
    import warp as wp
    if args.device == 'cpu':
        if not args.build_only or args.mode != 'eager' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU is for explicitly CUDA-hidden model construction only')
        wp.config.enable_cuda = False
    else:
        from collection.mpm.collect import gpu_status
        if replaying and getattr(args, 'shared_gpu', False):
            from .replay import shared_gpu_status as gpu_status
        if not args.uuid or os.environ.get('CUDA_VISIBLE_DEVICES') != args.uuid:
            raise ValueError('Exactly the allocated UUID must be visible')
        gpu_status(args.uuid)
    wp.config.kernel_cache_dir = str(args.out / 'kernel-cache')
    import newton
    source_root = args.sim1_source.resolve(strict=True)
    local_root = Path(__file__).resolve().parents[2]
    sources = {str(p.resolve()): digest(p) for root in
               (source_root, local_root / 'collection', Path(newton.__file__).resolve().parent)
               for p in sorted(root.rglob('*.py'))}
    sources.update({str(p.resolve()): digest(p) for p in (args.input, args.home)})
    if replaying:
        sources[str(args.commands.resolve())] = digest(args.commands)
    sources.update(robot_assets(args.assets_root))
    for obj in (newton.ModelBuilder, newton.solvers.SolverMuJoCo, newton.solvers.SolverVBD):
        p = Path(inspect.getfile(obj)).resolve()
        sources[str(p)] = digest(p)
    write_once(args.out / 'SOURCES.json', sources)
    source_archive = snapshot_sources(sources, args.out / 'executed_sources.tar.gz')
    env, model, material = build_environment(args.input, device=args.device,
        sim1_source=source_root, assets_root=args.assets_root)
    from components.function.warp_kernel import update_control_kernel
    if Path(inspect.getfile(update_control_kernel.func)).resolve() != source_root / 'components/function/warp_kernel.py':
        raise ValueError('Controller imported from a different simulator checkout')
    if len(inspect.signature(update_control_kernel.func).parameters) != 19:
        raise ValueError('Native controller signature changed')
    home, openness = read_home(args.home, env, model)
    lifecycle = GripperCollisionFilter(env, model) if replaying else None
    filtering = lifecycle.apply(openness) if replaying else gripper_filter(env, model, openness)
    commands, command_metadata = None, None
    if replaying:
        from .replay import SCHEMA, command_sequence
        with np.load(args.commands, allow_pickle=False) as raw_commands:
            expected_base = np.array([0., 0., env.ROBOT_BASE_HEIGHT, 0., 0., 0., 1.])
            if 'base_transform' not in raw_commands or not np.allclose(
                    raw_commands['base_transform'], expected_base, rtol=0., atol=1e-7):
                raise ValueError('Recorded robot base differs from the actual fixed-base replay')
            commands, command_metadata = command_sequence(raw_commands, home,
                model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy(),
                [env.left_gripper_joint_indices, env.right_gripper_joint_indices])
        if getattr(args, 'retime_return', False):
            from .retime import retime_return
            fk_state = model.state()
            fk_state.joint_qd.zero_()
            def command_fk(q):
                fk_state.joint_q.assign(q)
                newton.eval_fk(model, fk_state.joint_q, fk_state.joint_qd, fk_state)
                return fk_state.body_q.numpy()[[env.lee_index, env.ree_index]]
            commands, command_metadata = retime_return(commands, command_metadata, command_fk)
        args.frames = len(commands['command_joint_q'])
        save_arrays(args.out / 'commands.npz', commands)
    final = {key: getattr(model, key).numpy() for key in REST_FIELDS}
    final.update({key: getattr(model, key).numpy() for key in
        ('particle_q', 'tri_indices', 'edge_indices', 'particle_inv_mass', 'tri_materials',
         'particle_radius', 'shape_material_mu', 'shape_flags', 'joint_limit_lower', 'joint_limit_upper')})
    final.update(home=home, openness=openness, soft_contact_ke=np.array(model.soft_contact_ke),
        soft_contact_kd=np.array(model.soft_contact_kd), soft_contact_mu=np.array(model.soft_contact_mu))
    save_arrays(args.out / 'finalized_model.npz', final)
    if any(i >= model.particle_count for i in args.probe_vertex):
        raise ValueError('Contact probe outside material vertex range')
    protocol = dict(schema='sim1-matched-home-execution-v2', mode=args.mode, frames=args.frames,
        substeps=args.substeps, fps=60, dt_s=1/(60*args.substeps),
        input_sha256=digest(args.input), home_sha256=digest(args.home),
        finalized_sha256=digest(args.out / 'finalized_model.npz'),
        simulator_source=str(source_root), assets_root=str(args.assets_root.resolve()),
        source_hashes=sources, source_archive=source_archive, numpy_version=np.__version__, warp_version=wp.__version__,
        device=str(model.device), uuid=args.uuid, robot_included=True,
        gripper_filter=filtering, initial_policy='FK initialized at recorded home in both state buffers',
        physical_scope='one-way MuJoCo rigid to VBD cloth; fixed joint target; no IK',
        contact_policy=f'native interval -1, {1000*args.self_contact_distance:g}mm self-contact radius, 3mm margin, 10mm rigid margin',
        self_contact_distance_m=args.self_contact_distance, probe_vertex_ids=args.probe_vertex,
        probe_scope='pre-solve detector vertex/triangle IDs copied after each substep; not force measurements',
        rigid_update_data_interval=1,
        actual_cloth_solver_file=inspect.getfile(newton.solvers.SolverVBD),
        data_scope='60Hz actual states and all physical substeps; no export or grasp success claim',
        recorder_max_bytes=2*1024**3 if replaying else 1024**3,
        accepted_for_training=False)
    if replaying:
        protocol.update(schema=SCHEMA, command_source_sha256=digest(args.commands),
            gpu_sharing=bool(getattr(args, 'shared_gpu', False)),
            gpu_memory_policy=('shared: >=32GiB free at admission, >=16GiB free during replay, own observed <=16GiB; not an allocation cap'
                if getattr(args, 'shared_gpu', False) else 'exclusive GPU'),
            gripper_original_flags=lifecycle.baseline.tolist(),
            fixed_robot_base_xyzw=expected_base.tolist(),
            end_effector_body_ids=[env.lee_index, env.ree_index],
            command_plan_sha256=digest(args.out / 'commands.npz'), command_metadata=command_metadata,
            physical_scope='one-way MuJoCo rigid to VBD cloth; recorded joint commands; no IK or cloth attachment',
            data_scope='60Hz actual states; every-substep online diagnostics; dense substep states not persisted',
            settling_policy='fixed recorded 1s prefix; quiet/hold metrics diagnostic only, no wait-until-quiet gate',
            gripper_policy='restore original collision eligibility on closing; native openness>0.6 disables contacts')
    write_once(args.out / 'PROTOCOL.json', protocol)
    if args.build_only:
        for path, expected in sources.items():
            if digest(path) != expected: raise ValueError('Source changed during build: ' + path)
        return dict(status='model_built_only', particles=model.particle_count, bodies=model.body_count,
            finalized_sha256=protocol['finalized_sha256'])
    with wp.ScopedDevice(model.device):
        state, nxt = model.state(), model.state()
        for buffer in (state, nxt):
            buffer.joint_q.assign(home)
            buffer.joint_qd.zero_()
            newton.eval_fk(model, buffer.joint_q, buffer.joint_qd, buffer)
        control = model.control()
        control.joint_target.assign(home)
        dt = protocol['dt_s']
        rigid = newton.solvers.SolverMuJoCo(model, njmax=150000, solver='newton', cone='elliptic',
            use_mujoco_cpu=False, use_mujoco_contacts=False, contact_stiffness_time_const=dt,
            update_data_interval=1)
        cloth = newton.solvers.SolverVBD(model, iterations=16, handle_self_contact=True,
            self_contact_radius=args.self_contact_distance, self_contact_margin=.003,
            vertex_collision_buffer_pre_alloc=32, edge_collision_buffer_pre_alloc=64,
            integrate_with_external_rigid_solver=True, collision_detection_interval=-1, use_tile_solve=True)
        ids = [wp.array(x, dtype=int) for x in (env.left_gripper_joint_indices,
            env.right_gripper_joint_indices, env.controllable_joint_indices)]
        q0, target = wp.array(home, dtype=float), wp.array(home, dtype=float)
        qd = wp.zeros(model.joint_dof_count, dtype=float)
        jaws = wp.array([openness[0], openness[0], openness[1], openness[1]], dtype=float)
        # Device copies work in a CUDA graph without host .numpy() reads.
        qs = [wp.empty_like(state.particle_q) for _ in range(args.substeps)]
        vs = [wp.empty_like(state.particle_qd) for _ in range(args.substeps)]
        bs = [wp.empty_like(state.body_q) for _ in range(args.substeps)]
        js = [wp.empty_like(state.joint_q) for _ in range(args.substeps)]
        detector = cloth.trimesh_collision_detector
        contact_fields = ('vertex_colliding_triangles_count', 'vertex_colliding_triangles_buffer_sizes',
                          'edge_colliding_edges_count', 'edge_colliding_edges_buffer_sizes')
        counts = {key: [wp.empty_like(getattr(detector, key)) for _ in range(args.substeps)]
                  for key in contact_fields}
        offsets = detector.vertex_colliding_triangles_offsets.numpy()
        probe_offsets = {i: int(offsets[i])*2 for i in args.probe_vertex}
        probes = {i: [wp.empty(int(offsets[i+1]-offsets[i])*2, dtype=wp.int32, device=model.device)
                      for _ in range(args.substeps)] for i in args.probe_vertex}

    def frame():
        nonlocal state, nxt
        for substep in range(args.substeps):
            wp.launch(update_control_kernel, dim=max(len(x) for x in ids),
                inputs=[model.joint_limit_lower, model.joint_limit_upper, *ids, q0, target,
                    state.joint_q, control.joint_target, qd, state.joint_qd,
                    len(env.left_gripper_joint_indices), len(env.right_gripper_joint_indices),
                    len(env.controllable_joint_indices), args.substeps, substep, dt, jaws, 1],
                device=model.device)
            contacts = model.collide(state)
            state.clear_forces(); nxt.clear_forces()
            count = model.particle_count
            try:
                model.particle_count = 0
                rigid.step(state, nxt, control, contacts, dt)
                state.particle_f.zero_()
            finally:
                model.particle_count = count
            contacts = model.collide(state, soft_contact_margin=.01)
            cloth.step(state, nxt, None, contacts, dt)
            state, nxt = nxt, state
            wp.copy(qs[substep], state.particle_q)
            wp.copy(vs[substep], state.particle_qd)
            wp.copy(bs[substep], state.body_q)
            wp.copy(js[substep], state.joint_q)
            for key in contact_fields: wp.copy(counts[key][substep], getattr(detector, key))
            for vertex, values in probes.items():
                wp.copy(values[substep], detector.vertex_colliding_triangles,
                        src_offset=probe_offsets[vertex], count=values[substep].size)

    def arrays():
        return dict(particle_q=state.particle_q.numpy(), particle_qd=state.particle_qd.numpy(),
                    body_q=state.body_q.numpy(), joint_q=state.joint_q.numpy())
    recorder = StateRecorder(60, model.particle_count, model.body_count, model.joint_coord_count,
                             max_states=args.frames+1, max_bytes=protocol['recorder_max_bytes'])
    recorder.append_initial(**arrays())
    graph = None
    if args.mode == 'captured':
        with wp.ScopedCapture(device=model.device) as capture: frame()
        graph = capture.graph
    gate, trace, flag_history = SettlingGate(dt), [], []
    chunk_start = 0
    for index in range(args.frames):
        gpu_status(args.uuid, allow_own=True, memory_limit_mib=16384)
        cloth.rebuild_bvh(state)
        next_q = commands['command_joint_q'][index] if replaying else home
        next_op = commands['command_openness'][index] if replaying else openness
        q0.assign(state.joint_q); target.assign(next_q)
        if replaying:
            previous_op = commands['command_openness'][index - 1] if index else openness
            jaws.assign([previous_op[0], next_op[0], previous_op[1], next_op[1]])
            flag_history.append(lifecycle.apply(next_op)['shape_flags_after_filter'])
        if graph is None: frame()
        else: wp.capture_launch(graph)
        wp.synchronize_device(model.device)
        qvalues, vvalues = np.stack([v.numpy() for v in qs]), np.stack([v.numpy() for v in vs])
        bvalues, jvalues = np.stack([v.numpy() for v in bs]), np.stack([v.numpy() for v in js])
        cvalues = {key: np.stack([v.numpy() for v in values]) for key, values in counts.items()}
        pvalues = {f'probe_vertex_{i}_pairs': np.stack([v.numpy().reshape(-1,2) for v in values])
                   for i,values in probes.items()}
        for s in range(args.substeps):
            overflow = max(0, int((cvalues['vertex_colliding_triangles_count'][s] -
                                  cvalues['vertex_colliding_triangles_buffer_sizes'][s]).max(initial=0)),
                int((cvalues['edge_colliding_edges_count'][s] -
                     cvalues['edge_colliding_edges_buffer_sizes'][s]).max(initial=0)))
            row = gate.observe(index*args.substeps+s+1, qvalues[s], vvalues[s], overflow)
            row['overflow_coverage'] = 'every_physical_substep_device_copy'
            if not np.isfinite(bvalues[s]).all() or not np.isfinite(jvalues[s]).all():
                row['guard'] = 'nonfinite_rigid_state'
            trace.append(row)
        if not replaying:
            save_arrays(args.out / f'substeps-{index:04d}.npz', dict(particle_q=qvalues, particle_qd=vvalues,
                body_q=bvalues, actual_joint_q=jvalues, **cvalues, **pvalues))
        elif any(row['guard'] for row in trace[-args.substeps:]):
            # Preserve the exact triggering substep before recorder validation can
            # reject a nonfinite endpoint. This changes evidence, not the solver.
            from .replay import save_failure
            save_failure(args.out, index, trace[-args.substeps:], dict(
                particle_q=qvalues, particle_qd=vvalues, body_q=bvalues,
                actual_joint_q=jvalues, **cvalues, **pvalues))
        recorder.append_transition(index, command_joint_q=next_q, command_openness=next_op, **arrays())
        if replaying and ((index + 1) % 120 == 0 or index + 1 == args.frames or
                          any(row['guard'] for row in trace[-args.substeps:])):
            from .replay import save_chunk
            chunk_start = save_chunk(recorder, chunk_start, args.out)
        if any(row['guard'] for row in trace[-args.substeps:]):
            raise ValueError('Substep numerical guard failed; prefix retained')
        print(json.dumps(dict(frame=index+1, max_speed_m_s=max(r['max_speed_m_s'] for r in trace[-args.substeps:]))), flush=True)
    save_arrays(args.out / 'actual_states.npz', recorder.arrays(args.frames))
    if replaying:
        save_arrays(args.out / 'collision_flags.npz', dict(shape_flags=np.asarray(flag_history, dtype=np.int32)))
    write_once(args.out / 'SUBSTEPS.json', trace)
    for path, expected in sources.items():
        if digest(path) != expected: raise ValueError('Source changed during execution: ' + path)
    return dict(status='complete_diagnostic', frames=args.frames+1, substeps=len(trace),
        sources_unchanged=True, physics_qualified=False, grasp_verified=False,
        artifacts={p.name: digest(p) for p in args.out.iterdir() if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'home', 'sim1-source', 'assets-root', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
    parser.add_argument('--uuid')
    parser.add_argument('--mode', choices=['eager', 'captured'], default='eager')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--substeps', type=int, default=24)
    parser.add_argument('--wall-seconds', type=int, default=1200)
    parser.add_argument('--self-contact-distance', type=float, choices=[.0005,.001,.002], default=.002)
    parser.add_argument('--probe-vertex', type=int, action='append', default=[])
    args = parser.parse_args()
    if args.device == 'cpu': run(args)
    else:
        with gpu_lease(args.uuid): run(args)


if __name__ == '__main__': main()
