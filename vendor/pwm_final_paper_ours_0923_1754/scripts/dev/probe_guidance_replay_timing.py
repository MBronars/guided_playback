"""Retiming diagnostic for fixed gripper commands; no model or planner changes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

from guidance.persistent_replay import PersistentPickupReplay
from scripts.dev.guidance_force_audit import DetailedContactRecorder
from utils.reproducibility import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--controls', required=True, type=Path)
    p.add_argument('--vendor', action='append', default=[])
    args = p.parse_args()
    for path in args.vendor:sys.path.append(path)
    root = args.root.resolve();out = root/'timing_probe'
    out.mkdir(exist_ok=False)
    (out/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    report = dict(status='qualifying', source_sha256=file_sha256(__file__), checks=[], rows=[],
        started_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Same saved commands and initial physics state at 1x/2x duration. '
            'Retiming changes dynamics and the held duration; it is not a new model prediction '
            'or an amendment to the frozen campaign acceptance results.')
    def save():
        temp=out/'report.pending.json';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(out/'report.json')
    save()
    controls=json.loads(args.controls.read_text())['results']
    keys=['bottle_8270_left_i64_r050','bottle_8270_left_max_strong_interaction_i64_r050',
        'teacup_7803_left_i64_r050','teacup_7803_left_max_strong_interaction_i64_r050',
        'spice_8026_left_replication_i64_r050']
    cases=[];control_scenes=set()
    for key in keys:
        protocol=json.loads((root/'exports'/key/'protocol.json').read_text())
        scene=Path(protocol['arguments']['input']).parent
        control=next(v for v in controls if v['scene']==protocol['arguments']['task']+'_'+scene.name)
        assert control['qualified_for_generated_diagnostic']
        cases.append(dict(key=key,scene=scene,engines=[root/'replays'/key/tag for tag in ('dt0005','dt00025')],
            references=[root/'replays'/key/tag for tag in ('dt0005','dt00025')],target=protocol['target'][0],kind='generated'))
        if scene not in control_scenes:
            cases.append(dict(key='recorded_'+scene.name,scene=scene,engines=cases[-1]['engines'],
                references=[Path(control['paths'][k]) for k in ('mesh_preload','mesh_preload_fine')],target=None,kind='recorded control'))
            control_scenes.add(scene)
    try:
        # Qualify every exact 1x replay before doing any retimed intervention.
        for case in cases:
            for engine_path,reference_path in zip(case['engines'],case['references']):
                reference=np.load(reference_path/'execution.npz')
                c=torch.load(case['scene']/'calibration.pt',map_location='cpu',weights_only=False)['calibration']
                engine=PersistentPickupReplay(engine_path,frame=0)
                result=engine.advance(reference['desired_R'][:,1],reference['desired_T'][:,1],reference['commanded_q'],
                    dt=float(reference['dt']),joint_limits=np.asarray(c['limits'])[:,:2])
                for actual,expected in [('R','model_R'),('T','model_T'),('q','joint_q')]:
                    np.testing.assert_array_equal(result[actual],reference[expected])
                if (reference_path/'integration_states.npz').exists():
                    np.testing.assert_array_equal(result['integration_states'],np.load(reference_path/'integration_states.npz')['states'])
                report['checks'].append(dict(case=case['key'],step=float(engine.model.opt.timestep),
                    bitwise_pose_and_joint_replay=True,reference_sha256=file_sha256(reference_path/'execution.npz'),
                    state_sha256=file_sha256(engine_path/'integration_states.npz')));save()
        report['status']='running';save()
        for case in cases:
            for factor in (1,2):
                for engine_path,reference_path in zip(case['engines'],case['references']):
                    reference=np.load(reference_path/'execution.npz')
                    c=torch.load(case['scene']/'calibration.pt',map_location='cpu',weights_only=False)['calibration']
                    engine=PersistentPickupReplay(engine_path,frame=0)
                    recorder=DetailedContactRecorder(engine);engine.mujoco=recorder
                    result=engine.advance(reference['desired_R'][:,1],reference['desired_T'][:,1],reference['commanded_q'],
                        dt=float(reference['dt'])*factor,joint_limits=np.asarray(c['limits'])[:,:2])
                    target=np.asarray(case['target'] if case['target'] is not None else reference['desired_T'][-1,0])
                    distance=np.linalg.norm(result['T'][:,0]-target,axis=-1)*1000
                    normal=np.asarray([v['normal'] for v in recorder.rows])[:,2:]
                    depth=np.asarray([v['object_gripper_depth'] for v in recorder.rows])
                    track=np.linalg.norm(result['T'][:,1]-reference['desired_T'][:,1],axis=-1)*1000
                    key=f'{case["key"]}_duration{factor}_step{engine.model.opt.timestep:g}'
                    np.savez_compressed(out/(key+'.npz'),**result)
                    row=dict(key=case['key'],kind=case['kind'],duration_factor=factor,step=float(engine.model.opt.timestep),
                        final_goal_mm=float(distance[-1]),held_worst_goal_mm=float(distance[-6:].max()),
                        final_lift_mm=float((result['T'][-1,0,2]-result['T'][0,0,2])*1000),
                        peak_lift_mm=float((result['T'][:,0,2]-result['T'][0,0,2]).max()*1000),
                        hand_tracking_max_mm=float(track.max()),peak_overlap_mm=float(depth.max()*1000),
                        both_fingers_force_fraction=float((normal>.01).all(-1).mean()),
                        warnings=result['warning_counts'].tolist(),trajectory=key+'.npz',
                        trajectory_sha256=file_sha256(out/(key+'.npz')))
                    report['rows'].append(row);save();print(json.dumps(row),flush=True)
        report['status']='passed'
    except Exception as error:
        report.update(status='failed',error=repr(error));raise
    finally:save()
    lines=['# Fixed-command timing diagnostic','',report['interpretation'],'',
        'Normal-speed replay of every generated plan and recorded control must match the saved '
        'poses and joints bitwise before retiming is attempted. Finger commands, including '
        'the original preload program, are frozen. The simulator step and physical initial '
        'velocities stay unchanged; only time between command waypoints doubles from 50 to 100 ms.', '',
        '| Case | Duration factor | Step (ms) | Final goal error | Worst held error | Peak / final lift | Peak overlap |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in report['rows']:
        lines.append(f'| {r["key"]} | {r["duration_factor"]} | {r["step"]*1000:g} | {r["final_goal_mm"]:.1f} mm | '
            f'{r["held_worst_goal_mm"]:.1f} mm | {r["peak_lift_mm"]:.1f} / {r["final_lift_mm"]:.1f} mm | {r["peak_overlap_mm"]:.2f} mm |')
    lines+=['', 'Recorded controls target their recorded final object position and only calibrate '
        'the diagnostic. Their recorded future commands are never supplied to generated plans. '
        'Twice the command duration also doubles the six-frame hold duration. No new complete-task '
        'pass is inferred from these endpoint measurements. Recorded lift controls are still '
        'moving during their last six frames; their distance-to-final-position over that interval '
        'is descriptive and is not a held-goal qualification test. '
        '[All checks and numeric results](timing_probe/report.json) · '
        '[Exact diagnostic source](timing_probe/source_snapshot.py).', '']
    (root/'TIMING_PROBE.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
