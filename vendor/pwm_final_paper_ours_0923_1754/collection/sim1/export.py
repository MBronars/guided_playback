"""Fail-closed SIM1 export shape/order/time checks and per-file accounting.

Legacy nominal fps is not evidence of a complete recorded physical timeline.
Schema-valid legacy records remain diagnostic, not newly accepted training data.
"""
from __future__ import annotations
import numpy as np
from .contracts import finite_array,require_link_membership,unit_quaternions


def outgoing_state_deltas(arrays):
    """Derive deltas AFTER selecting retained states, never slice old deltas.

    This helper implements the existing padded-T export convention explicitly.
    The terminal entry is not an observed transition. It returns fresh arrays
    plus that validity mask, and never changes the source recording or infers
    that command joint positions are achieved physical states.
    """
    joint=finite_array(arrays['joint_q'],'joint_q',(None,None))
    T=len(joint)
    if T<2:raise ValueError('At least two retained states required')
    output={}
    for delta,base in [('delta_joint_q','joint_q'),('delta_openness','openness'),
                       ('delta_left_ee_pos','left_ee_pose'),('delta_right_ee_pos','right_ee_pose')]:
        x=finite_array(arrays[base],base,(T,None))
        if base.endswith('_pose'):
            if x.shape[1]!=7:raise ValueError('XYZ/XYZW poses required')
            x=x[:,:3]
        elif base=='openness' and x.shape[1]!=2:raise ValueError('Both jaw commands required')
        output[delta]=np.concatenate((np.diff(x,axis=0),np.zeros_like(x[:1])),axis=0)
        if not np.isfinite(output[delta]).all():raise ValueError('Derived delta overflow')
    for side in ('left','right'):
        q=unit_quaternions(np.asarray(arrays[side+'_ee_pose'])[:,3:])
        # q_next * inverse(q_previous): world-frame relative rotation, XYZW.
        v1,w1=q[1:,:3],q[1:,3:];v0,w0=-q[:-1,:3],q[:-1,3:]
        delta=np.concatenate((w1*v0+w0*v1+np.cross(v1,v0),w1*w0-np.sum(v1*v0,axis=1,keepdims=True)),axis=1)
        delta=np.where(delta[:,3:]<0,-delta,delta)
        output['delta_'+side+'_ee_quat']=np.concatenate((delta,[[0.,0.,0.,1.]]),axis=0)
    output['transition_valid']=np.arange(T)<T-1
    return output


def check_export(arrays,expected_ee_points=1000):
    required={'cloth_points','left_ee_points','right_ee_points','ee_link_ids',
              'joint_q','openness','left_ee_pose','right_ee_pose','fps','table_top_z'}
    missing=required-set(arrays)
    if missing: raise ValueError('MISSING_FIELDS:'+','.join(sorted(missing)))
    q=finite_array(arrays['cloth_points'],'cloth_points',(None,None,3))
    T,N=q.shape[:2]
    if T<2 or N<1: raise ValueError('EMPTY_OR_TOO_SHORT_EPISODE')
    for name in ('left_ee_points','right_ee_points'):
        finite_array(arrays[name],name,(T,expected_ee_points,3))
    counts=require_link_membership(arrays['ee_link_ids'],expected_ee_points)
    for name in ('left_ee_pose','right_ee_pose'):
        pose=finite_array(arrays[name],name,(T,7));unit_quaternions(pose[:,3:],name)
    joint=finite_array(arrays['joint_q'],'joint_q',(T,None))
    if joint.shape[1]<1: raise ValueError('EMPTY_JOINT_STATE')
    op=finite_array(arrays['openness'],'openness',(T,2))
    if np.any((op<0)|(op>1)): raise ValueError('OPENNESS_OUT_OF_RANGE')
    fps=finite_array(arrays['fps'],'fps',())
    if float(fps)<=0: raise ValueError('NONPOSITIVE_FPS')
    finite_array(arrays['table_top_z'],'table_top_z',())
    checks=['required_numeric_schema','finite_arrays','all_frames_same_shape',
            'ordered_link_membership','unit_pose_quaternions','openness_range']
    for name,source in [('delta_joint_q','joint_q'),('delta_openness','openness'),
                        ('delta_left_ee_pos','left_ee_pose'),('delta_right_ee_pos','right_ee_pose')]:
        if name not in arrays:continue
        base=np.asarray(arrays[source])
        if source.endswith('_pose'):base=base[:,:3]
        delta=finite_array(arrays[name],name,base.shape)
        expected=np.concatenate((np.diff(base,axis=0),np.zeros_like(base[:1])),axis=0)
        if not np.allclose(delta,expected,rtol=1e-5,atol=1e-6):
            raise ValueError(f'DELTA_STATE_CONTRACT_MISMATCH:{name}')
        checks.append(name+'_matches_recorded_state_difference')
    for side in ('left','right'):
        name='delta_'+side+'_ee_quat'
        if name not in arrays:continue
        delta=unit_quaternions(finite_array(arrays[name],name,(T,4)),name)
        expected=outgoing_state_deltas(arrays)[name]
        if not np.allclose(np.abs(np.sum(delta*expected,axis=1)),1,rtol=0,atol=1e-7):
            raise ValueError(f'DELTA_STATE_CONTRACT_MISMATCH:{name}')
        checks.append(name+'_matches_world_relative_rotation')
    if 'transition_valid' in arrays:
        valid=np.asarray(arrays['transition_valid'])
        if valid.dtype.kind!='b' or not np.array_equal(valid,np.arange(T)<T-1):
            raise ValueError('INVALID_TERMINAL_TRANSITION_MASK')
        checks.append('terminal_padding_explicitly_invalid')
    time_status='legacy_nominal_fps_only_actual_time_alignment_unverified'
    if 'state_time_s' in arrays:
        times=finite_array(arrays['state_time_s'],'state_time_s',(T,))
        indices=finite_array(arrays['source_state_index'],'source_state_index',(T,))
        if indices.dtype.kind not in 'iu' or np.any(np.diff(indices)<=0) or np.any(np.diff(times)<=0):
            raise ValueError('INVALID_EXPLICIT_TIME_OR_FRAME_IDENTITY')
        if not np.allclose(np.diff(times),1/float(fps),rtol=1e-6,atol=1e-10):
            raise ValueError('FPS_AND_ACTUAL_SAVED_TIMES_DISAGREE')
        time_status='explicit_monotone_uniform_times_present_needs_raw_correspondence'
        checks.append('explicit_time_shape_monotonicity_and_nominal_cadence')
    return dict(schema_pass=True,frames=T,cloth_particles=N,ee_particles=expected_ee_points,
                link_counts=counts.tolist(),checks=checks,time_status=time_status,
                accepted_for_training=False,pending=['raw/export correspondence',
                    'source geometry/sewing', 'physics/refinement/settling', 'task success'])


def qualify_files(paths, *, expected_ee_points=1000):
    """Every requested episode gets a receipt, including read/schema failures."""
    paths=list(paths)
    if not paths: raise ValueError('No episodes to qualify')
    if len(set(map(str,paths)))!=len(paths): raise ValueError('Duplicate input path in qualification request')
    rows=[]
    for path in paths:
        try:
            with np.load(path,allow_pickle=False) as data:
                result=check_export(data,expected_ee_points)
            rows.append(dict(path=str(path),status='diagnostic',**result))
        except Exception as error:
            rows.append(dict(path=str(path),status='rejected',schema_pass=False,
                             reason_codes=['EXPORT_READ_OR_SCHEMA_FAILURE'],detail=str(error),
                             accepted_for_training=False))
    return dict(requested=len(paths),examined=len(rows),rows=rows,
                schema_pass=sum(r['schema_pass'] for r in rows),
                rejected=sum(not r['schema_pass'] for r in rows),
                accepted_for_training=0)
