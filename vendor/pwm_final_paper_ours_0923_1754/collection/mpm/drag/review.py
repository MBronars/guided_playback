"""Independent CPU review of real drag raws, commands, loaders and parity."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from collection.common import digest,write_once
from ..export import POLICY,export_candidates,validate_candidate,eligible
from ..validate import read_raw
from ..wall_materials import SAND_ONLY,STRICT
from ..walls import FIELDS
from .protocol import SEQUENCE_SCHEMAS


def wall_policy(manifest):
    return STRICT if manifest.get('action_schema') in SEQUENCE_SCHEMAS else SAND_ONLY


def compare_events(a,b):
    if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=5 or np.any(a < -1) or np.any(b < -1):
        raise ValueError('Invalid wall first-contact arrays')
    same=bool(np.array_equal(a>=0,b>=0));joint=(a>=0)&(b>=0)
    maximum=int(np.abs(a[joint]-b[joint]).max(initial=0))
    return dict(exact=bool(np.array_equal(a,b)),contact_presence_equal=same,maximum_step_difference=maximum)


def compare(reference,candidate):
    mr,a=read_raw(reference,POLICY);mc,b=read_raw(candidate,POLICY)
    if mr['case']!=mc['case']:raise ValueError('Not the same prepared scene')
    for key in ('object_ids','particle_volume','E','nu','density','mat_type','yield_stress',
                'selected','command_velocity','transition_active','active_substeps','state_time_s'):
        np.testing.assert_array_equal(a[key],b[key])
    for key in ('settle_x','settle_v','settle_F_trial'):
        np.testing.assert_array_equal(a[key][0],b[key][0])
    metrics={};events=[]
    for prefix in ('settle_',''):
        delta=a[prefix+'x'].astype(float)-b[prefix+'x'].astype(float)
        metrics[prefix+'position_rms_m']=float(np.sqrt(np.mean(np.sum(delta**2,axis=-1))))
        metrics[prefix+'maximum_particle_difference_m']=float(np.linalg.norm(delta,axis=-1).max())
        for key in ('wall_covered_substeps','wall_simulation_steps'):
            np.testing.assert_array_equal(a[prefix+key],b[prefix+key])
        for key in ('wall_first_grid_step','wall_first_particle_step'):
            events.append(compare_events(a[prefix+key],b[prefix+key]))
    left=validate_candidate(reference,wall_policy=wall_policy(mr));right=validate_candidate(candidate,wall_policy=wall_policy(mc))
    same=eligible(left)==eligible(right) and set(left['reasons'])==set(right['reasons'])
    # Predeclared vector RMS <=0.1mm in BOTH phases; contact presence unchanged,
    # native first-contact timing at most two steps. No mean-only selection waiver.
    passed=(max(metrics['position_rms_m'],metrics['settle_position_rms_m'])<=1e-4 and same
            and left['integrity_pass'] and right['integrity_pass']
            and all(e['contact_presence_equal'] and e['maximum_step_difference']<=2 for e in events))
    return dict(passed=bool(passed),metrics=metrics,events=events,selection_equal=same,
                reference_sha256=digest(reference),candidate_sha256=digest(candidate))


def render(raws,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,axes=plt.subplots(2,len(raws),figsize=(4*len(raws),7),squeeze=False,dpi=100)
    artists=[];values=[];summaries=[]
    for col,raw in enumerate(raws):
        m,a=read_raw(raw,POLICY);values.append((m,a));selected=a['selected'].astype(bool)
        # Equal metric scales; limits fixed over the entire clip, no camera tracking.
        low=a['x'].min((0,1))-.025;high=a['x'].max((0,1))+.025
        center=(low+high)/2;width=max(high-low)
        color=np.array(['#277fba','#cb4a93','#d39612'])[a['mat_type'][a['object_ids']]]
        for row,(u,v) in enumerate(((0,1),(0,2))):
            ax=axes[row,col];ax.set_aspect('equal')
            ax.set(xlim=(center[u]-width/2,center[u]+width/2),ylim=(center[v]-width/2,center[v]+width/2),
                   xlabel='x (m)',ylabel=('y (m)' if row==0 else 'z (m)'))
            if row==1:ax.axhline(.1,color='black',lw=1)
            dots=ax.scatter(a['x'][0,:,u],a['x'][0,:,v],s=3,c=color)
            patch=ax.scatter(a['x'][0,selected,u],a['x'][0,selected,v],s=8,c='black')
            artists.append((col,u,v,dots,patch,ax))
        delta=np.diff(a['x'].astype(float),axis=0);cmd=a['command_velocity'].astype(float)*np.diff(a['state_time_s'])[:,None,None]
        summaries.append(dict(raw=str(raw),raw_sha256=digest(raw),family=m['action']['family'],
            selected=int(selected.sum()),selected_path_mean_m=float(np.linalg.norm(delta[:,selected],axis=-1).sum(0).mean()),
            commanded_path_mean_m=float(np.linalg.norm(cmd[:,selected],axis=-1).sum(0).mean()),
            max_selected_excursion_m=float(np.linalg.norm(a['x'][:,selected]-a['x'][0,selected],axis=-1).max())))
    fig.tight_layout(rect=(0,.07,1,.93));fig.subplots_adjust(hspace=.48);heading=fig.suptitle('')
    fig.text(.5,.025,'Actual saved particles: XY above / XZ below. Black = selected material points, not a robot.\n'
             'Blue elastic / pink plasticine / gold sand. 2x slow; metric axes fixed per clip. Diagnostic, not scale approval.',ha='center',fontsize=9)
    frames=[]
    frame_count=max(len(a['x']) for _,a in values)
    for frame in range(frame_count):
        heading.set_text(f"MPM drag | t={frame*.0417:.4f} s (shorter clips freeze at end)")
        for col,u,v,dots,patch,ax in artists:
            m,a=values[col];k=min(frame,len(a['x'])-1);selected=a['selected'].astype(bool);p=a['x'][k]
            dots.set_offsets(p[:,[u,v]]);patch.set_offsets(p[selected][:,[u,v]])
            gate=bool(a['transition_active'][k]) if k<len(a['transition_active']) else False
            phase='HOLD' if gate and not a['command_velocity'][k].any() else 'DRIVE' if gate else 'RELEASE'
            events=m['action'].get('events',[])
            description=m['action']['family'] if not events else f"{len(events)} events / "+','.join(e['family'].split('_')[0] for e in events)
            ax.set_title(f"{Path(raws[col]).parent.name}: {description}\n{phase} (outgoing transition)",fontsize=9)
        fig.canvas.draw();frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    frames[0].save(output,save_all=True,append_images=frames[1:],duration=83,loop=0,optimize=False)
    frames[20].save(output.with_suffix('.png'));plt.close(fig)
    with Image.open(output) as im:
        if im.n_frames!=frame_count:raise ValueError('Missing rendered frames')
    write_once(output.with_suffix('.json'),dict(gif_sha256=digest(output),frames=frame_count,rows=summaries))
    return summaries


def normalization_review(stats, actions):
    """Check the historical affine formulas, not an imaginary exact inverse.

    normalize divides by (scale + 1e-8); denormalize multiplies by scale.
    Compare both stages to independent float64 arithmetic. Eight float32 eps
    times the participating magnitudes bounds the short affine calculation's
    roundoff, including cancellation around zero. Raw commands are checked
    bit-for-bit separately; this bound is NOT a simulation/command tolerance.
    """
    import torch
    values = (actions, stats.min_val, stats.max_val, stats.scale)
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in values):
        raise ValueError('Normalization review requires finite float32 values')
    if not (stats.scale > 0).all():raise ValueError('Nonpositive normalization scale')
    x = actions.double();lo = stats.min_val.to(actions.device).double()
    scale = stats.scale.to(actions.device).double();eps = torch.finfo(torch.float32).eps
    expected = 2 * (x - lo) / (scale + 1e-8) - 1
    normalized = stats.normalize(actions)
    forward_bound = 8 * eps * (1 + expected.abs())
    if (not torch.isfinite(normalized).all()
            or not ((normalized.double() - expected).abs() <= forward_bound).all()):
        raise AssertionError('Normalization differs from affine float64 oracle')
    back = stats.denormalize(normalized)
    expected_back = lo + (x - lo) * scale / (scale + 1e-8)
    backward_bound = 8 * eps * ((x - lo).abs() + lo.abs() + scale)
    if (not torch.isfinite(back).all()
            or not ((back.double() - expected_back).abs() <= backward_bound).all()):
        raise AssertionError('Denormalization differs from affine float64 oracle')
    return dict(policy='affine-float64-oracle-fp32-v1',
        maximum_roundtrip_error_m=float((back.double()-x).abs().max()),
        maximum_oracle_error_m=float((back.double()-expected_back).abs().max()),
        maximum_oracle_bound_m=float(backward_bound.max()))


def loader_review(export,raws):
    import torch
    from data.mpm_drag import UnifiedMPMDragDataset,commands
    from data.unified_dataset import UnifiedMPMDataset,CANONICAL_TRANSLATION
    expected={read_raw(raw,POLICY)[0]['episode_id']:raw for raw in raws};rows=[]
    for profile in ('xy_p99_per_domain','xy_p99_global'):
        ds=UnifiedMPMDragDataset(str(export),normalization=profile,context_len=4,future_len=16,training=False,shuffle=False)
        ds.strict_read=True;ds.no_shard_split=True;seen=set();normalization=[]
        for sample in ds._create_pipeline():
            batch=ds._process_sample(sample);delta,gate,selection=commands(sample)
            for i,(row,start) in enumerate(zip(sample['provenance.json'],ds._window_starts)):
                raw=expected[row['episode_id']]
                if row['episode_id'] in seen or digest(raw)!=row['raw_sha256']:raise ValueError('Duplicate/changed raw')
                seen.add(row['episode_id']);_,a=read_raw(raw,POLICY)
                positions=torch.from_numpy(a['x'][start:start+20])+torch.tensor(CANONICAL_TRANSLATION['mpm'])
                torch.testing.assert_close(batch.positions[i],positions,atol=0,rtol=0)
                torch.testing.assert_close(batch.actions[i],delta[i,start+3:start+19],atol=0,rtol=0)
                torch.testing.assert_close(batch.action_active[i],gate[i,start+3:start+19],atol=0,rtol=0)
                normalization.append(normalization_review(batch.delta_stats,batch.actions[i]))
            try:UnifiedMPMDataset._validate_command_input(ds,sample)
            except ValueError:pass
            else:raise ValueError('Passive loader accepted drag')
        if seen!=set(expected):raise ValueError('Loader coverage mismatch')
        rows.append(dict(profile=profile,episodes=len(seen),exact_raw_window_and_command_match=True,
            normalization_oracle=dict(policy='affine-float64-oracle-fp32-v1',
                maximum_roundtrip_error_m=max(r['maximum_roundtrip_error_m'] for r in normalization),
                maximum_oracle_error_m=max(r['maximum_oracle_error_m'] for r in normalization))))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pilot',required=True)
    p.add_argument('--output',required=True);p.add_argument('--reference')
    args=p.parse_args();root=Path(args.pilot);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=json.loads((root/'COMPLETE.json').read_text());raws=[Path(r['raw_path']) for r in result['results'] if 'raw_path' in r]
    comparisons=[]
    if args.reference:
        for raw in raws:comparisons.append(dict(label=raw.parent.name,**compare(Path(args.reference)/raw.parent.name/'raw.h5',raw)))
    selection=export_candidates(raws,out/'filtered',wall_policy=wall_policy(read_raw(raws[0],POLICY)[0])) if raws else dict(retained=0)
    retained=[Path(r['raw_path']) for r in result['results'] if r['retained']]
    loader=loader_review(out/'filtered/diagnostic-export',retained) if retained else []
    visuals=[]
    for offset in range(0,len(raws),3):visuals.extend(render(raws[offset:offset+3],out/f'drag-{offset:03d}.gif'))
    report=dict(comparisons=comparisons,selection=selection,loader=loader,visuals=visuals,scaling_approved=False)
    write_once(out/'REVIEW.json',report)
    print(json.dumps(dict(output=str(out),retained=len(retained),comparison_passes=[r['passed'] for r in comparisons],loader=loader)),flush=True)


if __name__=='__main__':main()
