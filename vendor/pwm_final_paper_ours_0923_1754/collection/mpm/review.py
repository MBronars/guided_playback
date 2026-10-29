"""Independent saved-state, export, actual-loader and visual pilot review (CPU)."""
import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

from collection.common import canonical, check_hashes, digest, write_once
from .collect import inspect, verify_attempt
from .runtime import ROOT, STATIC
from .recipes import INTERACTION_VERSIONS


def read_and_check(root, case, attempt=0):
    folder = root / 'cases' / case['label']
    run = folder / 'attempts' / f'attempt-{attempt:05d}'
    if not (run / 'RESULT.json').exists():
        return dict(label=case['label'], kind=case['kind'], status='unrun_or_incomplete'), None
    result = verify_attempt(run)
    if result['status'] != 'complete': return result, None
    from .export import POLICY, validate_wall_free
    from .validate import read_raw
    from .contracts import Clock
    from .walls import validate_evidence
    raw = run / 'raw.h5'
    manifest, arrays = read_raw(raw, POLICY)
    with np.load(folder / 'initial.npz', allow_pickle=False) as initial:
        for key in ('x', 'v'): np.testing.assert_array_equal(arrays[key][0], initial[key])
        for key in STATIC: np.testing.assert_array_equal(arrays[key], initial[key])
    np.testing.assert_array_equal(np.bincount(arrays['object_ids']), [1024] * manifest['n_objects'])
    np.testing.assert_array_equal(arrays['state_time_s'], Clock(1e-4, 417, 48).state_time_s)
    evidence = validate_evidence(manifest, arrays)
    quality = validate_wall_free(raw)
    if quality != result['quality']: raise ValueError('Validation verdict changed')
    delta = np.linalg.norm(np.diff(arrays['x'].astype(float), axis=0), axis=-1).mean(axis=1)
    window_motion = [float(delta[start + 3:start + 19].mean()) for start in range(30)]
    row = dict(label=case['label'], kind=case['kind'], status=result['status'],
               scenario=manifest.get('scenario'), count=manifest['n_objects'],
               materials=manifest['material_names'], quality=quality, raw_path=str(raw), raw_sha256=digest(raw),
               source_and_clock_and_arrays_checked=True, observed_substeps=evidence['covered_substeps'],
               mean_point_step_mm=float(delta.mean()*1000),
               future_window_point_step_mm=[v*1000 for v in window_motion], wall_seconds=result['wall_seconds'])
    if manifest.get('geometry_provenance', {}).get('recipe_version') in INTERACTION_VERSIONS:
        from .interactions import analyze
        row['interaction'] = analyze(manifest, arrays)
    if case['kind'] == 'matched_control':
        reference = Path(case['provenance']['reference_raw'])
        if digest(reference) != case['provenance']['reference_raw_sha256']: raise ValueError('Historical control raw changed')
        _, old = read_raw(reference, POLICY)
        for key in STATIC + ('state_time_s',): np.testing.assert_array_equal(arrays[key], old[key])
        for key in ('x', 'v'): np.testing.assert_array_equal(arrays[key][0], old[key][0])
        differences = arrays['x'].astype(float) - old['x'].astype(float)
        row['reference_position_rms_mm'] = float(np.sqrt(np.mean(np.sum(differences**2, axis=-1)))*1000)
        row['reference_final_position_rms_mm'] = float(np.sqrt(np.mean(np.sum(differences[-1]**2, axis=-1)))*1000)
        expected = case['provenance']['expected_wall_free']
        row['functionality_control_pass'] = bool(quality['integrity_pass'] and quality['eligible_for_wall_free_export'] == expected)
        if not row['functionality_control_pass']: raise ValueError('Pinned wall functionality control failed')
    return row, arrays


def check_model(export, raw_paths):
    from collection.consumer import import_consumer
    clean, unified = import_consumer()
    import torch
    from data.normalization import resolve_normalization
    from .contracts import require_passive_consumer_input
    torch.set_num_threads(1)
    if Path(unified.__file__).resolve() != clean / 'data/unified_dataset.py': raise ValueError('Wrong loader import')
    sources = {str(clean / name): digest(clean / name) for name in ('data/unified_dataset.py','data/dataset.py',
               'data/batch.py','data/normalization.py','data/blocklists.py','stats/normalization.json')}
    rows = []
    for profile in ('xy_p99_per_domain', 'xy_p99_global'):
        norm = resolve_normalization(profile)
        dataset = unified.UnifiedMPMDataset(str(export / 'filtered/diagnostic-export'), normalization=norm,
                                            training=False, shuffle=False, context_len=4, future_len=-1)
        dataset.strict_read = True
        seen = set()
        for sample in dataset._create_pipeline():
            require_passive_consumer_input(sample)
            lineage = sample['provenance.json']
            actual = []
            for record in lineage:
                eid = record['episode_id']
                if eid not in raw_paths or eid in seen: raise ValueError('Excluded/duplicate model row')
                if digest(raw_paths[eid]) != record['raw_sha256']: raise ValueError('Model row raw changed')
                with h5py.File(raw_paths[eid], 'r') as f:
                    actual.append({k:f[k][()] for k in ('x', 'object_ids', 'E', 'nu', 'density', 'mat_type', 'state_time_s')})
                np.testing.assert_array_equal(record['state_time_s'], actual[-1]['state_time_s'])
                seen.add(eid)
            expected = torch.from_numpy(np.stack([a['x'] for a in actual])) + torch.tensor(unified.CANONICAL_TRANSLATION['mpm'])
            dataset.future_len = -1
            batch = dataset._process_sample(sample)
            torch.testing.assert_close(batch.positions, expected, rtol=0, atol=0)
            np.testing.assert_array_equal(batch.object_ids.numpy(), np.stack([a['object_ids'] for a in actual]))
            for field, key in [('E','E'), ('nu','nu'), ('rho','density'), ('mat_type','mat_type')]:
                np.testing.assert_array_equal(getattr(batch, field).numpy(), np.stack([a[key][a['object_ids']] for a in actual]))
            if batch.actions.any() or batch.action_mask.any(): raise ValueError('Passive actions changed')
            torch.testing.assert_close(batch.future_deltas, expected[:,4:] - expected[:,3:-1], rtol=0, atol=0)
            dataset.future_len = 16; dataset.first_window = True
            window = dataset._process_sample(sample)
            torch.testing.assert_close(window.positions, expected[:,:20], rtol=0, atol=0)
            normalized = window.delta_stats.normalize(window.future_deltas)
            if not torch.isfinite(normalized).all(): raise ValueError('Nonfinite normalized target')
            rows.append(dict(profile=profile, norm_sha256=norm['sha256'], episodes=[v['episode_id'] for v in lineage],
                             full_shape=list(batch.positions.shape), window_shape=list(window.positions.shape)))
        if seen != set(raw_paths): raise ValueError('Missing model rows')
    if any(digest(p) != h for p,h in sources.items()): raise ValueError('Model source changed')
    return dict(profiles=2, episodes_per_profile=len(raw_paths), batches=rows, source_sha256=sources)


def render(items, out, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    columns = min(5, len(items)); rows = (len(items) + columns - 1)//columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns*3.3, rows*3+1), squeeze=False, dpi=100)
    colors = np.array(['#277fba','#cb4a93','#d39612'])
    artists = []
    for ax, (row, arrays) in zip(axes.flat, items):
        ax.set(xlim=(0,1.5), ylim=(0,1.5), xlabel='x (m)', ylabel='z (m)')
        ax.set_aspect('equal'); ax.axhline(.1, color='black', lw=.8)
        for x in (.03,1.47): ax.axvline(x, color='#a33', lw=.8)
        ax.axhline(1.47, color='#a33', lw=.8)
        ax.set_title(row['label'], fontsize=8)
        dots = ax.scatter(arrays['x'][0,:,0], arrays['x'][0,:,2], s=1,
                          c=colors[arrays['mat_type'][arrays['object_ids']]])
        title = ax.text(.02,.97,'',transform=ax.transAxes,va='top',fontsize=8)
        artists.append((dots,title))
    for ax in list(axes.flat)[len(items):]: ax.set_visible(False)
    fig.tight_layout(rect=(0,.08,1,.91))
    heading = fig.suptitle('', fontsize=12)
    fig.text(.5,.025,'Actual saved XZ projections (y hidden); all original points. 17/49 states, 3x slow.\nBlue elastic / pink plasticine / gold sand. Black floor; red side walls/ceiling. Not accepted training data.',ha='center',fontsize=9)
    frames=[]
    for k in range(0,49,3):
        heading.set_text(f'{name} | actual saved time {items[0][1]["state_time_s"][k]:.4f} s')
        for (row,a),(dots,label) in zip(items,artists):
            dots.set_offsets(a['x'][k][:,[0,2]])
            wall = np.any(a['wall_first_grid_step'][k]>=0) or np.any(a['wall_first_particle_step'][k]>=0)
            eligible = (row['quality']['eligible_for_wall_free_export']
                        and row.get('interaction', {}).get('semantic_eligible', True))
            final_label = ('eligible diagnostic' if eligible else 'integrity/semantics rejected')
            label.set_text('WALL — excluded' if wall else ('wall-free so far' if k<48 else final_label))
            label.set_color('#a33' if wall else '#174')
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    path = out / (name+'.gif')
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=380, loop=0, optimize=False)
    frames[-1].save(out / (name+'-last.png'))
    plt.close(fig)
    with Image.open(path) as im:
        for i in range(im.n_frames): im.seek(i); im.load()
        if im.n_frames != 17: raise ValueError('Incomplete GIF')
    return dict(path=path.name,sha256=digest(path),frames=17,source_frame_indices=list(range(0,49,3)),
                raw_sha256=[r['raw_sha256'] for r,a in items],projection='XZ; y hidden; equal metric axes')


def run(root, out, export=None, attempt=0):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only review')
    plan = inspect(root)
    out.mkdir(parents=True, exist_ok=False)
    rows, items, retained = [], [], {}
    for case in plan['cases']:
        row, arrays = read_and_check(root, case, attempt)
        rows.append(row)
        if arrays is not None:
            items.append((row, arrays))
            if (row['kind']=='coverage' and row['quality']['eligible_for_wall_free_export']
                    and row.get('interaction', {}).get('semantic_eligible', True)):
                with h5py.File(row['raw_path'],'r') as f: eid=json.loads(f.attrs['manifest_json'])['episode_id']
                retained[eid]=Path(row['raw_path'])
    model = None
    if export:
        complete=json.loads((export/'COMPLETE.json').read_text());check_hashes(export,complete['artifacts'])
        if complete['retained'] != len(retained): raise ValueError('Export count mismatch')
        if retained:
            with redirect_stdout(io.StringIO()): model=check_model(export,retained)
    media=[]
    controls=[item for item in items if item[0]['kind']=='matched_control']
    if controls: media.append(render(controls,out,'controls'))
    for count in range(1, 8):
        selected=[item for item in items if item[0]['kind']=='coverage' and item[0]['count']==count]
        for start in range(0, len(selected), 6):
            media.append(render(selected[start:start + 6],out,f'{count}obj-coverage-{start // 6:02d}'))
    strata=defaultdict(lambda:dict(completed=0,retained=0))
    for row in rows:
        if row['status']=='complete' and row['kind']=='coverage':
            for key in ('scenario:'+row['scenario'],'count:'+str(row['count'])):
                strata[key]['completed']+=1
                strata[key]['retained']+=int(row['quality']['eligible_for_wall_free_export']
                    and row.get('interaction', {}).get('semantic_eligible', True))
    result=dict(plan_sha256=digest(root/'PLAN.json'), rows=rows, model=model, media=media,
                strata=dict(strata),status_counts=dict(Counter(r['status'] for r in rows)),
                retained_coverage=len(retained),accepted_for_training=False)
    write_once(out/'RESULT.json',result)
    print(canonical(dict(status_counts=result['status_counts'],retained=len(retained),media=len(media))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--export',type=Path);p.add_argument('--attempt',type=int,default=0)
    args=p.parse_args();run(args.root.resolve(),args.out.resolve(),args.export.resolve() if args.export else None,args.attempt)
