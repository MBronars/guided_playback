"""Training-only approval from preserved qualification evidence; no GPU rerun.

Never changes the failed 1–7-object verdict. Approves only the pre-existing
1–4-object TRAINING strata, on the same two physical GPUs, with unchanged code.
"""
import math
import os
from pathlib import Path
import mpm_scale as support

SCOPE='train-1-to-4-only-v1'


def training_indices(selected):
    ids=[i for i,s in enumerate(selected) if s['case']['template']['split']=='train']
    if len(ids)!=20:raise ValueError('Require all twenty original training strata')
    for count in range(1,5):
        rows=[selected[i]['case'] for i in ids if selected[i]['case']['template']['n_objects']==count]
        if len(rows)!=5 or {c['provenance']['stiffness_phase'] for c in rows}!={0,1,2}:
            raise ValueError('Training count/stiffness coverage changed')
    return ids


def same_sources(current,previous):
    names={n for n in current['files'] if n.startswith('source/')}
    if names!={n for n in previous['files'] if n.startswith('source/')}:
        raise ValueError('Numerical/consumer source inventory changed')
    if any(current['files'][n]!=previous['files'][n] for n in names):
        raise ValueError('Cannot reuse qualification across changed simulator/filter/consumer source')
    for k in ('banks','bank_complete_sha256','reference'):
        if current[k]!=previous[k]:raise ValueError('Data/reference lineage changed')


def verify_attempt(folder,case):
    from collection.mpm.journal import verify_journal
    result=support.read(folder/'RESULT.json')
    if (result['status']!='complete' or result['initial_sha256']!=case['initial_sha256']
            or result['raw_sha256']!=support.sha(folder/'raw.h5')):
        raise ValueError('Changed/incomplete qualification trajectory')
    journal=verify_journal(folder/'frames',initial_sha256=case['initial_sha256'],
        manifest_sha256=case['manifest_sha256'],complete=True,raw_path=folder/'raw.h5')
    if journal!=result['journal']:raise ValueError('Qualification journal changed')


def certify(bundle,root,evidence,evidence_bundle):
    import mpm_four as four
    from collection.common import check_hashes
    from collection.mpm.export import export_candidates
    from collection.mpm.wall_materials import SAND_ONLY
    from collection.mpm.review import check_model
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only certificate review')
    current=four.verify(bundle);old=support.read(evidence_bundle/'BUNDLE.json')
    if not current['training_only'] or current['gpu_count']!=2:raise ValueError('Training-only two-GPU bundle required')
    check_hashes(evidence_bundle,old['files']);same_sources(current,old)
    request=support.read(evidence/'REQUEST.json')
    if request['bundle_sha256']!=support.sha(evidence_bundle/'BUNDLE.json'):
        raise ValueError('Original qualification source mismatch')
    selected=request['cases'];ids=training_indices(selected)
    if len(request['uuids'])!=2 or len(set(request['uuids']))!=2:raise ValueError('Two distinct qualified GPUs required')
    root.mkdir(parents=True,exist_ok=False)
    mode=four.QUALIFICATION_MODES[0];source_files={}
    def bind_file(path):source_files[str(path.resolve())]=support.sha(path)
    for f in (evidence/'REQUEST.json',evidence_bundle/'BUNDLE.json'):bind_file(f)
    # Preserve the full failure as explicit evidence, not a retroactive full PASS.
    for f in evidence.glob('*PASS.json'):bind_file(f)
    for f in evidence.glob('FAILURE.json'):bind_file(f)
    lanes=[];reviews=[]
    for lane in range(2):
        unit=evidence/f'lane-{lane}'
        isolation=support.read(unit/'ISOLATION.json')
        if isolation['grid']!={'grid_cells':1000000,'mutation_rejected':True,'velocity_and_events_bit_exact':True}:
            raise ValueError('Grid isolation test failed')
        for name in ('group4','launch64','fused4'):
            if isolation['failure_injections'][name]!={'healthy_world_unchanged':True,'nonfinite_stopped':True,'physics_steps':0}:
                raise ValueError('Failure isolation test failed')
        for name in ('ISOLATION.json','RESULT.json','REPEATABILITY.json'):bind_file(unit/name)
        original=next(r for r in support.read(unit/'RESULT.json')['results'] if r['mode']==mode)
        if not math.isfinite(original['speedup']) or original['speedup']<1.05:raise ValueError('No measured speed benefit')
        comparisons=[];repeats=[];raws=[]
        for i in ids:
            case=selected[i]['case'];input_dir=Path(selected[i]['root'])/'cases'/case['label']
            for name,key in [('initial.npz','initial_sha256'),('MANIFEST.json','manifest_sha256')]:
                f=input_dir/name
                if support.sha(f)!=case[key]:raise ValueError('Prepared input changed')
                bind_file(f)
            reference=unit/'serial'/f'{i:02d}';candidate=unit/mode['name']/f'{i:02d}'
            folders=[reference,candidate]
            for folder in folders:verify_attempt(folder,case)
            c=four.compare(reference/'raw.h5',candidate/'raw.h5')
            if not c['passed']:raise ValueError(f'Training comparison failed: lane {lane}, case {i}')
            comparisons.append(dict(case=i,**c));raws.append(candidate/'raw.h5')
            if i in four.SERIAL_REPEAT_CASES:
                for repeat in range(four.SERIAL_REPEATS):
                    folder=unit/'serial-repeat'/f'{i:02d}-{repeat}';verify_attempt(folder,case);folders.append(folder)
                    a=four.compare(reference/'raw.h5',folder/'raw.h5')
                    b=four.compare(folder/'raw.h5',candidate/'raw.h5')
                    if not a['passed'] or not b['passed']:raise ValueError('Training repeatability failed')
                    repeats.append(dict(case=i,repeat=repeat,serial=a,candidate=b))
            for folder in folders:
                for f in folder.rglob('*'):
                    if f.is_file():bind_file(f)
            print(f'Training certificate: GPU {lane}, case {i} passed',flush=True)
        consumer=root/f'consumer-{lane}'
        export_candidates(raws,consumer/'filtered',wall_policy=SAND_ONLY)
        rows=support.read(consumer/'filtered/FILTER.json')['episodes']
        kept={r['episode_id']:Path(r['raw_path']) for r in rows if r['eligible_for_export']}
        if not kept:raise ValueError('No retained training qualification cases')
        reviews.append(check_model(consumer,kept))
        lanes.append(dict(uuid=request['uuids'][lane],comparisons=comparisons,repeats=repeats,
                          full_matrix_diagnostic_speedup=original['speedup']))
    for path,digest in source_files.items():
        if support.sha(path)!=digest:raise ValueError('Evidence changed during review')
    support.write(root/'TRAINING_PASS.json',dict(scope=SCOPE,gpu_count=2,mode=mode,indices=ids,
        bundle_sha256=support.sha(bundle/'BUNDLE.json'),evidence=str(evidence),evidence_bundle=str(evidence_bundle),
        lanes=lanes,consumer_reviews=reviews,source_files=source_files,
        files={str(p.relative_to(root)):support.sha(p) for p in root.rglob('*') if p.is_file()},
        evaluation_approved=False,accepted_for_training=False))


def qualified(root,bundle):
    import mpm_four as four
    from collection.common import check_hashes
    r=support.read(root/'TRAINING_PASS.json')
    if (r['scope']!=SCOPE or r['gpu_count']!=2 or r['evaluation_approved'] is not False
            or r['bundle_sha256']!=support.sha(bundle/'BUNDLE.json') or r['mode']!=four.QUALIFICATION_MODES[0]):
        raise ValueError('Wrong scoped training certificate')
    check_hashes(root,r['files'])
    for path,digest in r['source_files'].items():
        if support.sha(path)!=digest:raise ValueError('Bound qualification evidence changed')
    selected=support.read(Path(r['evidence'])/'REQUEST.json')['cases'];ids=training_indices(selected)
    if r['indices']!=ids or len(r['lanes'])!=2:raise ValueError('Training scope/lanes changed')
    for lane in r['lanes']:
        if [c['case'] for c in lane['comparisons']]!=ids:raise ValueError('Missing training comparison')
        if not all(c['passed'] and four.comparison_passes(c) for c in lane['comparisons']):raise ValueError('Training comparison failed')
        expected=[(i,j) for i in ids if i in four.SERIAL_REPEAT_CASES for j in range(four.SERIAL_REPEATS)]
        if [(c['case'],c['repeat']) for c in lane['repeats']]!=expected:raise ValueError('Missing training repeats')
        for row in lane['repeats']:
            if not all(row[k]['passed'] and four.comparison_passes(row[k]) for k in ('serial','candidate')):
                raise ValueError('Training repeat failed')
    return r
