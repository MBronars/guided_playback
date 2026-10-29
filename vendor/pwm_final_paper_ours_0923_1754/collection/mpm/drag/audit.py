"""Account for every planned scene, including rejected and unfinished cases."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from collection.common import digest,write_once
from ..journal import verify_journal
from .review import compare


def audit(root,output,*,partial=False):
    root=Path(root).resolve(strict=True);request=json.loads((root/'REQUEST.json').read_text())
    planned=Path(request['root']);plan=json.loads((planned/'PLAN.json').read_text())
    if digest(planned/'PLAN.json')!=request['plan_sha256']:raise ValueError('Matrix input plan changed')
    entries={c['label']:c for c in plan['cases']};rows=[]
    for index,labels in enumerate(request['groups']):
        for label in labels:
            entry=entries[label]
            for filename,key in (('initial.npz','initial_sha256'),('MANIFEST.json','manifest_sha256')):
                if digest(planned/'cases'/label/filename)!=entry[key]:raise ValueError('Prepared audit input changed')
            m=json.loads((planned/'cases'/label/'MANIFEST.json').read_text())
            row=dict(label=label,objects=m['n_objects'],family=m['action']['family'],
                     material_names=m['material_names'],asset_ids=m['case']['asset_ids'],modes={})
            with np.load(planned/'cases'/label/'initial.npz') as initial:row['E_Pa']=initial['E'].tolist()
            paths={}
            for mode in request['modes']:
                folder=root/mode/f'group-{index:03d}'/label;result=folder/'RESULT.json'
                if not result.is_file():
                    # v4 rejected an entire independent group when one member
                    # timed out. Preserve that failure, not a false unfinished
                    # label or a retroactively retained healthy scene.
                    group_result=folder.parent/'COMPLETE.json'
                    old=json.loads(group_result.read_text()) if group_result.is_file() else {}
                    if old.get('reason')=='settling' and len(old.get('per_world',[]))==len(labels):
                        value=dict(status='group_settling_rejected',retained=False)
                        row['modes'][mode]=dict(value,group_receipt_sha256=digest(group_result))
                    else:
                        row['modes'][mode]=dict(status='unfinished');continue
                else:
                    value=json.loads(result.read_text());row['modes'][mode]=dict(status=value['status'],retained=value['retained'],
                        result_sha256=digest(result))
                settle=folder/'SETTLING.json'
                if settle.is_file():
                    measured=json.loads(settle.read_text())
                    row['modes'][mode]['settling_pass']=measured['passed']
                    row['modes'][mode]['settling_last']=measured['history'][-1]
                journal=verify_journal(folder/'settle-frames',initial_sha256=entry['initial_sha256'],
                    manifest_sha256=entry['manifest_sha256'],complete=False)
                if journal['states']!=97:raise ValueError('Settling evidence incomplete')
                if value['status']=='complete':
                    path=Path(value['raw_path'])
                    if path!=folder/'raw.h5' or digest(path)!=value['raw_sha256']:raise ValueError('Raw receipt changed')
                    verify_journal(folder/'frames',initial_sha256=entry['initial_sha256'],
                        manifest_sha256=entry['manifest_sha256'],complete=True,raw_path=path)
                    import h5py
                    with np.load(folder/'settle-frames/frame-00096.npz') as settled,h5py.File(path) as raw:
                        for key in ('x','v','F_trial','C'):np.testing.assert_array_equal(settled[key],raw[key][0])
                        expected_checkpoint=json.loads(raw.attrs['manifest_json']).get('execution',{}).get('settled_checkpoint_sha256')
                    checkpoint=folder/'SETTLED_CHECKPOINT.json'
                    if expected_checkpoint is not None or checkpoint.exists():
                        if digest(checkpoint)!=expected_checkpoint:raise ValueError('Full settled checkpoint receipt changed/unbound')
                        cp=json.loads(checkpoint.read_text())
                        if digest(folder/'settled-full-state.npz')!=cp['arrays_sha256']:raise ValueError('Full settled checkpoint changed')
                        with np.load(folder/'settled-full-state.npz') as state,h5py.File(path) as raw:
                            for key in ('x','v','F_trial','C'):np.testing.assert_array_equal(state[key],raw[key][0])
                    paths[mode]=path
            if set(paths)=={'serial','grouped'}:
                try:row['comparison']=compare(paths['serial'],paths['grouped'])
                except (AssertionError,ValueError,KeyError) as error:
                    row['comparison']=dict(passed=False,error=str(error))
            elif all(v['status']!='unfinished' for v in row['modes'].values()) and len(row['modes'])==2:
                states=list(row['modes'].values())
                row['rejection_agreement']=all(v['status']==states[0]['status']
                    and v.get('settling_pass')==states[0].get('settling_pass') for v in states)
            rows.append(row)
    unfinished=sum(any(v['status']=='unfinished' for v in r['modes'].values()) for r in rows)
    if unfinished and not partial:raise ValueError('Matrix is unfinished')
    comparisons=[r['comparison'] for r in rows if 'comparison' in r]
    report=dict(planned=len(rows),unfinished=unfinished,compared=len(comparisons),
        parity_failures=[r['label'] for r in rows if not r.get('comparison',{}).get('passed',True)
                         or r.get('rejection_agreement') is False],
        statuses={mode:dict(Counter(r['modes'][mode]['status'] for r in rows)) for mode in request['modes']},
        retained={mode:sum(r['modes'][mode].get('retained',False) for r in rows) for mode in request['modes']},
        matrix_root=str(root),request_sha256=digest(root/'REQUEST.json'),rows=rows,scaling_approved=False)
    write_once(output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--partial',action='store_true')
    a=p.parse_args();audit(a.root,a.output,partial=a.partial)
