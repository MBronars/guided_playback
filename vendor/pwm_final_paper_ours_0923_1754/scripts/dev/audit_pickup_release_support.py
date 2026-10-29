"""Audit release-related observations in a preselected training-shard sample.

Sampled point clearances and motor targets are descriptive proxies, not a
physical release certificate or a test of trained model performance.
"""
import argparse
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
import torch
from data.blocklists import filter_blocklisted
from data.unified_dataset import UnifiedPickupDataset,exclusion_files
from utils.reproducibility import code_revision,file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-protocol',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--dependency-path',action='append',default=[])
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    import sys
    for path in args.dependency_path:sys.path.append(str(Path(path).resolve()))
    from scipy.spatial import cKDTree
    root=Path('/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success').resolve()
    reference=json.loads(args.reference_protocol.read_text());exclusions=exclusion_files('pickup','train')
    ds=UnifiedPickupDataset([str(root)],split='train',context_len=4,future_len=16,training=False,shuffle=False,blocklist_paths=exclusions,normalization=reference['normalization'])
    rng=np.random.default_rng(19751);shards=sorted(ds.shards);chosen=rng.choice(len(shards),size=min(8,len(shards)),replace=False)
    selection=[]
    for i in chosen:
        shard=shards[i]
        with tarfile.open(shard) as archive:
            keys=sorted(n.removesuffix('.particle_pos.pyd') for n in archive.getnames() if n.endswith('.particle_pos.pyd'))
        ids=rng.choice(len(keys),size=min(4,len(keys)),replace=False)
        selection.extend(dict(shard=shard,key=keys[k]) for k in ids)
    args.out.mkdir(parents=True)
    protocol=dict(root=str(root),split='train',seed=19751,selection='Eight shards uniformly without replacement, four batch keys per shard uniformly without replacement; all surviving batch elements; no outcome/asset filtering.',entries=selection,exclusions={str(Path(p).resolve()):file_sha256(p) for p in exclusions},reference_protocol_sha256=file_sha256(args.reference_protocol),runner_sha256=file_sha256(__file__),**code_revision())
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];decisions=[]
    required=['particle_pos','obj_ids','joint_pos_obs','joint_pos_target','link_local_npts','asset_id','reward','material_properties']
    for entry in selection:
        with tarfile.open(entry['shard']) as archive:
            raw={k+'.pyd':pickle.load(archive.extractfile(entry['key']+'.'+k+'.pyd')) for k in required}
        raw.update(__url__=entry['shard'],__key__=entry['key']);raw['original_elem.pyd']=np.arange(len(raw['particle_pos.pyd']))
        survivors=list(filter_blocklisted([raw],ds.blocklist,ds.roots))
        if not survivors:decisions.append(dict(**entry,status='all elements excluded'));continue
        sample=survivors[0];npts=np.asarray(sample['link_local_npts.pyd']);sizes=[int(npts[i]) for i in (8,9,10)]
        for i,asset in enumerate(np.asarray(sample['asset_id.pyd'])):
            p=np.asarray(sample['particle_pos.pyd'])[i];oid=np.asarray(sample['obj_ids.pyd'])[i];q=np.asarray(sample['joint_pos_obs.pyd'])[i,:,-2:];targets=np.asarray(sample['joint_pos_target.pyd'])[i,:,-2:]
            if not all(np.isfinite(x).all() for x in (p,q,targets)):raise ValueError('Nonfinite source state in selected sample')
            assert (oid[:sum(sizes)]==1).all() and (oid[sum(sizes):]==0).all()
            offsets=np.cumsum([0]+sizes);gaps=[]
            for frame in p:
                obj=cKDTree(frame[oid==0]);gaps.append([float(obj.query(frame[offsets[f]:offsets[f+1]])[0].min()) for f in (1,2)])
            gaps=np.asarray(gaps);near=gaps.max(-1)<=.005;clear=gaps.min(-1)>=.005
            later_clear=False
            for t in range(1,len(p)-5):
                if near[:t].any() and clear[t:t+6].all():later_clear=True
            row=dict(**entry,elem=int(np.asarray(sample['original_elem.pyd'])[i]),asset_id=int(asset),frames=len(p),ever_both_fingers_near=bool(near.any()),final6_both_fingers_clear=bool(clear[-6:].all()),near_then_six_clear_frames=later_clear,final6_min_finger_gap_mm=(gaps[-6:].min(0)*1000).tolist(),final6_max_finger_gap_mm=(gaps[-6:].max(0)*1000).tolist(),opening_change_last6_vs_first4_mm=((q[-6:].mean(0)-q[:4].mean(0))*1000).tolist(),commanded_opening_change_mm=((targets[-6:].mean(0)-targets[:4].mean(0))*1000).tolist(),target_q_range_mm=(np.stack([targets.min(0),targets.max(0)])*1000).tolist(),peak_reward=float(np.asarray(sample['reward.pyd'])[i].max()),exported_friction=float(np.asarray(sample['material_properties.pyd'])[i,oid==0,3][0]))
            rows.append(row)
        decisions.append(dict(**entry,status='measured',surviving_elements=len(sample['asset_id.pyd'])))
    report=dict(status='completed',protocol=protocol,segments=len(rows),distinct_assets=len(set(v['asset_id'] for v in rows)),near_then_six_clear_frames=sum(v['near_then_six_clear_frames'] for v in rows),final6_both_fingers_clear=sum(v['final6_both_fingers_clear'] for v in rows),ever_both_fingers_near=sum(v['ever_both_fingers_near'] for v in rows),rows=rows,decisions=decisions,limitations=['Seeded sample of stored training segments, not the full corpus or full source episodes.','Point clearances can miss mesh contact; motor targets do not prove object release.','No source-simulator launch configuration was recovered. The model may receive other pickup subsets with different motion.'])
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('protocol','rows','decisions')},indent=2))


if __name__=='__main__':main()
