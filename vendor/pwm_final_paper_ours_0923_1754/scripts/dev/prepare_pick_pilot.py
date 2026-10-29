"""Pin distinct development/evaluation pickup geometries before model sampling.

Eligibility uses observed geometry only. Four development and four reserved
scenes come from the registered pick-success validation shards and exclusions.
"""
import argparse
import json
from pathlib import Path
import pickle
import tarfile

import numpy as np
import torch

from data.blocklists import filter_blocklisted
from data.unified_dataset import UnifiedPickupDataset, exclusion_files
from utils.reproducibility import file_sha256, code_revision


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-protocol',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--seed',type=int,default=831)
    parser.add_argument('--task',choices=['pick','place'],default='pick')
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists(): raise FileExistsError(out)
    proto=json.loads(Path(args.reference_protocol).read_text())
    root=Path(f'/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_{args.task}_success').resolve()
    exclusions=exclusion_files('pickup','val')
    ds=UnifiedPickupDataset([str(root)],split='val',context_len=4,future_len=36,training=False,
        shuffle=False,blocklist_paths=exclusions,normalization=proto['normalization'])
    ds.emit_sidecar=True; ds.articulated=True
    rng=np.random.default_rng(args.seed)
    shards=list(ds.shards); rng.shuffle(shards)
    required=('particle_pos','obj_ids','material_properties','valid_particles','link_pos','link_quat',
              'link_local_pts','link_local_npts','asset_id')
    if args.task=='place': required+=('goal_pose',)
    entries=[]; archives=[]
    try:
        for shard in shards:
            archive=tarfile.open(shard); archives.append(archive)
            keys=sorted(n[:-len('.particle_pos.pyd')] for n in archive.getnames() if n.endswith('.particle_pos.pyd'))
            rng.shuffle(keys); entries.append((shard,archive,keys))
        out.mkdir(parents=True)
        selected=[]; seen={8059}; decisions=[]
        for ordinal in range(40):
            for shard,archive,keys in entries:
                if ordinal>=len(keys) or len(selected)>=8: continue
                key=keys[ordinal]
                raw={k+'.pyd':pickle.load(archive.extractfile(key+'.'+k+'.pyd')) for k in required}
                raw.update(__url__=shard,__key__=key)
                raw['original_elem.pyd']=np.arange(len(raw['particle_pos.pyd']))
                filtered=list(filter_blocklisted([raw],ds.blocklist,ds.roots))
                if not filtered:
                    decisions.append(dict(shard=shard,key=key,reason='registered exclusion')); continue
                sample=filtered[0]
                for index,asset in enumerate(np.asarray(sample['asset_id.pyd'])):
                    if int(asset) in seen: continue
                    x=np.asarray(sample['particle_pos.pyd'])[index]
                    obj=np.asarray(sample['obj_ids.pyd'])[index]==0
                    if len(x)<40: continue
                    travel=np.linalg.norm(x[:4,obj].mean(1)-x[0,obj].mean(0),axis=-1).max()
                    extent=np.ptp(x[3,obj],axis=0)
                    gap=torch.cdist(torch.tensor(x[3,obj]),torch.tensor(x[3,~obj])).min().item()
                    eligible=(travel<=.005 and gap>=.01 and extent.min()<=.075 and extent.max()<=.20) if args.task=='pick' else (
                        np.isfinite(np.asarray(sample['goal_pose.pyd'])[index]).all())
                    if not eligible:
                        decisions.append(dict(shard=shard,key=key,asset=int(asset),reason='observed eligibility',
                            object_context_travel_m=float(travel),minimum_hand_gap_m=gap,extent_m=extent.tolist()))
                        continue
                    B=len(sample['particle_pos.pyd'])
                    one={name:value[index:index+1] if (name.endswith('.pyd') and hasattr(value,'ndim')
                        and value.ndim>0 and value.shape[0]==B) else value for name,value in sample.items()}
                    batch=ds._process_sample(one)
                    if not bool(torch.isfinite(batch.positions).all()): continue
                    batch.domain='pickup'; batch.source_shard=shard
                    group='development' if len(selected)<4 else 'reserved'
                    name=f'{group}_{len(selected)%4:02d}_asset_{int(asset)}'
                    destination=out/name; destination.mkdir()
                    torch.save(batch,destination/'input.pt')
                    elem=int(np.asarray(sample['original_elem.pyd'])[index])
                    source=dict(asset_id=np.array(int(asset)),link_pos=np.asarray(raw['link_pos.pyd'])[elem],
                        link_quat=np.asarray(raw['link_quat.pyd'])[elem],link_local_pts=raw['link_local_pts.pyd'],
                        link_local_npts=raw['link_local_npts.pyd'],key=np.array(key),elem=np.array(elem))
                    np.savez_compressed(destination/'source_geometry.npz',**source)
                    episode=dict(shard=shard,key=key,elem=elem,asset_id=int(asset),full_positions_max_error_m=0.)
                    if args.task=='place':
                        episode['requested_goal_native']=np.asarray(sample['goal_pose.pyd'])[index].tolist()
                        episode['goal_provenance']='Segment task request, not achieved future; native environment-local xyz + wxyz quaternion (collector comment incorrectly says xyzw; PlaceEnv consumes wxyz)'
                    (destination/'source_episode.json').write_text(json.dumps(episode,indent=2)+'\n')
                    selected.append(dict(name=name,group=group,episode=episode,input_sha256=file_sha256(destination/'input.pt'),
                        object_context_travel_m=float(travel),minimum_hand_gap_m=gap,extent_m=extent.tolist()))
                    seen.add(int(asset)); break
            if len(selected)>=8: break
        manifest=dict(seed=args.seed,checkpoint_sha256=proto['checkpoint_sha256'],normalization=proto['normalization'],
            roots=[str(root)],validation_shards=shards,exclusions={str(Path(v).resolve()):file_sha256(v) for v in exclusions},
            task=args.task,selection='seeded shard/key traversal; distinct assets; '+(
                'observed object travel<=5mm, hand gap>=10mm, min extent<=75mm, max extent<=200mm' if args.task=='pick' else
                'finite separately recorded task goal; no geometry, generated-outcome or future-motion eligibility filter'),
            holdout='reserved from guidance tuning; no claim of unseen training assets',
            context_len=4,horizon=36,scenes=selected,decisions=decisions,**code_revision())
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if len(selected)!=8: raise RuntimeError(f'Only {len(selected)} eligible assets found; partial manifest preserved')
        print(json.dumps(dict(scenes=[(s['name'],s['episode']) for s in selected],manifest=str(out/'manifest.json')),indent=2))
    finally:
        for archive in archives: archive.close()


if __name__=='__main__': main()
