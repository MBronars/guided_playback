"""Publish exact simulation-ready MPM banks to a lab-readable immutable copy.

Does not change source data, split membership, running bundles, or home access.
"""
import argparse
import grp
import json
import os
from pathlib import Path
import shutil
import sys

import mpm_collect as operator


def group_read_tree(root,gid):
    paths=[root]+sorted(root.rglob('*'))
    for p in paths:
        if p.is_symlink():raise ValueError('Shared release cannot contain symlinks: '+str(p))
        os.chown(p,-1,gid)
        os.chmod(p,0o2750 if p.is_dir() else 0o640)


def publish(source,destination,group):
    source=source.resolve(strict=True);destination=destination.absolute()
    gid=grp.getgrnam(group).gr_gid
    if gid not in os.getgroups():raise PermissionError('Publisher must belong to '+group)
    complete=operator.read(source/'COMPLETE.json')
    expected={'train':10000,'validation':258,'test':262}
    if complete['cohorts']!=expected:raise ValueError('Unexpected prepared bank membership')
    artifacts=complete['artifacts']
    for name,sha in artifacts.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts or (source/rel).is_symlink():raise ValueError('Unsafe bank artifact')
        if operator.digest(source/rel)!=sha:raise ValueError('Source bank changed: '+name)
    if destination.exists():raise FileExistsError('Never overwrite an existing asset release')
    destination.mkdir(parents=True,exist_ok=False)
    for name in list(artifacts)+['COMPLETE.json']:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/name,target)
    for name,sha in artifacts.items():
        if operator.digest(destination/name)!=sha:raise ValueError('Published bank differs: '+name)
    if operator.digest(source/'COMPLETE.json')!=operator.digest(destination/'COMPLETE.json'):
        raise ValueError('Bank completion changed during publication')
    split=operator.read(destination/'SPLITS.json')
    sets={}
    for role in expected:
        rows=operator.read(destination/role/'MANIFEST.json')['objects']
        sets[role]={r['asset_uid'] for r in rows}
        if len(rows)!=expected[role] or len(sets[role])!=len(rows):raise ValueError('Duplicate/missing bank asset')
        if not sets[role]<=set(split['cohorts'][role]):raise ValueError('Asset outside assigned cohort')
    if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Split leakage')
    receipt=dict(schema='shared-mpm-assets-v1',group=group,source=str(source),root=str(destination),
        bank_complete_sha256=operator.digest(destination/'COMPLETE.json'),cohorts=expected,
        exact_payload_copy=True,disjoint_prepared_cohorts=True,excluded=len(complete['exclusions']),
        roles={r:str(destination/r) for r in expected},
        note='Prepared 1024-particle simulation assets, not a new mesh conversion or evaluation-run approval.')
    operator.write_once(destination/'SHARED_ASSETS.json',receipt)
    group_read_tree(destination,gid)
    # Only the newly created assets/mpm ancestor is involved; existing project
    # directory permissions are never broadened here.
    for p in (destination.parent,destination.parent.parent):
        os.chown(p,-1,gid);os.chmod(p,0o2750)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--group',default='kfragki2_group')
    a=p.parse_args()
    print(json.dumps(publish(a.source,a.destination,a.group),indent=2))
