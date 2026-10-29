"""Synthetic bank tests of planned diversity; not simulation qualification."""
import json
from pathlib import Path
import numpy as np
import pytest
from collection.common import write_once
from collection.mpm.drag.prepare import prepare
from collection.mpm.fresh_banks import VOLUME_POLICY


def bank(tmp_path,monkeypatch):
    root=tmp_path/'assets';root.mkdir();(root/'train').mkdir()
    rng=np.random.default_rng(10);points=rng.uniform(-.5,.5,(1024,3)).astype(np.float32)
    points/=2*np.abs(points).max()
    obj=dict(points_unit=points,source_x=points,source_particle_index=np.arange(1024,dtype=np.int32),
             particle_volume=np.full(1024,1e-6,np.float32))
    meta=dict(bank_role='train',schema='fresh-candidate-bank-v1',volume_policy=VOLUME_POLICY,
              objects=[dict(asset_uid=f'fixture-{i:03d}',object_size_m=.1) for i in range(16)])
    write_once(root/'train/MANIFEST.json',meta);write_once(root/'SPLITS.json',{})
    write_once(root/'COMPLETE.json',{'synthetic':True})
    monkeypatch.setattr('collection.mpm.drag.prepare.load_bank',lambda *a,**k:(meta,[obj]*16))
    return root


@pytest.mark.parametrize('layout',['homogeneous','mixed'])
def test_bank_coverage_object_targets_and_material_fields(tmp_path,monkeypatch,layout):
    assets=bank(tmp_path,monkeypatch);out=tmp_path/'prepared'
    prepare(out,assets,name='unit-tests',count=36,material_layout=layout)
    plan=json.loads((out/'PLAN.json').read_text())
    assert plan['planned_unique_shapes']==16 and plan['object_counts']==[1,2,3,4]
    targets={n:set() for n in (1,2,3,4)}
    for c in plan['cases']:
        folder=out/'cases'/c['label'];m=json.loads((folder/'MANIFEST.json').read_text())
        targets[m['n_objects']].add(m['action']['object_id'])
        with np.load(folder/'initial.npz') as f:
            assert np.isin(f['mat_type'],[0,1]).all()
            assert len(set(m['case']['asset_ids']))==m['n_objects']
            assert m['material_names']==[('elastic','plasticine','sand')[int(v)] for v in f['mat_type']]
            assert not f['v'].any()
            assert np.isclose(f['x'][:,2].min(),.115)
            if layout=='homogeneous':assert len(set(f['mat_type']))==1
            elif m['n_objects']>1:assert len(set(f['mat_type']))>1
        assert all(pair['gap_m']>=pair['required_m'] for pair in m['geometry_provenance']['pair_gaps'])
    assert all(targets[n]==set(range(n)) for n in targets)


def test_sequence_full_crossing_continuous_speed_and_no_sand(tmp_path,monkeypatch):
    from collection.mpm.drag.protocol import declared_schedule,CURRENT_SEQUENCE_VERSION as SEQUENCE_VERSION
    assets=bank(tmp_path,monkeypatch);crossing=set();speeds=[]
    for block in range(2):
        out=tmp_path/f'block-{block}'
        prepare(out,assets,name='crossing',count=108,start=108*block)
        plan=json.loads((out/'PLAN.json').read_text())
        for case in plan['cases']:
            m=json.loads((out/'cases'/case['label']/'MANIFEST.json').read_text())
            with np.load(out/'cases'/case['label']/'initial.npz') as a:
                assert np.isin(a['mat_type'],[0,1]).all()
                crossing.add((m['n_objects'],int(a['mat_type'][0]),float(a['E'][0]),
                              len(m['action']['events']),m['requested_first_family']))
            assert m['scenario']==m['action']['events'][0]['family']
            if m['scenario']!=m['requested_first_family']:
                layout=m['geometry_provenance']['drag_interaction']
                assert layout['mode'] in ('pair','crowded') and layout['all_lift_final_event_changed_to_pull']
                assert len(m['action']['events'])==1 and m['scenario']=='pull_release'
            selected=np.array([1,0],np.int32)
            cmd,gate=declared_schedule(selected,m['action'],schema=SEQUENCE_VERSION,transitions=m['clock']['transitions'])
            assert np.count_nonzero(np.diff(np.r_[0,gate,0])==1)==len(m['action']['events'])
            speeds.extend(e['speed'] for e in m['action']['events'])
    assert len(crossing)==4*2*3*3*3
    assert len(set(speeds))==len(speeds) and min(speeds)<.1 and .2<max(speeds)<.25


@pytest.mark.parametrize('kwargs',[dict(event_counts=[]),dict(event_counts=[0]),dict(event_counts=[3,1]),
    dict(event_counts=[True]),dict(speed_range=[.1,.7]),dict(speed_range=[.2,.2]),dict(speed_range=[float('nan'),.3])])
def test_invalid_sequence_sampler_settings(tmp_path,kwargs):
    with pytest.raises(ValueError):prepare(tmp_path/'bad',tmp_path/'missing-assets',name='invalid',**kwargs)


@pytest.mark.parametrize('counts',[[0],[5],[2,1],[1,1],[],[True]])
def test_invalid_count_selection_rejected_before_bank_read(tmp_path,monkeypatch,counts):
    assets=bank(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Object counts'):
        prepare(tmp_path/'bad',assets,name='unit',object_counts=counts)


def test_count_subset_and_no_overwrite(tmp_path,monkeypatch):
    assets=bank(tmp_path,monkeypatch);out=tmp_path/'subset'
    prepare(out,assets,name='unit',count=4,object_counts=[1,3])
    plan=json.loads((out/'PLAN.json').read_text());assert plan['object_counts']==[1,3]
    with pytest.raises(FileExistsError):prepare(out,assets,name='unit',count=4,object_counts=[1,3])


def test_grid256_native_preparation(tmp_path,monkeypatch):
    assets=bank(tmp_path,monkeypatch);out=tmp_path/'g256'
    prepare(out,assets,name='grid-test',count=1,start=16,n_grid=256)
    plan=json.loads((out/'PLAN.json').read_text());assert plan['n_grid']==256
    m=json.loads((out/'cases'/plan['cases'][0]['label']/'MANIFEST.json').read_text())
    from dataclasses import asdict
    from collection.mpm.drag.fragmentation import FragmentationPolicy
    assert m['fragmentation_policy']==plan['fragmentation_policy']==asdict(FragmentationPolicy())
    assert m['n_grid']==m['solver_config']['n_grid']==256
    assert m['clock']['substep_dt_s']==5e-5
    assert m['wave_budget']['initial_budgeted_courant']<.4
    for n in (192,257,True):
        with pytest.raises(ValueError):prepare(tmp_path/'bad',assets,name='bad',n_grid=n)
