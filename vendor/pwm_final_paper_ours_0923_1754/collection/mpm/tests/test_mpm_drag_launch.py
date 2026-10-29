"""Operator rejection tests. Synthetic evidence here is NOT a scale gate."""
import json
from pathlib import Path
import pytest
from collection.common import digest,write_once
from collection.mpm import runtime
from collection.mpm.drag import launch


def request(**changes):
    return dict(dict(name='unit-only',attempts=108,gpus=2,hours=48,split='train',evaluation='train',
                     object_counts=[1,2,3,4],material_layout='homogeneous'),**changes)


@pytest.mark.parametrize('change',[{'gpus':True},{'hours':True},{'attempts':True},{'attempts':0},
    {'hours':49},{'gpus':8},{'name':'../bad'},{'object_counts':[True]},{'object_counts':[5]},
    {'object_counts':[2,1]},{'object_counts':[]},{'material_layout':'unknown'},
    {'split':'test','evaluation':'train'}])
def test_settings_reject_unsupported(change):
    with pytest.raises(ValueError):launch.settings(**request(**change))


def fixture(tmp_path):
    assets=tmp_path/'synthetic-assets';assets.mkdir()
    write_once(assets/'COMPLETE.json',dict(synthetic_unit_test_only=True))
    evidence=tmp_path/'synthetic-evidence.json';write_once(evidence,dict(synthetic_unit_test_only=True))
    gate=dict(schema='mpm-drag-scale-gate-v1',approved=True,variation=launch.VARIATION,bank_sha256=digest(assets/'COMPLETE.json'),
        critical_source_sha256={p:digest(runtime.ROOT/p) for p in launch.CRITICAL},
        checks={k:True for k in ('parity','convergence','consumer','failure_isolation','sustained_pilot','interaction_review')},
        evidence={str(evidence):digest(evidence)},capabilities=[dict(split='train',evaluation='train',
            material_layout='homogeneous',object_counts=[1,2,3,4])],synthetic_unit_test_only=True)
    path=tmp_path/'SYNTHETIC_GATE.json';write_once(path,gate)
    return assets,path,gate


def test_gate_is_mandatory_and_capability_specific(tmp_path):
    assets,path,gate=fixture(tmp_path)
    scope={k:request()[k] for k in ('split','evaluation','object_counts','material_layout')}
    launch.check_gate(path,assets,**scope)
    for key,value in [('approved',False),('variation',{}),('checks',{}),('evidence',{}),('critical_source_sha256',{})]:
        wrong=tmp_path/(key+'.json');write_once(wrong,dict(gate,**{key:value}))
        with pytest.raises(ValueError):launch.check_gate(wrong,assets,**scope)
    with pytest.raises(ValueError,match='not qualified'):
        launch.check_gate(path,assets,**dict(scope,split='validation',evaluation='unseen'))


def fixed_resolution_fixture(tmp_path):
    assets,path,gate=fixture(tmp_path)
    decision=tmp_path/'SYNTHETIC_DECISION.json'
    write_once(decision,dict(policy=launch.FIXED_RESOLUTION_POLICY,authorized_by='user',
        action_schema=launch.VARIATION['schema'],accepted_timestep_sensitivity=True,
        convergence_passed=False,instruction='Synthetic unit-test decision, not real approval'))
    diagnostic=tmp_path/'SYNTHETIC_TIMESTEP.json'
    write_once(diagnostic,dict(diagnostic='timestep-control',passed=False,comparisons=[dict(passed=False)]))
    gate.update(schema='mpm-drag-scale-fixed-resolution-gate-v1',resolution_review=dict(
        policy=launch.FIXED_RESOLUTION_POLICY,accepted_timestep_sensitivity=True,
        clock_source_sha256=digest(runtime.ROOT/'collection/mpm/clock.py'),
        decision_path=str(decision),decision_sha256=digest(decision),
        diagnostic_results={str(diagnostic):digest(diagnostic)}))
    gate['checks'].update(convergence=False,coverage=True,physical_quality=True,visual_review=True,fixed_resolution_review=True)
    gate['evidence'].update({str(decision):digest(decision),str(diagnostic):digest(diagnostic)})
    return assets,gate


def test_fixed_resolution_gate_keeps_failed_convergence_explicit(tmp_path):
    assets,gate=fixed_resolution_fixture(tmp_path)
    path=tmp_path/'SYNTHETIC_FIXED_GATE.json';write_once(path,gate)
    scope={k:request()[k] for k in ('split','evaluation','object_counts','material_layout')}
    result=launch.check_gate(path,assets,**scope)
    assert result['checks']['convergence'] is False
    # The old strict-convergence policy is not weakened by the new method choice.
    old=tmp_path/'SYNTHETIC_OLD_POLICY.json'
    write_once(old,dict(gate,schema='mpm-drag-scale-gate-v1'))
    with pytest.raises(ValueError):launch.check_gate(old,assets,**scope)


def test_fixed_resolution_pilot_is_not_production_approval(tmp_path):
    assets,gate=fixed_resolution_fixture(tmp_path)
    gate['schema']='mpm-drag-pilot-fixed-resolution-gate-v1'
    gate['checks'].pop('sustained_pilot')
    path=tmp_path/'SYNTHETIC_FIXED_PILOT.json';write_once(path,gate)
    scope={k:request()[k] for k in ('split','evaluation','object_counts','material_layout')}
    launch.check_gate(path,assets,**scope,purpose='pilot')
    with pytest.raises(ValueError):launch.check_gate(path,assets,**scope,purpose='production')


@pytest.mark.parametrize('damage',['missing_decision','unbound_decision','missing_diagnostics',
    'changed_clock','claim_convergence','no_visual_review','no_coverage','bad_policy'])
def test_fixed_resolution_gate_cannot_skip_evidence_or_other_checks(tmp_path,damage):
    assets,gate=fixed_resolution_fixture(tmp_path);review=gate['resolution_review']
    if damage=='missing_decision':review.pop('decision_path')
    elif damage=='unbound_decision':gate['evidence'].pop(review['decision_path'])
    elif damage=='missing_diagnostics':review['diagnostic_results']={}
    elif damage=='changed_clock':review['clock_source_sha256']='not-qualified'
    elif damage=='claim_convergence':gate['checks']['convergence']=True
    elif damage=='no_visual_review':gate['checks']['visual_review']=False
    elif damage=='no_coverage':gate['checks'].pop('coverage')
    else:review['policy']='ignore-everything'
    path=tmp_path/'SYNTHETIC_BAD_FIXED_GATE.json';write_once(path,gate)
    scope={k:request()[k] for k in ('split','evaluation','object_counts','material_layout')}
    with pytest.raises(ValueError):launch.check_gate(path,assets,**scope)


def test_bundle_no_submit_no_overwrite_and_changed_request(tmp_path,monkeypatch):
    import sys
    assets,gate,_=fixture(tmp_path);out=tmp_path/'synthetic-bundle'
    def no_scheduler(*a,**k):raise AssertionError('No subprocess allowed while freezing/verifying')
    monkeypatch.setattr(launch.subprocess,'run',no_scheduler)
    result=launch.bundle(out,assets,gate,python=sys.executable,**request())
    assert result['submitted'] is False
    config=launch.verify(out)
    assert config['namespace'].startswith('unit-only-')
    assert (out/'source/data/mpm_drag.py').is_file()
    state=launch.status(out)
    assert state['state']=='prepared' and state['attempted_with_receipts']==state['exported_clips']==0
    reservation=Path(config['registry'])/(config['namespace']+'.json')
    assert json.loads(reservation.read_text())['bundle']==str(out)
    import shutil
    clone=tmp_path/'cloned-bundle';shutil.copytree(out,clone)
    with pytest.raises(ValueError,match='reservation mismatch'):launch.verify(clone)
    with pytest.raises(FileExistsError):launch.bundle(out,assets,gate,python=sys.executable,**request())
    changed=dict(config,attempts=100)
    (out/'BUNDLE.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='request changed'):launch.verify(out)


def test_no_real_gate_in_worktree():
    # A missing real qualification must continue to stop the operator.
    assert not (runtime.ROOT/'collection/mpm/drag/GATE.json').exists()


def test_grid256_requires_its_own_method_review(tmp_path):
    assets,path,gate=fixture(tmp_path)
    scope={k:request()[k] for k in ('split','evaluation','object_counts','material_layout')}
    with pytest.raises(ValueError):launch.check_gate(path,assets,**scope,n_grid=256)
    result=tmp_path/'SYNTHETIC_CONTACT.json';write_once(result,dict(n_grid=256))
    decision=tmp_path/'SYNTHETIC_G256_CHOICE.json'
    write_once(decision,dict(policy=launch.GRID256_POLICY,n_grid=256,grid_lim_m=100/64,authorized_by='user',
        known_contact_artifacts_accepted=True,convergence_established=False,instruction='Synthetic test only',
        contact_results={str(result):digest(result)}))
    gate.update(schema='mpm-drag-scale-reviewed-grid-gate-v1',method_review=dict(decision_path=str(decision),decision_sha256=digest(decision)))
    gate['checks'].update(convergence=False,coverage=True,physical_quality=True,visual_review=True,fixed_resolution_review=True)
    gate['evidence'].update({str(decision):digest(decision),str(result):digest(result)})
    gate['capabilities'][0]['n_grid']=256
    reviewed=tmp_path/'SYNTHETIC_G256_GATE.json';write_once(reviewed,gate)
    launch.check_gate(reviewed,assets,**scope,n_grid=256)
    with pytest.raises(ValueError):launch.check_gate(reviewed,assets,**scope,n_grid=100)
    gate['checks']['physical_quality']=False
    failed=tmp_path/'SYNTHETIC_G256_FAILED.json';write_once(failed,gate)
    with pytest.raises(ValueError):launch.check_gate(failed,assets,**scope,n_grid=256)


def test_bounded_pilot_gate_cannot_authorize_production(tmp_path):
    assets,path,gate=fixture(tmp_path)
    del gate['checks']['sustained_pilot'];gate['schema']='mpm-drag-pilot-gate-v1'
    pilot=tmp_path/'SYNTHETIC_PILOT_GATE.json';write_once(pilot,gate)
    scope={k:request()[k] for k in ('split','evaluation','object_counts','material_layout')}
    launch.check_gate(pilot,assets,**scope,purpose='pilot')
    with pytest.raises(ValueError):launch.check_gate(pilot,assets,**scope)
    launch.check_purpose('pilot',108,1,2)
    for args in (('pilot',109,1,2),('pilot',108,2,2),('pilot',108,1,48),('unknown',108,1,2)):
        with pytest.raises(ValueError):launch.check_purpose(*args)


def test_actual_allocation_deadline_not_requested_walltime(monkeypatch):
    from datetime import datetime
    now=launch.time.time();end=now+7200
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:
        'JobId=123 JobState=RUNNING EndTime='+datetime.fromtimestamp(end).isoformat())
    assert abs(launch.allocation_deadline('123',48)-(end-300))<.01
    monkeypatch.setattr(launch.subprocess,'check_output',lambda *a,**k:
        'JobId=123 JobState=RUNNING EndTime='+datetime.fromtimestamp(now+600).isoformat())
    with pytest.raises(ValueError,match='Insufficient'):launch.allocation_deadline('123',48)


@pytest.mark.parametrize('divisors',[(1,3),(4,2),(2,2),(),(1,),(True,2),(16,64)])
def test_convergence_rejects_unbounded_divisors_before_gpu(divisors):
    from collection.mpm.drag.convergence import run
    with pytest.raises(ValueError,match='divisors'):run(None,[],None,None,divisors=divisors)
