"""Evaluation orchestration/selection tests; synthetic fixtures are not physics evidence."""
from collections import Counter
import json
from pathlib import Path
import stat

import pytest

from collection.mpm import evaluation_suite as suite


@pytest.mark.parametrize('category', suite.CATEGORIES)
def test_fixed_budgets_unique_candidates_and_count_quotas(category):
    p = suite.candidate_plan('release-v1', 'test', category, [f'uid-{i}' for i in range(30)])
    assert p['target'] == 100 and sum(p['quotas'].values()) == 100
    assert len(p['candidates']) == 800
    assert len({r['id'] for r in p['candidates']}) == 800
    assert len({r['seed_namespace'] for r in p['candidates']}) == 800
    assert Counter(r['count'] for r in p['candidates']) == {int(n): q * 8 for n, q in p['quotas'].items()}
    for row in p['candidates']:
        assert len(set(row['asset_uids'])) == row['count']
        assert set(row['asset_uids']) <= set(p['allowed_uids'])
    assert sorted(p['quotas'].values()) in ([25]*4, [33,33,34])


def test_planning_is_independent_of_uid_input_order_and_split_namespaces_are_disjoint():
    uids = [f'uid-{i}' for i in range(30)]
    a = suite.candidate_plan('release-v1', 'test', 'drag_seen', uids)
    assert a == suite.candidate_plan('release-v1', 'test', 'drag_seen', list(reversed(uids)))
    b = suite.candidate_plan('release-v1', 'validation', 'drag_seen', uids)
    assert not {r['seed_namespace'] for r in a['candidates']} & {r['seed_namespace'] for r in b['candidates']}


def test_drag_native_factor_crossing_and_seen_restriction():
    p = suite.candidate_plan('release-v1', 'test', 'drag_seen', [f'seen-{i}' for i in range(15)])
    for n in (1,2,3,4):
        rows = [r for r in p['candidates'] if r['count'] == n][:54]
        assert len({(r['material'],r['stiffness_phase'],r['events'],r['first_family']) for r in rows}) == 54
        assert all(r['native_index'] % 4 + 1 == n for r in rows)


def test_restrict_bank_preserves_values_order_and_cohort():
    objects = [object() for _ in range(10)]
    meta = dict(bank_role='train', objects=[dict(asset_uid=str(i), sha256=str(i)) for i in range(10)])
    m, a = suite.restrict_bank(meta, objects, ['9','2','5'])
    assert m['bank_role'] == 'train' and [r['asset_uid'] for r in m['objects']] == ['2','5','9']
    assert a == [objects[i] for i in (2,5,9)]
    assert len(meta['objects']) == len(objects) == 10
    with pytest.raises(ValueError): suite.restrict_bank(meta, objects, ['missing'])


@pytest.mark.parametrize('kwargs', [dict(split='train'),dict(category='bad'),dict(target=3),
                                  dict(reserves=0),dict(reserves=True),dict(name='../escape')])
def test_invalid_plans_fail_closed(kwargs):
    args=dict(name='suite',split='test',category='passive_seen',allowed_uids=[str(i) for i in range(10)])
    with pytest.raises(ValueError): suite.candidate_plan(**dict(args, **kwargs))


def fixture_plan(tmp_path):
    category = 'passive_seen'
    config = suite.candidate_plan('unit', 'validation', category, [str(i) for i in range(10)], target=4, reserves=2)
    plan = dict(categories={category:config}, split='validation')
    suite.write(tmp_path/'PLAN.json', plan)
    return plan, category


def outcome(root, plan, category, index, *, retained=True, episode=None):
    row = plan['categories'][category]['candidates'][index]
    folder = suite.case_root(root, category, row); attempt = folder/'attempt-0000'
    review = dict(candidate_id=row['id'], retained=retained, artifacts={}, count=row['count'])
    suite.write(attempt/'REVIEW.json', review)
    value = dict(candidate_id=row['id'], plan_sha256=suite.sha(root/'PLAN.json'),
                 attempt=attempt.name, review_sha256=suite.sha(attempt/'REVIEW.json'),
                 retained=retained, episode_id=episode or f'episode-{index}', count=row['count'], index=index)
    suite.write(folder/'OUTCOME.json', value)
    return value


def test_completion_order_cannot_change_membership(tmp_path):
    plan, category = fixture_plan(tmp_path)
    outcome(tmp_path, plan, category, 1)
    state = suite.selection(tmp_path, plan, category)
    assert state['pending']['index'] == 0 and not state['chosen']
    outcome(tmp_path, plan, category, 0, retained=False)
    for i in (2,3,4): outcome(tmp_path, plan, category, i)
    state = suite.selection(tmp_path, plan, category)
    assert state['complete'] and [r['index'] for r in state['chosen']] == [1,2,3,4]
    assert state['counts'] == {1:1,2:1,3:1,4:1}


def test_budget_exhaustion_is_not_completion(tmp_path):
    plan, category = fixture_plan(tmp_path)
    for i in range(8): outcome(tmp_path, plan, category, i, retained=False)
    state = suite.selection(tmp_path, plan, category)
    assert state['exhausted'] and not state['complete']
    with pytest.raises(ValueError, match='incomplete quotas'): suite.publish(tmp_path, plan, category)


def test_duplicate_episode_and_changed_receipt_fail(tmp_path):
    plan, category = fixture_plan(tmp_path)
    outcome(tmp_path, plan, category, 0, episode='duplicate')
    outcome(tmp_path, plan, category, 1, episode='duplicate')
    with pytest.raises(ValueError, match='Duplicate retained'): suite.selection(tmp_path, plan, category)
    row=plan['categories'][category]['candidates'][0]
    (suite.case_root(tmp_path,category,row)/'attempt-0000/REVIEW.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed candidate review'): suite.selection(tmp_path, plan, category)


def test_atomic_receipts_do_not_overwrite(tmp_path):
    path=tmp_path/'receipt.json'; suite.write(path, dict(value=1)); suite.write(path, dict(value=1))
    with pytest.raises(ValueError, match='Refusing'): suite.write(path, dict(value=2))
    assert suite.read(path)==dict(value=1)
    assert stat.S_IMODE(path.stat().st_mode)==0o640


def test_interrupted_attempt_requires_explicit_retry(tmp_path):
    plan, category=fixture_plan(tmp_path); row=plan['categories'][category]['candidates'][0]
    (suite.case_root(tmp_path,category,row)/'attempt-0000').mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Interrupted attempt preserved'):
        suite.finish_candidate(tmp_path,plan,category,row,'GPU-test')


def test_resume_after_completed_review_does_not_resimulate(tmp_path, monkeypatch):
    plan, category=fixture_plan(tmp_path); row=plan['categories'][category]['candidates'][0]
    folder=suite.case_root(tmp_path,category,row)/'attempt-0000'
    suite.write(folder/'ATTEMPT.json',dict(candidate_id=row['id'],plan_sha256=suite.sha(tmp_path/'PLAN.json'),
                                         attempt='attempt-0000',prior_attempts=[]))
    suite.write(folder/'PREPARED.json',dict(candidate_id=row['id'],placement_rejected=False))
    suite.write(folder/'SIMULATION.json',dict(status='complete'))
    suite.write(folder/'REVIEW.json',dict(candidate_id=row['id'],retained=True,episode_id='e0',status='complete',artifacts={}))
    monkeypatch.setattr(suite,'run_child',lambda *a,**k:pytest.fail('Completed work was rerun'))
    result=suite.finish_candidate(tmp_path,plan,category,row,'GPU-test')
    assert result['retained'] and (folder.parent/'OUTCOME.json').exists()


def test_foreign_category_worker_is_not_replaced(tmp_path):
    with suite.lock_category(tmp_path,'passive_seen'):
        with pytest.raises(RuntimeError,match='already has a worker'):
            with suite.lock_category(tmp_path,'passive_seen'): pass


def test_lazy_bank_reads_only_selected_unchanged_payloads(tmp_path):
    import numpy as np
    x=np.arange(9,dtype=np.float32).reshape(3,3)
    np.savez_compressed(tmp_path/'a.npz',source_x=x,points_unit=x)
    rows=[dict(file='a.npz',sha256=suite.sha(tmp_path/'a.npz')),
          dict(file='unused.npz',sha256='not-present')]
    bank=suite.LazyBank(tmp_path,rows)
    assert len(bank)==2 and bank.cache=={}
    np.testing.assert_array_equal(bank[0]['source_x'],x)
    assert set(bank.cache)=={0}
    with pytest.raises(FileNotFoundError): bank[1]
    rows[0]['sha256']='changed'; bank=suite.LazyBank(tmp_path,rows)
    with pytest.raises(ValueError,match='payload changed'): bank[0]


def test_publication_rechecks_exact_manifest_and_shard_membership(tmp_path):
    plan,category=fixture_plan(tmp_path)
    for i in range(4):
        outcome(tmp_path,plan,category,i)
        row=plan['categories'][category]['candidates'][i]
        export=suite.case_root(tmp_path,category,row)/'attempt-0000/export/filtered/diagnostic-export'
        export.mkdir(parents=True); (export/'00000.tar').write_bytes(f'synthetic-{i}'.encode())
    manifest=suite.publish(tmp_path,plan,category)
    assert manifest['retained']==4 and len(manifest['shards'])==4
    destination=tmp_path/'datasets/validation'/category
    assert stat.S_IMODE(destination.stat().st_mode)==0o2750
    assert stat.S_IMODE((destination/'MANIFEST.json').stat().st_mode)==0o640
    assert suite.publish(tmp_path,plan,category)==manifest
    path=tmp_path/'datasets/validation'/category/'MANIFEST.json'
    changed=dict(manifest,shards={});path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='Published manifest'):suite.publish(tmp_path,plan,category)
    path.write_text(json.dumps(manifest))
    extra=path.parent/'unexpected.tar';extra.write_bytes(b'not declared')
    with pytest.raises(ValueError,match='shard membership'):suite.publish(tmp_path,plan,category)
    extra.unlink()
    shard=path.parent/next(iter(manifest['shards']))
    shard.unlink()
    with pytest.raises(ValueError,match='shard membership'):suite.publish(tmp_path,plan,category)


def test_deep_selection_rechecks_reviewed_payload(tmp_path):
    plan,category=fixture_plan(tmp_path);row=plan['categories'][category]['candidates'][0]
    folder=suite.case_root(tmp_path,category,row)/'attempt-0000';folder.mkdir(parents=True)
    payload=folder/'raw.h5';payload.write_bytes(b'synthetic fixture, not simulator output')
    review=dict(candidate_id=row['id'],retained=True,artifacts={'raw.h5':suite.sha(payload)},count=row['count'])
    suite.write(folder/'REVIEW.json',review)
    suite.write(folder.parent/'OUTCOME.json',dict(candidate_id=row['id'],plan_sha256=suite.sha(tmp_path/'PLAN.json'),
        attempt=folder.name,review_sha256=suite.sha(folder/'REVIEW.json'),retained=True,episode_id='fixture',
        count=row['count'],index=0))
    assert suite.selection(tmp_path,plan,category,deep=True)['counts']=={1:1}
    payload.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed bound artifact'):
        suite.selection(tmp_path,plan,category,deep=True)


def test_completed_count_skips_reserves_without_resampling(tmp_path):
    plan,category=fixture_plan(tmp_path)
    for i in (0,2,3):outcome(tmp_path,plan,category,i)
    outcome(tmp_path,plan,category,1,retained=False)
    state=suite.selection(tmp_path,plan,category)
    assert 4 in state['skipped'] and state['pending']['index']==5
    outcome(tmp_path,plan,category,5)
    state=suite.selection(tmp_path,plan,category)
    assert state['complete'] and [r['index'] for r in state['chosen']]==[0,2,3,5]


def test_all_category_namespaces_are_disjoint_even_for_same_shapes():
    plans=[suite.candidate_plan('same-release',split,category,[str(i) for i in range(30)])
           for split in ('validation','test') for category in suite.CATEGORIES]
    namespaces=[r['seed_namespace'] for p in plans for r in p['candidates']]
    assert len(namespaces)==len(set(namespaces))


def test_qualification_requires_retained_multi_object_multi_event_drag(tmp_path,monkeypatch):
    category='drag_seen'
    config=suite.candidate_plan('unit','validation',category,[str(i) for i in range(10)],target=4)
    plan=dict(split='validation',categories={category:config})
    suite.write(tmp_path/'PLAN.json',plan)
    manifest=tmp_path/'datasets/validation'/category/'MANIFEST.json'
    suite.write(manifest,dict(synthetic_fixture=True))
    chosen=[]
    for i,row in enumerate(config['candidates'][:4]):
        folder=suite.case_root(tmp_path,category,row)/'attempt-0000'
        suite.write(folder/'REVIEW.json',dict(count=row['count'],events=1))
        chosen.append(dict(index=i,attempt=folder.name))
    monkeypatch.setattr(suite,'verify_plan',lambda root:(tmp_path,plan))
    monkeypatch.setattr(suite,'selection',lambda *a,**k:dict(complete=True,chosen=chosen))
    monkeypatch.setattr(suite,'publish',lambda *a,**k:None)
    monkeypatch.setattr(suite,'compatibility',lambda p:dict(fixture=True))
    with pytest.raises(ValueError,match='multi-object multi-event'):
        suite.qualify(tmp_path,'drag')
    assert not (tmp_path/'QUALIFIED-drag.json').exists()
    row=config['candidates'][1]
    (suite.case_root(tmp_path,category,row)/'attempt-0000/REVIEW.json').write_text(json.dumps(dict(count=2,events=2)))
    result=suite.qualify(tmp_path,'drag')
    assert result['reference_execution_verified']
    assert result['physical_accuracy_claim'] is result['acceleration_approved'] is False


def test_allocated_gpu_resolution_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1,3')
    monkeypatch.setattr(suite.subprocess,'check_output',lambda *a,**k:'0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n')
    assert suite.allocated_gpus()==['GPU-b','GPU-d']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-d,GPU-b')
    assert suite.allocated_gpus()==['GPU-d','GPU-b']
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-b,GPU-b')
    with pytest.raises(ValueError,match='Exactly two'):suite.allocated_gpus()


def test_infrastructure_retry_cannot_change_initial_scene(tmp_path,monkeypatch):
    plan,category=fixture_plan(tmp_path);row=plan['categories'][category]['candidates'][0]
    folder=suite.case_root(tmp_path,category,row)
    suite.write(folder/'attempt-0000/PREPARED.json',dict(candidate_id=row['id'],placement_rejected=False,initial_sha256='old'))
    def child(root,plan,category,row,attempt,phase,gpu):
        assert phase=='prepare'
        suite.write(folder/attempt/'PREPARED.json',dict(candidate_id=row['id'],placement_rejected=False,initial_sha256='new'))
    monkeypatch.setattr(suite,'run_child',child)
    with pytest.raises(ValueError,match='retry changed'):
        suite.finish_candidate(tmp_path,plan,category,row,'GPU-test',retry_interrupted=True)
    assert not (folder/'OUTCOME.json').exists()


def test_test_execution_requires_successful_matching_validation(monkeypatch):
    plan=dict(split='test',validation_root='/validation')
    monkeypatch.setattr(suite,'qualify',lambda *a:(_ for _ in ()).throw(ValueError('incomplete validation')))
    with pytest.raises(ValueError,match='incomplete'):suite.authorize_test(plan,'passive')
    monkeypatch.setattr(suite,'qualify',lambda *a:dict(compatibility={'different':'source'}))
    monkeypatch.setattr(suite,'compatibility',lambda p:{'actual':'source'})
    with pytest.raises(ValueError,match='exact sources'):suite.authorize_test(plan,'drag')
