"""Readiness/identity tests; no claim of simulator qualification."""
import io
import json
import tarfile
import subprocess
import sys
from types import SimpleNamespace

import pytest

from collection.common import digest
from collection.mpm.evaluation import (
    training_inventory, prepare_smoke, _execute_frozen, review_smoke, REVIEW_SMOKE,
    index_records, index_export,
)


def shard(root, rows, name='00000.tar'):
    root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows).encode()
    with tarfile.open(root / name, 'w') as stream:
        member = tarfile.TarInfo('batch_00000000.provenance.json')
        member.size = len(payload)
        stream.addfile(member, io.BytesIO(payload))
    return root


def episode(**changes):
    row = dict(episode_id='mpm-example', asset_ids=['seen'], split='train', evaluation='train',
               object_cohort='train', split_manifest_sha256='split-hash')
    return dict(row, **changes)


def read(roots, *, task='passive'):
    return training_inventory(roots, {'seen', 'unused'}, 'split-hash', task=task)


def index_fixture(tmp_path, *, unseen=False, changes=None):
    evaluation='unseen' if unseen else 'seen'
    row=episode(task='passive',split='validation',evaluation=evaluation,
                object_cohort='validation' if unseen else 'train',
                asset_ids=['novel'] if unseen else ['seen'],raw_sha256='raw',
                state_time_s=[i*.0417 for i in range(49)])
    row.update(changes or {})
    root=shard(tmp_path/'published',[row])
    manifest=dict(schema='mpm-evaluation-suite-v1',category='passive_'+evaluation,
                  task='passive',split='validation',accepted_for_training=False,
                  episode_ids=['mpm-example'],retained=1,quotas={'1':1},
                  shards={'00000.tar':digest(root/'00000.tar')})
    train=dict(schema='pwm-indexed-episodes-v1',domain='mpm_fresh',split='train',
               records=[dict(episode_id='training-episode',group='training-episode',asset_ids=['seen'])])
    return root,manifest,train


@pytest.mark.parametrize('unseen',[False,True])
def test_index_bridge_preserves_evaluation_identity_without_random_split(tmp_path,unseen):
    root,manifest,train=index_fixture(tmp_path,unseen=unseen)
    domain,rows=index_records(root,manifest,train)
    assert domain=='mpm_fresh' and len(rows)==1
    assert rows[0]['episode_id']==rows[0]['group']=='mpm-example'
    assert rows[0]['frames']==49 and rows[0]['row']==0
    assert rows[0]['evaluation_category']==manifest['category']
    assert rows[0]['file']==str(root/'00000.tar')


@pytest.mark.parametrize('unseen',[False,True])
def test_index_bridge_keeps_drag_domain_and_command_schema(tmp_path,unseen):
    root,manifest,train=index_fixture(tmp_path,unseen=unseen,
        changes=dict(task='drag',action_schema='mpm-drag-command-sequence-v4',state_time_s=[i*.0417 for i in range(81)]))
    manifest.update(task='drag',category='drag_unseen' if unseen else 'drag_seen')
    train['domain']='mpm_drag'
    domain,rows=index_records(root,manifest,train)
    assert domain=='mpm_drag' and rows[0]['frames']==81
    assert rows[0]['action_schema']=='mpm-drag-command-sequence-v4'


@pytest.mark.parametrize('changes',[
    dict(episode_id='training-episode'),dict(asset_ids=['absent']),
    dict(state_time_s=[i*.04 for i in range(49)]),dict(task='drag'),
    dict(split='train'),dict(action_schema='wrong-command-schema'),
])
def test_index_bridge_rejects_mismatched_provenance(tmp_path,changes):
    root,manifest,train=index_fixture(tmp_path,changes=changes)
    with pytest.raises(ValueError):index_records(root,manifest,train)


def test_index_bridge_checks_full_actual_training_shape_membership(tmp_path):
    root,manifest,train=index_fixture(tmp_path,unseen=True)
    train['records'].append(dict(episode_id='another-train',group='another-train',asset_ids=['novel']))
    with pytest.raises(ValueError,match='Held-out shape'):index_records(root,manifest,train)


def test_index_bridge_checks_payload_membership_and_never_overwrites(tmp_path):
    root,manifest,train=index_fixture(tmp_path)
    (root/'extra.tar').write_bytes(b'not declared')
    with pytest.raises(ValueError,match='shard membership'):index_records(root,manifest,train)
    (root/'extra.tar').unlink();(root/'00000.tar').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Changed published shard'):index_records(root,manifest,train)
    with pytest.raises(FileExistsError):index_export('/missing','passive_seen','/missing',root)


def test_only_exported_shapes_are_seen(tmp_path):
    result = read([shard(tmp_path / 'train', [episode()])])
    assert result['episodes'] == 1
    assert result['seen_asset_uids'] == ['seen']
    assert result['object_counts'] == {1: 1}
    assert not result['full_shard_payloads_verified']


@pytest.mark.parametrize('changes', [dict(split='test'), dict(evaluation='seen'),
    dict(object_cohort='validation'), dict(split_manifest_sha256='changed'),
    dict(asset_ids=['heldout']), dict(asset_ids=[]), dict(asset_ids=['seen', 'seen']),
    dict(episode_id=''), dict(task='drag'), dict(action_schema='mpm-drag-command-sequence-v4')])
def test_bad_membership_fails_closed(tmp_path, changes):
    with pytest.raises(ValueError):
        read([shard(tmp_path / 'train', [episode(**changes)])])


def test_no_duplicated_episode_across_roots(tmp_path):
    roots = [shard(tmp_path / name, [episode()]) for name in ('a', 'b')]
    with pytest.raises(ValueError, match='duplicate episode'):
        read(roots)


def test_duplicate_and_nested_roots_rejected(tmp_path):
    root = shard(tmp_path / 'train', [episode()])
    for roots in ([root, root], [tmp_path, root]):
        with pytest.raises(ValueError, match='Duplicate or nested'):
            read(roots)


def test_tar_alias_rejected(tmp_path):
    a = shard(tmp_path / 'a', [episode()]); b = tmp_path / 'b'; b.mkdir()
    (b / 'alias.tar').symlink_to(a / '00000.tar')
    with pytest.raises(ValueError, match='Duplicate shard'):
        read([a, b])


def test_passive_and_drag_are_separate(tmp_path):
    root = shard(tmp_path / 'drag', [episode(task='drag', action_schema='v4')])
    assert read([root], task='drag')['seen_asset_uids'] == ['seen']
    with pytest.raises(ValueError, match='Mixed passive/drag'):
        read([root])


def test_empty_export_and_empty_provenance_rejected(tmp_path):
    with pytest.raises(ValueError, match='No exported'):
        read([tmp_path])
    with pytest.raises(ValueError, match='Nonempty batched'):
        read([shard(tmp_path / 'empty', [])])


def test_prepare_rejects_changed_bundle_before_creating_output(tmp_path):
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    (bundle / 'payload').write_text('changed')
    (bundle / 'BUNDLE.json').write_text(json.dumps({'files': {'payload': 'wrong'}}))
    output = tmp_path / 'out'
    with pytest.raises(ValueError, match='Artifact changed'):
        prepare_smoke(bundle, tmp_path, output, '/missing', '/missing', task='passive')
    assert not output.exists()


def test_prepare_rejects_unknown_task_without_io(tmp_path):
    with pytest.raises(ValueError, match='Unknown task'):
        prepare_smoke('/missing', '/missing', tmp_path / 'out', '/missing', '/missing', task='test')


@pytest.mark.parametrize('failure', ['timeout', 'exit', 'source_change'])
def test_child_failure_never_publishes_success(tmp_path, monkeypatch, failure):
    bundle = tmp_path / 'bundle'; bundle.mkdir()
    payload = bundle / 'payload'; payload.write_text('frozen')
    manifest = {'files': {'payload': digest(payload)}}
    out = tmp_path / 'out'; out.mkdir()
    def child(*args, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired('test-child', 900)
        if failure == 'source_change':
            payload.write_text('changed')
        return SimpleNamespace(returncode=int(failure == 'exit'), stdout='', stderr='test failure')
    monkeypatch.setattr(subprocess, 'run', child)
    with pytest.raises((subprocess.TimeoutExpired, RuntimeError, ValueError)):
        _execute_frozen(bundle, manifest, out, sys.executable, 'pass', {}, {}, 'COMPLETE.json')
    assert (out / 'FAILURE.json').is_file()
    assert not (out / 'COMPLETE.json').exists()
    assert (out / 'CHILD.json').is_file()


def test_review_execution_cannot_escape_prepared_root(tmp_path):
    with pytest.raises(ValueError, match='execution directory'):
        review_smoke('/missing', '/missing', tmp_path / 'out', '/missing', '/missing',
                     task='drag', drag_execution='../serial')


@pytest.mark.parametrize('family', [-1, 5, True, '0'])
def test_invalid_passive_smoke_family_rejected_without_io(tmp_path, family):
    with pytest.raises(ValueError, match='family index'):
        prepare_smoke('/missing', '/missing', tmp_path / 'out', '/missing', '/missing',
                      task='passive', passive_family_index=family)


@pytest.mark.parametrize('retain_counts', [True, False])
def test_smoke_review_separates_categories_and_reports_empty_coverage(tmp_path, monkeypatch, retain_counts):
    """Exercise the actual child script with synthetic outcomes, not physics."""
    root = tmp_path / 'inputs'; output = tmp_path / 'review'; output.mkdir()
    for category, n in [('unseen', 1), ('count_unseen', 7)]:
        plan = root / category; cases = []
        for label in ('control-00', 'control-02', f'case-{n}'):
            kind = 'matched_control' if label.startswith('control') else 'coverage'
            cases.append(dict(label=label, kind=kind, template=dict(n_objects=n)))
            folder = plan / 'cases' / label / 'attempts/attempt-00000'; folder.mkdir(parents=True)
            raw = folder / 'raw.h5'; raw.write_bytes(b'synthetic-not-a-simulation')
            (folder / 'RESULT.json').write_text(json.dumps(dict(status='complete',
                raw_sha256=digest(raw), quality=dict(integrity_pass=True,
                eligible_for_wall_free_export=label != 'control-02'))))
        (plan / 'PLAN.json').write_text(json.dumps(dict(cases=cases)))
    def quality(path, **kwargs):
        category = path.parents[4].name
        return dict(raw_path=str(path), episode_id=category, integrity_pass=True,
                    eligible_for_wall_free_export=category == 'unseen' or retain_counts)
    calls = []
    def export(paths, out, **kwargs):
        categories = {p.parents[4].name for p in paths}
        assert len(categories) == 1, 'Mixed evaluation categories in one export'
        calls.append(next(iter(categories)))
        return {'retained': sum(quality(p)['eligible_for_wall_free_export'] for p in paths)}
    monkeypatch.setitem(sys.modules, 'collection.mpm.export', SimpleNamespace(
        validate_candidate=quality, eligible=lambda q:q['eligible_for_wall_free_export'],
        export_candidates=export))
    monkeypatch.setitem(sys.modules, 'collection.mpm.wall_materials', SimpleNamespace(SAND_ONLY='sand', STRICT='strict'))
    monkeypatch.setitem(sys.modules, 'collection.mpm.review', SimpleNamespace(
        check_model=lambda out, kept:dict(profiles=2, episodes_per_profile=len(kept))))
    monkeypatch.setattr(sys, 'argv', ['review', json.dumps(dict(root=str(root), output=str(output), task='passive'))])
    exec(compile(REVIEW_SMOKE, '<review-smoke-unit-test>', 'exec'), {})
    report = json.loads((output / 'REVIEW.json').read_text())
    assert sorted(calls) == ['count_unseen', 'unseen']
    assert report['retained'] == 1 + int(retain_counts)
    if not retain_counts:
        assert report['consumer']['count_unseen']['checked'] is False
    assert report['production_approved'] is False
