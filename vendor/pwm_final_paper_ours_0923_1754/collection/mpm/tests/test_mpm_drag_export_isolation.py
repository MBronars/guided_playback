import json
import pytest
from collection.common import write_once
from collection.mpm.drag import launch,review
from collection.mpm import export


def fake_export(paths,output,**kwargs):
    output.mkdir(parents=True);write_once(output/'COMPLETE.json',{'retained':1})
    return {'retained':1}


def test_failed_consumer_never_publishes(tmp_path,monkeypatch):
    monkeypatch.setattr(export,'export_candidates',fake_export)
    def failed(*args):raise AssertionError('injected consumer failure')
    monkeypatch.setattr(review,'loader_review',failed)
    pending=tmp_path/'pending';public=tmp_path/'train/block-0'
    with pytest.raises(AssertionError):launch.reviewed_export(['raw'],['raw'],pending,public)
    assert pending.exists() and not public.exists()


def test_reviewed_publication_is_complete_and_no_clobber(tmp_path,monkeypatch):
    monkeypatch.setattr(export,'export_candidates',fake_export)
    monkeypatch.setattr(review,'loader_review',lambda *args:[dict(episodes=1,
        exact_raw_window_and_command_match=True,profile=p) for p in ('per-domain','global')])
    pending=tmp_path/'pending';public=tmp_path/'train/block-0'
    selection,consumer=launch.reviewed_export(['raw'],['raw'],pending,public)
    assert not pending.exists() and public.exists()
    assert json.loads((public/'LOADER_REVIEW.json').read_text())['retained']==1
    with pytest.raises(FileExistsError):launch.reviewed_export(['raw'],['raw'],pending,public)


def test_incomplete_consumer_or_selection_is_not_published(tmp_path,monkeypatch):
    monkeypatch.setattr(export,'export_candidates',fake_export)
    monkeypatch.setattr(review,'loader_review',lambda *args:[])
    with pytest.raises(ValueError):launch.reviewed_export(['raw'],['raw'],tmp_path/'p',tmp_path/'train')
    assert not (tmp_path/'train').exists()
    with pytest.raises(ValueError):launch.reviewed_export(['raw'],[],tmp_path/'p2',tmp_path/'train')


@pytest.mark.parametrize('failures',[1,3])
def test_worker_continues_isolated_export_failure_but_bounds_repeats(tmp_path,monkeypatch,failures):
    from collection.mpm.drag import prepare,matrix
    from collection.mpm import runtime
    import time
    from types import SimpleNamespace
    config=dict(gpus=1,hours=2,attempts=432,shutdown_reserve_seconds=60,
        minimum_free_bytes=1,assets='synthetic',namespace='test',python='unused',
        split='train',evaluation='train',object_counts=[1],material_layout='homogeneous',access_gid=0)
    monkeypatch.setattr(launch,'verify',lambda root:config)
    monkeypatch.setattr(runtime,'ROOT',tmp_path/'source')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','synthetic')
    monkeypatch.setattr(launch.os,'sched_getaffinity',lambda pid:{0})
    monkeypatch.setattr(launch.os,'sched_setaffinity',lambda *args:None)
    monkeypatch.setattr(launch.shutil,'disk_usage',lambda root:SimpleNamespace(free=1<<45))
    monkeypatch.setattr(launch,'share_created',lambda *args:None)
    monkeypatch.setattr(launch.subprocess,'run',lambda *args,**kwargs:None)
    def inputs(folder,*args,**kwargs):write_once(folder/'PLAN.json',{'cases':[{'label':'fake'}]})
    monkeypatch.setattr(prepare,'prepare',inputs)
    def simulate(inputs,output,*args,**kwargs):
        for i in range(108):write_once(output/'grouped/group-0'/f'drag-{i}'/'RESULT.json',
            dict(status='settling_rejected',retained=False))
    monkeypatch.setattr(matrix,'run',simulate)
    calls=[]
    def publish(*args):
        calls.append(args)
        if len(calls)<=failures:raise AssertionError('injected exporter fault')
        return dict(retained=0),[]
    monkeypatch.setattr(launch,'reviewed_export',publish)
    if failures==3:
        with pytest.raises(RuntimeError,match='Three consecutive'):launch.worker(tmp_path,0,'synthetic',time.time()+3600)
        assert len(calls)==3
    else:
        launch.worker(tmp_path,0,'synthetic',time.time()+3600)
        assert len(calls)==4 and (tmp_path/'collection/block-000003/COMPLETE.json').exists()
    assert len(list(tmp_path.glob('collection/*/EXPORT_FAILURE.json')))==failures
    assert not (tmp_path/'data/train/train/block-000000').exists()
