"""Command-driven MPM integration, synthetic arrays only."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from data.batch import TrajectoryBatch
from data.mpm_drag import UnifiedMPMDragDataset,commands
from data.unified_dataset import UnifiedMPMDataset,unified_position_stats,_shift
from data.normalization import resolve_normalization,domain_normalizers


def sample():
    b,t,n=2,49,32
    selected=np.zeros((b,n),np.int32);selected[:,:4]=1
    active=np.zeros((b,t),bool);active[:,4:28]=True
    velocity=np.zeros((b,t-1,n,3),np.float32);velocity[:,4:16,:4,2]=.15
    times=(np.arange(t)*.0417).tolist()
    return {'particle_pos.pyd':np.random.default_rng(8).uniform(.4,.5,(b,t,n,3)).astype(np.float32),
        'valid_particles.pyd':np.ones((b,t,n),bool),'material_properties.pyd':np.tile([1e4,.2,1000,0,0],(b,n,1)).astype(np.float32),
        'object_ids.pyd':np.zeros((b,n),np.int32),'grab_velocity.pyd':velocity,
        'is_grabbed.pyd':selected,'drag_active.pyd':active,'__key__':'test',
        'provenance.json':[dict(task='drag',action_schema='mpm-drag-command-schedule-v1',state_time_s=times) for _ in range(b)]}


def adapter(profile='xy_p99_per_domain',active=True,first=True):
    cls=UnifiedMPMDragDataset if active else UnifiedMPMDataset
    ds=object.__new__(cls);ds.training=False;ds.context_len=4;ds.future_len=16
    ds.first_window=first;ds.prediction_target='delta';ds._shift_t=_shift('mpm')
    ds.delta_stats,_,_=domain_normalizers(resolve_normalization(profile),'mpm')
    ds.position_stats=unified_position_stats()
    return ds


@pytest.mark.parametrize('profile',['xy_p99_per_domain','xy_p99_global'])
@pytest.mark.parametrize('first',[True,False])
def test_exact_window_commands_and_no_future_leakage(profile,first):
    ds=adapter(profile,first=first);s=sample();batch=ds._process_sample(s)
    delta,gate,selected=commands(s)
    for i,start in enumerate(ds._window_starts):
        torch.testing.assert_close(batch.actions[i],delta[i,start+3:start+19],atol=0,rtol=0)
        torch.testing.assert_close(batch.context_action_active[i],gate[i,start:start+3],atol=0,rtol=0)
    # Keep window selection pinned; modifying future targets must not change commands.
    s['particle_pos.pyd'][:,4:]+=10
    changed=ds._process_sample(s)
    torch.testing.assert_close(changed.actions,batch.actions,atol=0,rtol=0)
    torch.testing.assert_close(batch.delta_stats.denormalize(batch.delta_stats.normalize(batch.actions)),batch.actions,atol=1e-8,rtol=1e-5)
    assert torch.all(batch.get_valid_mask())  # includes every driven deformable particle


def test_passive_adapter_rejects_active_hold():
    s=sample();s['grab_velocity.pyd'].fill(0)
    with pytest.raises(ValueError,match='Active MPM'):adapter(active=False)._process_sample(s)


@pytest.mark.parametrize('profile',['xy_p99_per_domain','xy_p99_global'])
def test_gentle_v4_exact_commands_and_filter_provenance(profile):
    from collection.mpm.tests.test_mpm_drag_sequence import sequence_fixture
    from collection.mpm.drag.protocol import GENTLE_SEQUENCE_VERSION,declared_schedule
    from collection.mpm.validate import _batch_payload
    # Transport tests must not require SciPy or simulator dependencies.
    m,a=sequence_fixture();m['action_schema']=GENTLE_SEQUENCE_VERSION
    m['fragmentation_policy']={'version':'saved-point-fragmentation-review-v1'}
    for event in m['action']['events']:event['speed']*=.4
    a['command_velocity'],a['transition_active']=declared_schedule(a['selected'],m['action'],schema=GENTLE_SEQUENCE_VERSION,transitions=len(a['x'])-1)
    a['active_substeps']=834*a['transition_active']
    s={k+'.pyd':v for k,v in _batch_payload([(m,a)]).items()}
    s['provenance.json']=[dict(m,state_time_s=a['state_time_s'].tolist())]
    ds=adapter(profile);batch=ds._process_sample(s);delta,gate,_=commands(s)
    torch.testing.assert_close(batch.actions,delta[:,3:19],atol=0,rtol=0)
    torch.testing.assert_close(batch.action_active,gate[:,3:19],atol=0,rtol=0)
    del s['provenance.json'][0]['fragmentation_policy']
    with pytest.raises(ValueError,match='fragmentation'):commands(s)


@pytest.mark.parametrize('damage',['clock','speed','dtype','unselected'])
def test_command_consumer_rejects_bad_contract(damage):
    s=sample()
    if damage=='clock':s['provenance.json'][0]['state_time_s']=(np.arange(49)/24).tolist()
    elif damage=='speed':s['grab_velocity.pyd']*=10
    elif damage=='dtype':s['grab_velocity.pyd']=s['grab_velocity.pyd'].astype(np.float64)
    else:s['grab_velocity.pyd'][0,10,-1,0]=.1
    with pytest.raises(ValueError):commands(s)


def test_transforms_and_temporal_validation():
    batch=adapter()._process_sample(sample())
    for b in (batch.to('cpu'),batch.float(),batch.half().float(),TrajectoryBatch.collate([batch.slice(1),batch.slice(1)])):
        assert b.action_active.dtype==torch.bool
        torch.testing.assert_close(b.action_active[0],batch.action_active[0])
    with pytest.raises(ValueError,match='Both temporal'):replace(batch,context_action_active=None)


def test_long_windows_keep_hold_release_and_pad_inactive():
    from data.test_batch import LongTrajectoryBatch
    s=sample();delta,gate,selected=commands(s);batch=adapter()._process_sample(s)
    full=LongTrajectoryBatch(positions=torch.from_numpy(s['particle_pos.pyd']),actions=delta,
        action_mask=selected,action_active=gate,object_ids=batch.object_ids,valid_lengths=batch.valid_lengths,
        E=batch.E,nu=batch.nu,rho=batch.rho,mat_type=batch.mat_type,
        delta_stats=batch.delta_stats,position_stats=batch.position_stats)
    for start in (0,13,23,40):
        b=full.get_window(start,4,16,pad_to_length=True)
        available=min(16,48-start-3)
        torch.testing.assert_close(b.action_active[:,:available],gate[:,start+3:start+3+available])
        assert not b.action_active[:,available:].any()
        torch.testing.assert_close(b.context_action_active,gate[:,start:start+3])


@pytest.mark.parametrize('events',[1,2,3])
@pytest.mark.parametrize('profile',['xy_p99_per_domain','xy_p99_global'])
def test_current_sequence_every_rollout_window(events,profile):
    from collection.mpm.tests.test_mpm_drag_sequence import sequence_fixture
    from collection.mpm.validate import _batch_payload
    from inference import rollout_window
    m,a=sequence_fixture(events)
    s={k+'.pyd':v for k,v in _batch_payload([(m,a)]).items()}
    s['provenance.json']=[dict(m,state_time_s=a['state_time_s'].tolist())]
    ds=adapter(profile);ds.future_len=-1
    full=ds._process_sample(s);delta,gate,_=commands(s)
    torch.testing.assert_close(full.actions,delta[:,3:],atol=0,rtol=0)
    for start in range(4,len(a['x'])-16,7):
        window=rollout_window(full,full.positions.clone(),start,16)
        torch.testing.assert_close(window.actions,delta[:,start-1:start+15],atol=0,rtol=0)
        torch.testing.assert_close(window.action_active,gate[:,start-1:start+15])
        torch.testing.assert_close(window.context_action_active,gate[:,start-4:start-1])
        back=window.delta_stats.denormalize(window.delta_stats.normalize(window.actions))
        torch.testing.assert_close(back,window.actions,rtol=1e-5,atol=1e-8)


def test_drag_roots_are_explicit_and_passive_default_unchanged(tmp_path):
    from data.unified_dataset import resolve_data_roots,build_domain_dataset
    train=tmp_path/'train';val=tmp_path/'val';train.mkdir();val.mkdir()
    (train/'fake.tar').touch();(val/'fake.tar').touch()
    roots=resolve_data_roots({'mpm':dict(train=str(train),val=str(val),task='drag')})
    assert isinstance(build_domain_dataset('mpm',roots=roots,raw=True),UnifiedMPMDragDataset)
    roots=resolve_data_roots({'mpm':dict(train=str(train),val=str(val))})
    assert type(build_domain_dataset('mpm',roots=roots,raw=True)) is UnifiedMPMDataset


def test_rollout_uses_command_gates_across_ar_boundary():
    from inference import rollout_window
    ds=adapter();ds.future_len=-1;b=ds._process_sample(sample()).slice(1)
    committed=b.positions.clone()+.02
    for start in (4,17,29):
        window=rollout_window(b,committed,start,8)
        full=torch.cat([b.context_action_active,b.action_active],dim=1)
        torch.testing.assert_close(window.context_action_active,full[:,start-4:start-1])
        torch.testing.assert_close(window.action_active,full[:,start-1:start+7])
        torch.testing.assert_close(window.actions,b.actions[:,start-4:start+4])


def test_preflight_waits_only_for_idle_sampling(monkeypatch):
    from collection.mpm.drag.cli import wait_idle
    from collection.mpm import collect
    calls=[]
    def status(uuid):
        calls.append(uuid)
        if len(calls)==1:raise RuntimeError('GPU idle/memory guard failed')
        return dict(idle=True)
    monkeypatch.setattr(collect,'gpu_status',status)
    monkeypatch.setattr('collection.mpm.drag.cli.time.sleep',lambda _:None)
    assert wait_idle('GPU-test')['idle'] and len(calls)==2
    def foreign(uuid):raise RuntimeError('Foreign process on selected GPU; owned worker stops')
    monkeypatch.setattr(collect,'gpu_status',foreign)
    with pytest.raises(RuntimeError,match='Foreign'):wait_idle('GPU-test')


@pytest.mark.parametrize('architecture',['transformer','unet'])
def test_static_all_active_matches_temporal_all_active(architecture):
    from models.transformer import ModelConfig,TrajectoryTransformer
    from models.unet_std import UNetWrapper
    batch=adapter()._process_sample(sample()).slice(1)
    batch=replace(batch,action_active=batch.action_mask[:,None,:].expand_as(batch.action_active).clone(),
                  context_action_active=batch.action_mask[:,None,:].expand_as(batch.context_action_active).clone())
    static=replace(batch,action_active=None,context_action_active=None)
    cfg=ModelConfig(dim=32,depth=1,heads=4,num_latents=8,k_neighbors=4,dropout=0.,context_len=4,
        unet_actions=True,all_relative=True,unet_points=0,unet_strides='1,2',unet_widths='16,32,64',
        unet_enc_depths='1,1,1',unet_dec_depths='1,1',unet_head_dim=8,unet_t_dim=32,
        unet_l1_by_mat='0:16',unet_pool_k=4,unet_interp_k=3,unet_geo_r_m=.1,unet_fps_random_start=False)
    model=(TrajectoryTransformer(cfg) if architecture=='transformer' else UNetWrapper(cfg)).eval()
    noisy=torch.randn_like(batch.actions);t=torch.full((1,16),.5)
    with torch.no_grad():
        torch.manual_seed(12);left=model(batch=static,noisy_future=noisy,timestep=t)
        torch.manual_seed(12);right=model(batch=batch,noisy_future=noisy,timestep=t)
    torch.testing.assert_close(left,right,atol=0,rtol=0)


@pytest.mark.parametrize('architecture',['transformer','unet'])
def test_model_forward_backward_and_hold_release(architecture):
    from models.transformer import ModelConfig,TrajectoryTransformer
    from models.unet_std import UNetWrapper
    batch=adapter()._process_sample(sample()).slice(1)
    cfg=ModelConfig(dim=32,depth=1,heads=4,num_latents=8,k_neighbors=4,dropout=0.,context_len=4,
        unet_actions=True,all_relative=True,unet_points=0,unet_strides='1,2',unet_widths='16,32,64',
        unet_enc_depths='1,1,1',unet_dec_depths='1,1',unet_head_dim=8,unet_t_dim=32,
        unet_l1_by_mat='0:16,1:16,2:16',unet_pool_k=4,unet_interp_k=3,unet_geo_r_m=.1,
        unet_fps_random_start=False)
    torch.manual_seed(4)
    model=(TrajectoryTransformer(cfg) if architecture=='transformer' else UNetWrapper(cfg)).eval()
    noisy=torch.randn_like(batch.actions);t=torch.full((1,16),.5)
    result=model(batch=batch,noisy_future=noisy,timestep=t)
    assert result.shape==batch.actions.shape and torch.isfinite(result).all()
    result.square().mean().backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.parameters())
    # Explicitly exercise zero command while activity differs. Inspect inputs,
    # since zero-initialized output heads can mask real conditioning differences.
    hold=replace(batch,actions=torch.zeros_like(batch.actions),
        action_active=batch.action_mask[:,None].expand_as(batch.action_active).clone())
    release=replace(hold,action_active=torch.zeros_like(hold.action_active))
    if architecture=='transformer':
        embedded=model._embed_temporal(hold.delta_stats.normalize(hold.actions),model.action_embed)
        assert not torch.equal(model._mask_actions(embedded,hold.action_active),model._mask_actions(embedded,release.action_active))
    else:
        seen=[]
        hook=model.model.register_forward_pre_hook(lambda module,args,kwargs:seen.append(kwargs['feats'].detach().clone()),with_kwargs=True)
        with torch.no_grad():
            model(batch=hold,noisy_future=noisy,timestep=t);model(batch=release,noisy_future=noisy,timestep=t)
        hook.remove()
        assert not torch.equal(seen[0][...,-1],seen[1][...,-1])
