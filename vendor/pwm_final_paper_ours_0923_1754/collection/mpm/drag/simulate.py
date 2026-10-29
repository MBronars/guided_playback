"""Observed serial/grouped command pilot; no automatic scale approval."""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import warp as wp
from collection.common import digest,write_once
from .. import runtime,physics,observer
from .grouped import CommandGroup
from ..contracts import Clock,SettlingMonitor
from ..journal import FrameJournal,verify_journal
from ..recording import atomic_raw
from ..walls import FIELDS,validate_evidence
from ..export import validate_candidate,eligible
from ..wall_materials import SAND_ONLY,STRICT
from .protocol import SETTLING,patch_mask,declared_schedule,check_schedule,SEQUENCE_SCHEMAS


class DragWorld(observer.WallObservedBackend):
    def step(self,dt,command,selected):
        if dt!=self.clock.substep_dt_s or self.steps>=self.clock.transitions*self.clock.substeps_per_transition:
            raise ValueError('Changed/exhausted drag clock')
        self._unchanged();self._particles()
        wp.launch(observer.next_step,dim=1,inputs=[self.counter],device=self.device)
        self.backend.step(dt,command,selected);self._particles()

    def rebase_observation(self):
        """Reset observation epoch only; never reset physical state or momentum."""
        before=self.backend.checkpoint()['arrays']
        self.backend.steps=0;self.solver.time=0.
        self.counter.zero_();self.first_grid.fill_(observer.NEVER)
        self.first_particle.fill_(observer.NEVER);self.covered.zero_();self._particles()
        after=self.backend.checkpoint()['arrays']
        for key in before:np.testing.assert_array_equal(before[key],after[key])


def simulate(root,labels,output,*,serial=False,uuid=None,reference_graph=False):
    if not 1<=len(labels)<=4:raise ValueError('Bounded one-to-four-world pilot required')
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'PLAN.json').read_text());runtime.verify_sources(plan['sources'])
    entries={c['label']:c for c in plan['cases']};worlds=[];manifests=[];initials=[]
    started=time.monotonic();last_check=0
    for label in labels:
        entry=entries[label];folder=root/'cases'/label
        if digest(folder/'initial.npz')!=entry['initial_sha256'] or digest(folder/'MANIFEST.json')!=entry['manifest_sha256']:
            raise ValueError('Changed prepared drag input')
        m=json.loads((folder/'MANIFEST.json').read_text())
        with np.load(folder/'initial.npz',allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        if m['action_schema'] in SEQUENCE_SCHEMAS and not np.isin(a['mat_type'],[0,1]).all():
            raise ValueError('Sand is not allowed in sequence dragging')
        b=physics.CollectionBackend(a['x'],{k:a[k] for k in runtime.STATIC},m['solver_config'],a['v'],device='cuda:0')
        physics.install_original_walls_in_padded_grid(b)
        clk=Clock(**m['clock']);worlds.append(DragWorld(b,Clock(clk.substep_dt_s,clk.substeps_per_transition,SETTLING.max_transitions)))
        manifests.append(m);initials.append(a)
    if len({w.clock for w in worlds})!=1:raise ValueError('Group must have identical clocks')
    if len({Clock(**m['clock']) for m in manifests})!=1:raise ValueError('Group recording lengths must match')
    transitions=manifests[0]['clock']['transitions']
    block=256 if reference_graph else 64
    max_speed=.6 if plan['schema'] in SEQUENCE_SCHEMAS else .3
    group=None if serial else CommandGroup(worlds,block=block,fused=not reference_graph,max_command_speed=max_speed)
    zeros=[np.zeros(w.backend.n,np.int32) for w in worlds]
    monitor=[SettlingMonitor(a['object_ids'],SETTLING) for a in initials]
    settle=[{k:[v.copy()] for k,v in dict(w.read(),**w.wall_snapshot()).items()} for w in worlds]
    dirs=[output/label for label in labels]
    for d in dirs:d.mkdir()
    journals=[FrameJournal(d/'settle-frames',initial_sha256=entries[label]['initial_sha256'],
              manifest_sha256=entries[label]['manifest_sha256'],clock=w.clock) for d,label,w in zip(dirs,labels,worlds)]
    def check():
        nonlocal last_check
        if time.monotonic()-started>3600:raise TimeoutError('Bounded drag pilot deadline')
        if uuid and time.monotonic()-last_check>5:
            from ..collect import gpu_status
            gpu_status(uuid,allow_own=True,memory_limit_mib=80000);last_check=time.monotonic()
    def step(cmd=None,sel=None):
        check()
        if group is None:
            for i,w in enumerate(worlds):w.step(w.clock.substep_dt_s,None if cmd is None else cmd[i],zeros[i] if sel is None else sel[i])
        else:group.step()
    for j,w in zip(journals,worlds):j.append(0,dict(w.read(),**w.wall_snapshot()))
    for transition in range(96):
        for _ in range(worlds[0].clock.substeps_per_transition):step()
        for w,s,mon,j in zip(worlds,settle,monitor,journals):
            value=dict(w.read(),**w.wall_snapshot());mon.observe(s['x'][-1],value['x'],value['v'])
            for k in s:s[k].append(value[k].copy())
            j.append(w.steps,value)
        if transition%12==11:print(json.dumps(dict(phase='settling',completed=transition+1,total=96,labels=labels)),flush=True)
    passed=[m.consecutive>=SETTLING.consecutive_transitions for m in monitor]
    for d,mon,ok in zip(dirs,monitor,passed):write_once(d/'SETTLING.json',dict(passed=ok,history=mon.history,policy=asdict(SETTLING)))
    rejected=[];proposed=len(labels)
    for label,d,ok in zip(labels,dirs,passed):
        evidence=verify_journal(d/'settle-frames',initial_sha256=entries[label]['initial_sha256'],
                               manifest_sha256=entries[label]['manifest_sha256'],complete=False)
        if evidence['states']!=97:raise ValueError('Incomplete settling journal')
        if not ok:
            result=dict(status='settling_rejected',label=label,retained=False,settling_journal=evidence,
                        settling_receipt_sha256=digest(d/'SETTLING.json'))
            write_once(d/'RESULT.json',result);rejected.append(result)
    if not any(passed):
        return dict(passed=False,reason='settling',proposed=proposed,retained=0,results=rejected,
                    seconds=time.monotonic()-started)
    rebuild=group is not None and not all(passed)
    # Independent grids: remove only timed-out worlds. Preserve every healthy
    # physical state; the command graph is rebuilt after observation rebasing.
    labels,dirs,worlds,manifests,initials,settle,zeros=[
        [v for v,ok in zip(values,passed) if ok] for values in
        (labels,dirs,worlds,manifests,initials,settle,zeros)]
    masks=[];cmds=[];gates=[];records=[];journals=[]
    for label,d,w,m in zip(labels,dirs,worlds,manifests):
        a=m['action'];selected=patch_mask(w.read()['x'],w.backend.static['object_ids'],a['object_id'],a['fraction'])
        cmd,active=declared_schedule(selected,a,schema=m['action_schema'],transitions=transitions)
        check_schedule(cmd,selected,active,transitions=transitions,schema=m['action_schema'])
        masks.append(selected);cmds.append(cmd);gates.append(active)
        w.rebase_observation()
        w.clock=Clock(**m['clock'])
        # Keep the full restart state, not only plotted positions/velocities.
        # Future controlled comparisons can pin plastic history and patch choice.
        from ..collect import save_npz
        checkpoint=w.backend.checkpoint()
        save_npz(d/'settled-full-state.npz',checkpoint['arrays'])
        write_once(d/'SETTLED_CHECKPOINT.json',dict(
            {k:v for k,v in checkpoint.items() if k not in ('arrays','extrema')},
            extrema=checkpoint['extrema'].tolist(),arrays_sha256=digest(d/'settled-full-state.npz')))
        journals.append(FrameJournal(d/'frames',initial_sha256=entries[label]['initial_sha256'],
            manifest_sha256=entries[label]['manifest_sha256'],clock=Clock(**m['clock'])))
        records.append({k:[] for k in ('x','v','F_trial','C',*FIELDS)})
    if rebuild:
        before=[w.backend.checkpoint()['arrays'] for w in worlds]
        # These worlds already have the reviewed fused wall topology installed.
        group=CommandGroup(worlds,block=block,fused=False,max_command_speed=max_speed)
        for w,previous in zip(worlds,before):
            for key,value in previous.items():np.testing.assert_array_equal(value,w.backend.checkpoint()['arrays'][key])
    if group is not None:group.clock=worlds[0].clock
    guard_start=[w.backend.calls_checked for w in worlds]
    for frame in range(transitions+1):
        if frame:
            command=[c[frame-1] for c in cmds];selected=[mask*active[frame-1] for mask,active in zip(masks,gates)]
            if group is not None:group.set_commands(command,selected)
            for _ in range(worlds[0].clock.substeps_per_transition):step(command,selected)
        for w,record,j in zip(worlds,records,journals):
            value=dict(w.read(),**w.wall_snapshot());j.append(w.steps,value)
            for k in record:record[k].append(value[k].copy())
        if frame%12==0:print(json.dumps(dict(phase='drag',completed=frame,total=transitions,labels=labels)),flush=True)
    results=[]
    for i,(label,d,w,m,initial,record) in enumerate(zip(labels,dirs,worlds,manifests,initials,records)):
        if w.backend.calls_checked-guard_start[i]!=2*w.steps:raise ValueError('Native guard coverage changed')
        arrays={k:np.stack(v).astype(np.int64 if k in FIELDS else np.float32) for k,v in record.items()}
        arrays.update({k:initial[k] for k in runtime.STATIC})
        arrays.update(selected=masks[i],command_velocity=cmds[i],transition_active=gates[i],
            active_substeps=gates[i]*w.clock.substeps_per_transition,state_time_s=Clock(**m['clock']).state_time_s,
            settle_x=np.stack(settle[i]['x']),settle_v=np.stack(settle[i]['v']))
        arrays.update({'settle_'+k:np.stack(v) for k,v in settle[i].items() if k not in ('x','v','C')})
        m.update(wall_observation=w.wall_metadata,seed_uint64=int(m['episode_id'][4:20],16),
                 execution=dict(serial=serial,group_size=len(worlds),guard_calls=w.backend.calls_checked,
                                block=256 if serial else block,fused_walls=not serial and not reference_graph,
                                settled_checkpoint_sha256=digest(d/'SETTLED_CHECKPOINT.json'),
                                settling_native_steps=96*w.clock.substeps_per_transition,
                                observation_rebase_preserves_all_physical_arrays=True))
        validate_evidence(m,arrays);runtime.verify_sources(plan['sources'])
        raw=d/'raw.h5';atomic_raw(raw,m,arrays)
        journal=verify_journal(d/'frames',initial_sha256=entries[label]['initial_sha256'],
                              manifest_sha256=entries[label]['manifest_sha256'],complete=True,raw_path=raw)
        quality=validate_candidate(raw,wall_policy=STRICT if m['action_schema'] in SEQUENCE_SCHEMAS else SAND_ONLY)
        result=dict(status='complete',label=label,raw_path=str(raw.resolve()),raw_sha256=digest(raw),
                    journal=journal,quality=quality,retained=eligible(quality))
        write_once(d/'RESULT.json',result);results.append(result)
    return dict(passed=not rejected and all(r['quality']['integrity_pass'] for r in results),
                seconds=time.monotonic()-started,proposed=proposed,rejected_settling=len(rejected),
                retained=sum(r['retained'] for r in results),results=results+rejected)
