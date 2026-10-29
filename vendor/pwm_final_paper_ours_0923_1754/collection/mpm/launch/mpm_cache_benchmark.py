"""Bounded cold/warm cache A/B on identical prepared inputs; no job submission.

Run under an existing Slurm allocation. Never reserves production sequences or
exports a training release. Each arm is a fresh process, as in production.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','worker'])
    for name in ('bundle','plan','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--uuid',required=True)
    p.add_argument('--parent',type=int)
    p.add_argument('--cold',action='store_true')
    p.add_argument('--pair-only',action='store_true',help='Cold baseline and one reused-cache arm; inspect cache-hit logs before calling it warm')
    a=p.parse_args();bundle=a.bundle.resolve(strict=True)
    sys.path[:0]=[str(bundle),str(bundle/'source')]
    import mpm_four as four
    import mpm_training_campaign as campaign
    import mpm_scale as support
    from collection.mpm import collect
    config=campaign.verify(bundle)
    os.environ['PWM_MPM_REFERENCE_ROOT']=config['reference']
    os.environ['PWM_MODEL_ROOT']=str(bundle/'source')
    if a.command=='worker':
        four.worker(a.out,campaign.MODE,a.uuid,a.parent,training_only=True,reuse_cache=not a.cold)
        return
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!=a.uuid:
        raise ValueError('Explicit allocated GPU UUID required')
    collect.gpu_status(a.uuid)
    plan=collect.inspect(a.plan)
    four.require_training_plan({'split':'train','evaluation':'train'},plan['cases'])
    a.out.mkdir(parents=True,exist_ok=False)
    arms=('cold','reuse-warm-1') if a.pair_only else ('cold','reuse-cold','reuse-warm-1','reuse-warm-2')
    support.write(a.out/'REQUEST.json',dict(bundle=str(bundle),bundle_sha256=support.sha(bundle/'BUNDLE.json'),
        plan=str(a.plan.resolve()),plan_sha256=support.sha(a.plan/'PLAN.json'),uuid=a.uuid,
        driver_sha256=support.sha(__file__),mode=campaign.MODE,job=os.environ['SLURM_JOB_ID'],
        arms=list(arms),production_reservation=False))
    rows=[]
    try:
        for name in arms:
            unit=a.out/name;unit.mkdir()
            shutil.copytree(a.plan,unit/'plan')
            support.write(unit/'ENTRY.json',dict(split='train',evaluation='train',name='benchmark'))
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--bundle',str(bundle),'--plan',str(a.plan),
                 '--out',str(unit),'--uuid',a.uuid,'--parent',str(os.getpid())]
            if name=='cold':cmd.append('--cold')
            start=time.monotonic()
            with (unit/'worker.log').open('x') as log:
                subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
            elapsed=time.monotonic()-start
            comparisons=[]
            for c in plan['cases']:
                result=collect.verify_attempt(unit/'plan/cases'/c['label']/'attempts/attempt-00000')
                if c['kind']=='matched_control':
                    if (result['status']!='complete' or not result['quality']['integrity_pass']
                            or result['quality']['eligible_for_wall_free_export']!=c['provenance']['expected_wall_free']):
                        raise ValueError('Functionality control failed')
                elif result['status'] not in ('complete','placement_rejected'):
                    raise ValueError('Incomplete coverage attempt')
            if name!='cold':
                for c in plan['cases']:
                    if c['kind']!='coverage' or c['provenance'].get('placement_reasons'):continue
                    suffix=Path('plan/cases')/c['label']/'attempts/attempt-00000/raw.h5'
                    comparisons.append(dict(label=c['label'],**four.compare(a.out/'cold'/suffix,unit/suffix)))
            row=dict(arm=name,seconds=elapsed,comparisons=comparisons,
                passed=all(c['passed'] for c in comparisons),cache=support.read(unit/'KERNEL_CACHE.json'))
            support.write(unit/'BENCHMARK.json',row);rows.append(row)
            print(json.dumps(dict(arm=name,seconds=elapsed,passed=row['passed'])),flush=True)
        warm_rows=[r for r in rows if r['arm'].startswith('reuse-warm')]
        warm=sum(r['seconds'] for r in warm_rows)/len(warm_rows)
        passed=all(r['passed'] for r in rows)
        support.write(a.out/'RESULT.json',dict(arms=rows,passed=passed,
            cold_over_warm_speedup=rows[0]['seconds']/warm,
            training_release=False,qualification_waived=False))
        if not passed:raise ValueError('Numerical comparison failed; diagnostic results are not approval')
    except BaseException as exc:
        support.write(a.out/'FAILURE.json',dict(error=repr(exc),completed_arms=len(rows)))
        raise


if __name__=='__main__':main()
