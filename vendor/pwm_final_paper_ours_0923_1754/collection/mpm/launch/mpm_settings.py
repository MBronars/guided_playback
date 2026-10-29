"""Validated collection choices, separate from the pinned physics protocol."""
import copy
import hashlib
import json
import math

SINGLE=('drop','translate')
MULTI=('collision','drop_on','stack','scatter','random')
DEFAULT=dict(split='train',evaluation='train',object_counts=[1,2,3,4],
    object_weights=[1,1,1,1],action_weights={k:1 for k in SINGLE+MULTI},
    max_attempts=120000,target_retained=None,
    hardware=dict(gpus=4,gpu_type='RTX_PRO_6000',walltime_hours=48))
VERSION='mpm-collection-settings-v1'


def normalize(value=None):
    if value is None:value={}
    if not isinstance(value,dict) or set(value)-set(DEFAULT):raise ValueError('Unknown collection setting')
    result=copy.deepcopy(DEFAULT);result.update(copy.deepcopy(value))
    split,evaluation=result['split'],result['evaluation']
    if split not in ('train','validation','test') or evaluation not in ('train','seen','unseen','count_seen','count_unseen'):
        raise ValueError('Unknown split or evaluation category')
    if (split=='train')!=(evaluation=='train'):raise ValueError('Training/evaluation mismatch')
    allowed=[5,6,7] if evaluation.startswith('count_') else [1,2,3,4]
    if 'object_counts' not in value:result['object_counts']=allowed
    counts=result['object_counts']
    if not isinstance(counts,list) or not counts or any(type(n) is not int or n not in allowed for n in counts) or len(set(counts))!=len(counts):
        raise ValueError('Object counts must be distinct members of '+str(allowed))
    if 'object_weights' not in value:result['object_weights']=[1]*len(counts)
    weights=result['object_weights']
    if not isinstance(weights,list) or len(weights)!=len(counts):raise ValueError('One object weight per count required')
    def positive(x):return type(x) in (int,float) and math.isfinite(x) and x>0
    if not all(positive(x) for x in weights):raise ValueError('Object weights must be finite and positive')
    actions=result['action_weights']
    if not isinstance(actions,dict) or not actions or set(actions)-set(SINGLE+MULTI) or not all(positive(x) for x in actions.values()):
        raise ValueError('Action weights require known actions and finite positive weights; omit disabled actions')
    for n in counts:
        if not set(actions)&set(SINGLE if n==1 else MULTI):raise ValueError('No compatible action for object count '+str(n))
    budget=result['max_attempts'];target=result['target_retained']
    if type(budget) is not int or not 4<=budget<=120000:raise ValueError('max_attempts must be 4..120000 coverage attempts, excluding controls')
    if target is not None and (type(target) is not int or not 1<=target<=budget):raise ValueError('target_retained must be null or 1..max_attempts')
    hardware=result['hardware']
    if not isinstance(hardware,dict) or set(hardware)!=set(DEFAULT['hardware']):raise ValueError('Explicit gpus, gpu_type and walltime_hours required')
    if type(hardware['gpus']) is not int or hardware['gpus'] not in (2,4) or hardware['gpu_type']!='RTX_PRO_6000':
        raise ValueError('Only two/four-PRO6000 startup/collection contracts are supported')
    if type(hardware['walltime_hours']) is not int or not 12<=hardware['walltime_hours']<=48:
        raise ValueError('walltime_hours must be 12..48; preserve the five-hour shutdown reserve')
    return result


def identity(value):
    return hashlib.sha256(json.dumps(normalize(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def bank_role(value):
    r=normalize(value)
    return r['split'] if r['evaluation'] in ('unseen','count_unseen') else 'train'


def capability(value):
    r=normalize(value)
    return dict(launch_supported=r['split']=='train',preview_supported=True,bank_role=bank_role(r),
        reason=None if r['split']=='train' else 'Evaluation generation is not approved by the fast training runner; use preview for CPU-only preparation.')


def cycles(value):
    r=normalize(value);n=r['hardware']['gpus']
    return max(n,n*math.ceil(r['max_attempts']/(60*n)))


def attempts(value,index):
    r=normalize(value);count=cycles(r)
    if type(index) is not int or not 0<=index<count:raise ValueError('Outside configured cycle budget')
    q,remainder=divmod(r['max_attempts'],count)
    return q+int(index<remainder)


def sampling(value):
    r=normalize(value)
    return {k:r[k] for k in ('split','evaluation','object_counts','object_weights','action_weights')}


def specifications(metadata, *, recipe, cycle, limit):
    """Keep default inputs exact; custom mixtures use deterministic local RNGs.

    Weights are proposal probabilities, not guaranteed retained proportions.
    Preview outputs are not production reservations or qualification evidence.
    """
    r=normalize(recipe)
    if metadata.get('bank_role')!=bank_role(r) or metadata.get('schema')!='fresh-candidate-bank-v1':
        raise ValueError('Shape bank does not match configured split/evaluation')
    if type(cycle) is not int or not 0<=cycle<=100000 or type(limit) is not int or not 1<=limit<=60:
        raise ValueError('Bounded cycle and 1..60 attempts required')
    from collection.mpm.protocol import specifications as original
    # Preserve the historical balanced stream exactly for the original recipe.
    if sampling(r)==sampling(DEFAULT) and limit==60:
        return original(metadata,split='train',evaluation='train',cycles=1,cycle_start=cycle)
    import numpy as np
    signature=hashlib.sha256(json.dumps(sampling(r),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if len(metadata['objects'])<max(r['object_counts']):raise ValueError('Insufficient distinct assets')
    weights=np.asarray(r['object_weights'],float);weights/=weights.max();weights/=weights.sum()
    rows=[]
    for index in range(limit):
        key=f'fresh-shapes-v1:configured:{signature}:{cycle}:{index}'
        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')
        rng=np.random.default_rng(seed)
        n=int(rng.choice(r['object_counts'],p=weights))
        actions=[a for a in (SINGLE if n==1 else MULTI) if a in r['action_weights']]
        aw=np.array([r['action_weights'][a] for a in actions],float);aw/=aw.max();aw/=aw.sum()
        action=str(rng.choice(actions,p=aw))
        # The scene adds both offsets to choose stiffness. Advance them at
        # different rates so each material spans all three pinned stiffnesses.
        rows.append(dict(index=index,count=n,scenario=action,seed=seed,
            recipe_version='fresh-shapes-v1',material_offset=(index+cycle)%3,
            stiffness_phase=(index//3+cycle)%3,
            asset_indices=rng.choice(len(metadata['objects']),size=n,replace=False).tolist(),
            split=r['split'],evaluation=r['evaluation'],cycle=cycle,family_index=index,
            seed_namespace=key))
    return rows
