"""Fresh-shape passive protocol: explicit role/count matrix and coverage stream."""
import hashlib

import numpy as np

VERSION = 'fresh-shapes-v1'
EVALUATIONS = ('train', 'seen', 'unseen', 'count_seen', 'count_unseen')


def cohort(split, evaluation):
    if split not in ('train', 'validation', 'test') or evaluation not in EVALUATIONS:
        raise ValueError('Unknown split/evaluation')
    if (split == 'train') != (evaluation == 'train'):
        raise ValueError('Training/evaluation role mismatch')
    return split if evaluation in ('unseen', 'count_unseen') else 'train'


def specifications(metadata, *, split, evaluation, cycles=1, cycle_start=0):
    role = cohort(split, evaluation)
    if metadata.get('bank_role') != role or metadata.get('schema') != 'fresh-candidate-bank-v1':
        raise ValueError('Fresh shape bank with matching cohort required')
    if (type(cycles) is not int or not 1 <= cycles <= 1000
            or type(cycle_start) is not int or not 0 <= cycle_start <= 100000):
        raise ValueError('Explicit bounded cycle range required')
    counts = (5, 6, 7) if evaluation.startswith('count_') else (1, 2, 3, 4)
    # One permutation per split/evaluation; the cyclic stream guarantees shape
    # coverage after N object slots, unlike independent random scene draws.
    namespace = VERSION + ':' + split + ':' + evaluation
    order = sorted(range(len(metadata['objects'])), key=lambda i: hashlib.sha256(
        (namespace + ':' + metadata['objects'][i]['asset_uid']).encode()).hexdigest())
    if len(order) < max(counts):
        raise ValueError('Insufficient distinct objects')
    cursor = cycle_start * 15 * sum(counts)
    specs = []
    for cycle in range(cycle_start, cycle_start + cycles):
        for n in counts:
            for phase in range(3):
                for family in range(5):
                    scenarios = ('drop', 'translate') if n == 1 else ('collision', 'drop_on', 'stack', 'scatter', 'random')
                    scenario = scenarios[(family + cycle + phase) % len(scenarios)]
                    # RNG identity does not depend on bank draw outcomes, worker
                    # count, wall time, rejected layouts, or process ordering.
                    key = f'{namespace}:{cycle}:{n}:{phase}:{family}'
                    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')
                    picks = [order[(cursor + j) % len(order)] for j in range(n)]
                    cursor += n
                    specs.append(dict(index=len(specs), count=n, scenario=scenario, seed=seed,
                        recipe_version=VERSION, material_offset=(family + cycle) % 3,
                        stiffness_phase=phase, asset_indices=picks, split=split, evaluation=evaluation,
                        cycle=cycle, family_index=family, seed_namespace=key))
    return specs


def coverage(specs, metadata):
    from collections import Counter
    counts = Counter(metadata['objects'][i]['asset_uid'] for spec in specs for i in spec['asset_indices'])
    return dict(attempts=len(specs), planned_unique_shapes=len(counts), bank_shapes=len(metadata['objects']),
                object_slots=sum(counts.values()), per_shape_attempted_appearances=dict(sorted(counts.items())),
                note='Planned membership only; retained coverage must be measured after filtering')
