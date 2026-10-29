"""Collision-seeking layouts and honest saved-particle interaction proxies.

This changes sampling, not the MPM contact law. Worlds still have independent
grids; objects within a world share a grid. Nearest surface-sample distance is
NOT an exact mesh-contact detector and motion transfer is not a force sensor.
"""
import numpy as np

POLICY = 'drag-interaction-mixture-v1'
MIXTURE = dict(separated=.3, pair=.5, crowded=.2)


def mode_for(index, counts):
    """30/50/20 per multi-object count over each ten occurrences of that count."""
    n = counts[index % len(counts)]
    if n == 1:
        return 'single'
    slot = (index // len(counts)) % 10
    return 'separated' if slot < 3 else 'pair' if slot < 8 else 'crowded'


def place(initial, geometry, action, rng, mode):
    """Translate original point clouds; rotate the whole horizontal command plan.

    A targeted all-lift sequence gets a final pull (including one-event clips).
    Otherwise the first horizontal event is aimed at the neighbour, and reverse
    references rotate with it, preserving single-leg v3 semantics. Durations are
    extended toward a 14 cm command path where the sampled speed allows it; slow
    commands may still miss. No simulation outcome is resampled or repaired.
    """
    from .protocol import FRAME_DT
    if mode not in ('single', 'separated', 'pair', 'crowded'):
        raise ValueError('Unknown interaction mode')
    ids = initial['object_ids']; n = len(initial['E']); driven = action['object_id']
    if (n == 1) != (mode == 'single'):
        raise ValueError('Single-object scenes are their own control stratum')
    centers = []; local = []
    for oid in range(n):
        x = initial['x'][ids == oid]
        center = (x.min(0) + x.max(0)) / 2
        centers.append(center); local.append(x[:, :2] - center[:2])
    target_pairs = []; aimed_event = None; gap = None
    replaced_lift = False
    if mode in ('single', 'separated'):
        xy = [np.array([.75, .75]) + (0 if n == 1 else .36) *
              np.array([np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n)]) for i in range(n)]
    else:
        heading = float(rng.uniform(-np.pi, np.pi))
        direction = np.array([np.cos(heading), np.sin(heading)])
        lateral = np.array([-direction[1], direction[0]])
        gap = float(rng.uniform(.035, .055))
        order = [driven] + [i for i in range(n) if i != driven]
        chain = order if mode == 'crowded' else order[:2]
        lo = [float((p @ direction).min()) for p in local]
        hi = [float((p @ direction).max()) for p in local]
        distances = [0.]
        for left, right in zip(chain, chain[1:]):
            distances.append(distances[-1] + hi[left] - lo[right] + gap)
            target_pairs.append(dict(a=left, b=right))
        midpoint = (lo[chain[0]] + distances[-1] + hi[chain[-1]]) / 2
        xy = [None] * n
        for oid, distance in zip(chain, distances):
            xy[oid] = np.array([.75, .75]) + (distance-midpoint)*direction
        # Uninvolved pair-scene distractors occupy a separate parallel row.
        for j, oid in enumerate(order[2:] if mode == 'pair' else []):
            xy[oid] = np.array([.75, .75]) + (j-(n-3)/2)*.30*direction + .38*lateral
        events = action['events']
        horizontal = [i for i, e in enumerate(events) if e['family'] != 'lift_hold_release']
        if not horizontal:
            events[-1].update(family='pull_release', hold=0)
            horizontal = [len(events)-1]; replaced_lift = True
        aimed_event = horizontal[0]; event = events[aimed_event]
        actual_angle = event['angle'] + (np.pi if event['family'] == 'reverse_release' else 0.)
        rotation = heading - actual_angle
        for e in events:
            if e['family'] != 'lift_hold_release':
                a = e['angle'] + rotation
                e['angle'] = float(np.arctan2(np.sin(a), np.cos(a)))
        max_moving = min(20, int(np.floor(.25/(event['speed']*FRAME_DT))))
        event['moving'] = min(max_moving, max(event['moving'], int(np.ceil(.14/(event['speed']*FRAME_DT)))))
    for oid in range(n):
        sel = ids == oid; x = initial['x'][sel]
        shift = np.r_[xy[oid]-centers[oid][:2], .115-x[:, 2].min()].astype(np.float32)
        initial['x'][sel] += shift
        geometry['transforms'][oid]['drag_layout_shift_m'] = shift.tolist()
    return dict(policy=POLICY, mode=mode, requested_pairs=target_pairs,
                crowded_two_object_fallback=mode == 'crowded' and n == 2,
                projected_initial_gap_m=gap, aimed_event=aimed_event,
                all_lift_final_event_changed_to_pull=replaced_lift,
                outcome_guaranteed=False)


def measure(manifest, arrays):
    """Frame-level proximity + undriven COM movement after the first drive.

    Require at least two near frames (<=2 cm) AND >=5 mm target translation
    after that proximity to label a *transfer candidate*. Settling contact and
    baseline motion are reported separately. These are proxies, not exact
    collision/impulse measurements. Retention is independent of these labels.
    """
    from scipy.spatial import cKDTree
    layout = manifest.get('geometry_provenance', {}).get('drag_interaction', {})
    x = arrays['x'].astype(float); ids = arrays['object_ids']
    first = manifest['action']['events'][0]['start']; pairs = []
    for request in layout.get('requested_pairs', []):
        left, right = request['a'], request['b']
        a = x[:, ids == left]; b = x[:, ids == right]
        gaps = np.array([cKDTree(pa).query(pb)[0].min() for pa, pb in zip(a, b)])
        com = b.mean(1); baseline = com[first].copy()
        near = np.flatnonzero((gaps <= .02) & (np.arange(len(gaps)) >= first))
        displacement = np.linalg.norm(com-baseline, axis=-1)
        post = float(np.linalg.norm(com[near[0]:]-com[near[0]], axis=-1).max()) if len(near) else 0.
        pairs.append(dict(a=left, b=right, initial_gap_m=float(gaps[0]),
            minimum_gap_m=float(gaps.min()), near_frames_after_drive=int(len(near)),
            first_near_frame=int(near[0]) if len(near) else None,
            target_max_com_displacement_m=float(displacement[first:].max()),
            target_post_proximity_displacement_m=post,
            target_pre_drive_displacement_m=float(np.linalg.norm(com[:first+1]-com[0], axis=-1).max()),
            final_gap_m=float(gaps[-1]),
            transfer_candidate=bool(len(near) >= 2 and post >= .005)))
    return dict(policy=POLICY, mode=layout.get('mode', 'legacy_separated'),
                proximity_threshold_m=.02, transfer_threshold_m=.005,
                exact_contact_measurement=False, pairs=pairs,
                any_transfer_candidate=any(p['transfer_candidate'] for p in pairs))
