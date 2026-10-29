"""Saved-point fragmentation screening, NOT calibrated fracture detection.

Each original object/component is tracked separately. A persistent new spatial
split is quarantined for review; sparse surface sampling and extreme continuous
stretch can still trigger it. Never reconnect points or mutate trajectories.
"""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class FragmentationPolicy:
    version: str = 'saved-point-fragmentation-review-v1'
    spacing_multiplier: float = 2.5
    grid_multiplier: float = 2.
    minimum_fraction: float = .01
    minimum_particles: int = 4
    consecutive_states: int = 3

    def __post_init__(self):
        if (self.version != 'saved-point-fragmentation-review-v1'
                or not np.isfinite([self.spacing_multiplier, self.grid_multiplier, self.minimum_fraction]).all()
                or self.spacing_multiplier <= 0 or self.grid_multiplier <= 0
                or not 0 < self.minimum_fraction < .5
                or type(self.minimum_particles) is not int or self.minimum_particles < 2
                or type(self.consecutive_states) is not int or self.consecutive_states < 2):
            raise ValueError('Invalid fragmentation policy')


def _components(points, radius):
    pairs = cKDTree(points).query_pairs(radius, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points))).tocsr()
    return connected_components(graph, directed=False)


def analyze(x, object_ids, *, dx_m, selected=None, policy=None):
    policy = policy or FragmentationPolicy()
    x = np.asarray(x, dtype=float); ids = np.asarray(object_ids)
    if (x.ndim != 3 or x.shape[-1] != 3 or len(x) < 2 or x.shape[1] < 2
            or ids.shape != x.shape[1:2] or ids.dtype.kind not in 'iu'
            or not np.isfinite(x).all() or not np.isfinite(dx_m) or dx_m <= 0):
        raise ValueError('Invalid fragmentation input')
    if selected is not None:
        selected = np.asarray(selected)
        if selected.shape != ids.shape or not np.isin(selected, [0, 1]).all():
            raise ValueError('Invalid fragmentation selection')
    rows = []; reasons = []
    for oid in np.unique(ids):
        points = x[:, ids == oid]; n = points.shape[1]
        if n < 4: raise ValueError('At least four particles per object required')
        spacing = float(np.median(cKDTree(points[0]).query(points[0], k=4)[0][:, -1]))
        radius = max(policy.spacing_multiplier * spacing, policy.grid_multiplier * dx_m)
        nc, initial = _components(points[0], radius)
        minimum = max(policy.minimum_particles, int(np.ceil(n * policy.minimum_fraction)))
        # Track the same separated material points through consecutive states;
        # different one-frame outliers cannot accumulate into a persistent event.
        streak = np.zeros(n, dtype=int); counts = []; confirmed = []
        for frame in points:
            detached = np.zeros(n, dtype=bool)
            for component in range(nc):
                idx = np.flatnonzero(initial == component)
                if len(idx) < 2 * minimum: continue
                _, labels = _components(frame[idx], radius)
                sizes = np.bincount(labels); largest = int(np.argmax(sizes))
                for part, size in enumerate(sizes):
                    if part != largest and size >= minimum:
                        detached[idx[labels == part]] = True
            streak = np.where(detached, streak + 1, 0)
            counts.append(int(detached.sum()))
            confirmed.append(int(np.sum(streak >= policy.consecutive_states)))
        flagged = np.flatnonzero(np.asarray(confirmed) >= minimum)
        row = dict(object_id=int(oid), initial_components=int(nc), radius_m=radius,
                   initial_third_neighbor_median_m=spacing, minimum_particles=minimum,
                   separated_particles=counts, persistent_separated_particles=confirmed,
                   flagged_states=flagged.tolist(), max_separated_fraction=max(counts)/n)
        if selected is not None:
            patch = selected[ids == oid].astype(bool)
            if patch.any() and not patch.all():
                row['patch_body_min_gap_m'] = [float(cKDTree(p[~patch]).query(p[patch])[0].min()) for p in points]
                row['unselected_com_z_excursion_m'] = float(np.max(points[:, ~patch, 2].mean(1) - points[0, ~patch, 2].mean()))
        if len(flagged): reasons.append(f'object_{int(oid)}:persistent_new_spatial_split')
        rows.append(row)
    return dict(policy=asdict(policy), objects=rows, quarantine=bool(reasons), reasons=reasons,
                calibrated=False, production_approved=False,
                limitation='Saved-point proximity, not mesh topology or material damage. Sparse sampling and extreme stretching can cause false positives; visual review required.')
