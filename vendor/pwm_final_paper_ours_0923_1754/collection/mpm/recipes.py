"""Predeclared 30-case surface-point coverage, separate from historical sampling.

Two seeds x five scenarios x 2/4/7 objects. Original points and weights are kept;
only proper rigid placement, initial velocity and declared material draws change.
No failed layout is repaired or replaced after outcomes are observed.
"""
import json
from pathlib import Path

import numpy as np

from collection.common import digest, relative_path
from .runtime import reference_root

SCENARIOS = ("collision", "drop_on", "stack", "scatter", "random")
INTERACTION_VERSIONS = ('interaction-v1', 'centered-v2', 'fresh-shapes-v1')
VERSIONS = ('legacy-v1', *INTERACTION_VERSIONS)



def specifications(version='legacy-v1', *, seed_start=2026090900, seed_count=2):
    if version == 'fresh-shapes-v1':
        raise ValueError('Fresh protocol requires explicit split-aware specifications')
    if version not in VERSIONS:
        raise ValueError('Unknown recipe version')
    if type(seed_start) is not int or not 0 <= seed_start <= 2**32 - 32:
        raise ValueError('Explicit nonnegative seed start within uint32 range required')
    if type(seed_count) is not int or not 1 <= seed_count <= 32:
        raise ValueError('Bounded seed count must be 1..32; use separate plans for further blocks')
    rows = [dict(index=i, count=n, scenario=scenario, seed=seed,
                 material_offset=(j + seed - 2026090900) % 3)
            for i, (n, j, scenario, seed) in enumerate(
                (n, j, scenario, seed) for seed in range(seed_start, seed_start + seed_count)
                for n in (2, 4, 7) for j, scenario in enumerate(SCENARIOS))]
    if version == 'legacy-v1':
        return rows
    # Every fixed geometry/material assignment sees all three stiffness phases.
    # Do not confound each material type with one E as the legacy pilot did.
    return [dict(row, index=phase * len(rows) + row['index'], recipe_version=version,
                 geometry_index=row['index'], stiffness_phase=phase)
            for phase in range(3) for row in rows]


def load_bank(bank=None, pool=None, *, role='train', split_manifest=None):
    from .banks import allowed_uids
    bank = Path(bank) if bank is not None else reference_root() / "collection_variability/run-02/bank"
    metadata = json.loads((bank / "MANIFEST.json").read_text())
    pool = Path(pool) if pool is not None else reference_root() / "source-snapshot/source/configs"
    for name, key in (("objaverse_uid_train.json", "train_pool_sha256"),
                      ("objaverse_uid_holdout.json", "heldout_pool_sha256")):
        if digest(pool / name) != metadata[key]:
            raise ValueError("Asset split changed")
    train = set(json.loads((pool / "objaverse_uid_train.json").read_text()))
    holdout = set(json.loads((pool / "objaverse_uid_holdout.json").read_text()))
    if train & holdout:
        raise ValueError("Asset pools overlap")
    allowed = allowed_uids(metadata, pool, role=role, split_manifest=split_manifest)
    objects = []
    for row in metadata["objects"]:
        if row["asset_uid"] not in allowed:
            raise ValueError("Asset outside declared bank role")
        path = relative_path(bank, row["file"])
        if digest(path) != row["sha256"]:
            raise ValueError("Bank payload changed")
        with np.load(path, allow_pickle=False) as data:
            obj = {key: data[key].copy() for key in data.files}
        if obj["source_x"].shape != (1024, 3) or not np.isfinite(obj['source_x']).all():
            raise ValueError("Original surface count differs")
        volume, indices = obj['particle_volume'], obj['source_particle_index']
        if volume.shape != (1024,) or not np.isfinite(volume).all() or np.any(volume <= 0):
            raise ValueError('Original finite positive surface weights required')
        if indices.shape != (1024,) or indices.dtype.kind not in 'iu' or indices.min() < 0 or len(np.unique(indices)) != 1024:
            raise ValueError('Unique original material particle identities required')
        if not np.isfinite(row['object_size_m']) or row['object_size_m'] <= 0:
            raise ValueError('Positive finite declared source object size required')
        objects.append(obj)
    if len(objects) < 7 or len({r["asset_uid"] for r in metadata["objects"]}) != len(objects):
        raise ValueError("Unique source bank required")
    return metadata, objects


def scene(spec, metadata, objects):
    from scipy.spatial import cKDTree
    n, scenario = spec["count"], spec["scenario"]
    version = spec.get('recipe_version', 'legacy-v1')
    interaction = version in INTERACTION_VERSIONS
    fresh = version == 'fresh-shapes-v1'
    centered_recipe = version in ('centered-v2', 'fresh-shapes-v1')
    if version not in VERSIONS:
        raise ValueError('Unknown recipe version')
    if fresh:
        from .protocol import cohort
        expected_role = cohort(spec['split'], spec['evaluation'])
        allowed = (5, 6, 7) if spec['evaluation'].startswith('count_') else (1, 2, 3, 4)
        if (n not in allowed or metadata.get('bank_role') != expected_role
                or metadata.get('schema') != 'fresh-candidate-bank-v1'
                or scenario not in (('drop', 'translate') if n == 1 else SCENARIOS)):
            raise ValueError('Outside fresh split/count/family contract')
    elif n not in (2, 4, 7) or scenario not in SCENARIOS:
        raise ValueError("Outside declared pilot matrix")
    # Shared source draw within each count/seed; scenarios differ by declared
    # placement/motion/material offset, not selected simulation outcomes.
    rng = np.random.default_rng(np.random.SeedSequence([spec["seed"], n]))
    picks = np.asarray(spec['asset_indices']) if fresh else rng.choice(len(objects), n, replace=False)
    if picks.shape != (n,) or picks.dtype.kind not in 'iu' or len(set(picks)) != n or picks.min() < 0 or picks.max() >= len(objects):
        raise ValueError('Invalid explicit unique object selection')
    positions, velocities, transforms, weights, extents = [], [], [], [], []
    for oid, pick in enumerate(picks):
        src = objects[pick]["source_x"].astype(np.float64)
        volume = objects[pick]['particle_volume']
        extent = 2 * metadata['objects'][pick]['object_size_m']
        if fresh:
            from .fresh_banks import physical
            extent = float(rng.uniform(.12, .20))
            src, volume = physical(objects[pick]['points_unit'], extent)
            src = src.astype(np.float64)
        weights.append(volume)
        extents.append(extent)
        angle = float(rng.uniform(-np.pi, np.pi))
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        midpoint = (src.min(0) + src.max(0)) / 2
        centered = (src - midpoint) @ rotation.T
        velocity = np.zeros(3)
        if fresh and n == 1:
            xy = np.array([.75, .75])
            bottom = float(rng.uniform(.28, .50))
            if scenario == 'translate':
                theta = rng.uniform(-np.pi, np.pi)
                velocity[:2] = .3 * np.array([np.cos(theta), np.sin(theta)])
        elif interaction and scenario == 'collision':
            pair = oid // 2
            if oid == n - 1 and n % 2:
                xy = np.array([.75, .75]) if centered_recipe else np.array([1.25, .2])
            else:
                y_bounds = (.48, 1.02) if centered_recipe else (.4, 1.1)
                y = .75 if n // 2 == 1 else np.linspace(*y_bounds, n // 2)[pair]
                # Facing AABB planes have 200 mm (v2) or 260 mm (v1)
                # clearance; 1.2 m/s relative approach reaches a C4/F16 future.
                left = oid % 2 == 0
                half_gap = .10 if centered_recipe else .13
                x = .75 + (-half_gap - centered[:, 0].max() if left else half_gap - centered[:, 0].min())
                xy = np.array([x, y - centered[:, 1].mean()])
                velocity[0] = .6 if left else -.6
            bottom = .4
            if centered_recipe:
                bottom = .36 if n == 7 else .28
                if oid == n - 1 and n % 2:
                    bottom = .102
        elif interaction and scenario in ('stack', 'drop_on'):
            columns = min(3, n // 2)
            column = oid % columns
            xy = np.array([.75, .75]) if columns == 1 else np.array(
                [.75 + .30 * (column - (columns - 1) / 2), .75] if centered_recipe else [.4 + .35 * column, .75])
            if oid < columns:
                bottom = .102
            else:
                # Falling stacks, not a claim of already-equilibrated contact.
                # Delay interaction beyond the first four context states.
                bottom = float(positions[oid - columns][:, 2].max()) + (.22 if scenario == 'stack' else .35)
        elif interaction and scenario == 'scatter':
            theta = 2 * np.pi * oid / n
            direction = np.array([np.cos(theta), np.sin(theta)])
            radius = (.14 if n == 2 else .22 if n <= 4 else .30) if centered_recipe else (.18 if n == 2 else .34)
            xy = np.array([.75, .75]) + radius * direction
            bottom = .30 if centered_recipe else .45
            velocity[:2] = (.45 if centered_recipe else .65) * direction
        elif interaction and scenario == 'random':
            if centered_recipe:
                columns = 2 if n <= 4 else 3
                grid = np.array([(i % columns, i // columns) for i in range(n)])
                xy = .75 + .30 * (grid[oid] - (grid.min(0) + grid.max(0)) / 2)
                bottom = float(rng.uniform(.20, .28))
                velocity[:2] = rng.uniform(-.18, .18, size=2)
            else:
                xy = np.array([.35 + .4 * (oid % 3), .35 + .4 * (oid // 3)])
                bottom = float(rng.uniform(.25, .55))
                velocity[:2] = rng.uniform(-.35, .35, size=2)
        elif scenario in ("collision", "scatter"):
            theta = 2 * np.pi * oid / n
            direction = np.array([np.cos(theta), np.sin(theta)])
            xy = np.array([.75, .75]) + .38 * direction
            bottom = .102
            velocity[:2] = (-.45 if scenario == "collision" else .12) * direction
        elif scenario == "drop_on":
            column = oid // 2
            xy = np.array([.48 + .54 * (column % 2), .48 + .54 * (column // 2)])
            bottom = .102 if oid % 2 == 0 else float(positions[-1][:, 2].max()) + .15
        elif scenario == "stack":
            # Two towers; finite declared 80 mm initial clearance, not a claim
            # of an equilibrated stack or forced particle contact.
            xy = np.array([.48 + .54 * (oid % 2), .75])
            bottom = .102 if oid < 2 else float(positions[oid - 2][:, 2].max()) + .08
        else:
            xy = np.array([.43 + .32 * (oid % 3), .43 + .32 * (oid // 3)])
            bottom = .102 + float(rng.uniform(0, .25))
            velocity[:2] = rng.uniform(-.25, .25, size=2)
        shift = np.r_[xy, bottom - centered[:, 2].min()]
        placed = (centered + shift).astype(np.float32)
        positions.append(placed)
        velocities.append(np.tile(velocity, (1024, 1)).astype(np.float32))
        transforms.append(dict(bank_index=int(pick), source_midpoint_m=midpoint.tolist(),
                               rotation=rotation.tolist(), translation_after_centering_m=shift.tolist()))
        if fresh:
            transforms[-1].update(maximum_full_mesh_extent_m=extent,
                                 canonical_weight_center_m=[.75, .75, .75],
                                 proxy_volume_m3=float(volume.astype(float).sum()))
    pairs = []
    for a in range(n):
        for b in range(a):
            measured = float(cKDTree(positions[a]).query(positions[b], k=1)[0].min())
            required = .8 * max(metadata["objects"][picks[a]]["object_size_m"],
                                metadata["objects"][picks[b]]["object_size_m"])
            if fresh:
                required = .4 * max(extents[a], extents[b])
            pairs.append(dict(a=a, b=b, gap_m=measured, required_m=required))
    x = np.concatenate(positions)
    reasons = []
    if x[:, :2].min() <= .03 or x[:, :2].max() >= 1.47 or x[:, 2].max() >= 1.47:
        reasons.append("initial_physical_wall_bounds")
    if x[:, 2].min() < np.float32(.102):
        reasons.append("initial_floor_clearance")
    if any(p["gap_m"] < p["required_m"] for p in pairs):
        reasons.append("initial_spacing")
    mats = (np.arange(n) + spec["material_offset"]) % 3
    static = dict(object_ids=np.repeat(np.arange(n, dtype=np.int32), 1024),
                  particle_volume=np.concatenate(weights),
                  E=np.array([1e4, 1e5, 1e6], np.float32)[(np.arange(n) + spec["material_offset"] +
                      (spec['stiffness_phase'] if interaction else 0)) % 3],
                  nu=np.full(n, .2, np.float32), density=np.full(n, 1000, np.float32),
                  mat_type=mats.astype(np.int32), yield_stress=np.where(mats == 1, 10000, 0).astype(np.float32))
    initial = dict(x=x, v=np.concatenate(velocities), **static,
                   source_particle_index=np.concatenate([objects[p]["source_particle_index"] for p in picks]))
    provenance = dict(asset_ids=[metadata["objects"][p]["asset_uid"] for p in picks],
                      transforms=transforms, pair_gaps=pairs, placement_reasons=reasons,
                      source_rows=[metadata["objects"][p] for p in picks],
                      material_policy="Cyclic mixed types/E; fixed nu=.2 rho=1000 plasticine yield=10000; not full factorial",
                      representation="1024 original points and raw weights per object; rigid placement only")
    if interaction:
        columns = min(3, n // 2)
        pairs = ([[i, i + 1] for i in range(0, n - 1, 2)] if scenario == 'collision' else
                 [[i - columns, i] for i in range(columns, n)] if scenario in ('stack', 'drop_on') else [])
        provenance.update(recipe_version=version, requested_pairs=pairs,
            material_policy='Same geometry/type assignments crossed with three E phases; nu=.2 rho=1000 archived plasticine yield=10000',
            interaction_semantics='Facing moving pairs' if scenario == 'collision' else
                'Vertically overlapping falling towers; not a stable terminal stack certificate' if scenario in ('stack', 'drop_on') else
                'Radial outward initial velocity' if scenario == 'scatter' else 'Random passive drop/translation',
            training_approval=False)
    if fresh:
        provenance.update(representation='1024 deterministic surface points; per-scene full extent; canonical occupied-cell proxy weights frozen before rigid placement',
                          full_extent_range_m=[.12, .20], volume_policy=metadata['volume_policy'],
                          proxy_volume_qualified=False, split=spec['split'], evaluation=spec['evaluation'])
    return initial, provenance
