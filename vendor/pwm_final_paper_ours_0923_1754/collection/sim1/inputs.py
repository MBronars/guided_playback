"""Frozen source-rest array contracts; validation does not certify garment physics."""
from pathlib import Path
import hashlib
import numpy as np

REST_FIELDS = {
    'tri_poses': 'tri_poses', 'tri_areas': 'tri_areas_m2',
    'particle_mass': 'particle_mass_kg', 'edge_rest_angle': 'edge_rest_angle_rad',
    'edge_rest_length': 'edge_rest_length_m',
    'edge_bending_properties': 'edge_bending_properties',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_input(a):
    """Validate storage/index/unit contracts, not physical garment acceptance."""
    shapes = {'vertices_world_m': (None, 3), 'triangles': (None, 3),
              'edge_indices': (None, 4), 'panel_names': (None,)}
    for k, shape in shapes.items():
        x = np.asarray(a[k])
        if x.ndim != len(shape) or any(n is not None and x.shape[i] != n for i, n in enumerate(shape)) or not len(x):
            raise ValueError(f'Invalid shape: {k}')
    n, f, e = len(a['vertices_world_m']), len(a['triangles']), len(a['edge_indices'])
    shapes.update(tri_poses=(f, 2, 2), tri_areas_m2=(f,), particle_mass_kg=(n,),
                  edge_rest_angle_rad=(e,), edge_rest_length_m=(e,),
                  edge_bending_properties=(e, 2), face_panel=(f,),
                  material_triangle_uv_m=(f, 3, 2))
    for k, shape in shapes.items():
        x = np.asarray(a[k])
        if k == 'panel_names':
            if x.dtype.kind != 'U' or len(set(x.tolist())) != len(x):
                raise ValueError('Unique Unicode panel names required')
            continue
        if x.ndim != len(shape) or any(n is not None and x.shape[i] != n for i, n in enumerate(shape)):
            raise ValueError(f'Invalid shape: {k}')
        if x.dtype.kind not in 'fiu' or not np.isfinite(x).all():
            raise ValueError(f'Nonfinite or nonnumeric: {k}')
        if not np.isfinite(x.astype(np.float32)).all():
            raise ValueError(f'Float32 overflow: {k}')
    for k in ('triangles', 'edge_indices', 'face_panel'):
        if a[k].dtype.kind not in 'iu':
            raise ValueError(f'Integer indices required: {k}')
    t, edges = a['triangles'], a['edge_indices']
    if np.any((t < 0) | (t >= n)) or np.any(np.diff(np.sort(t, axis=1), axis=1) == 0):
        raise ValueError('Invalid triangle indices')
    if np.any((edges < -1) | (edges >= n)) or np.any(edges[:, 2:] < 0):
        raise ValueError('Invalid bending adjacency')
    if np.any((a['face_panel'] < 0) | (a['face_panel'] >= len(a['panel_names']))):
        raise ValueError('Invalid panel membership')
    for k in ('tri_areas_m2', 'particle_mass_kg', 'edge_rest_length_m'):
        if np.any(a[k] <= 0):
            raise ValueError(f'Nonpositive reference: {k}')
    if np.any(a['edge_bending_properties'] < 0) or np.any(a['edge_rest_angle_rad'] != 0):
        raise ValueError('This release requires nonnegative bending and flat source dihedrals')
    # Reference determinant and lumped mass catch inconsistent units/corruption.
    area = .5 / np.abs(np.linalg.det(a['tri_poses']))
    if not np.allclose(area, a['tri_areas_m2'], rtol=1e-6, atol=1e-12):
        raise ValueError('Reference inverse metric/area mismatch')
    uv = a['material_triangle_uv_m']
    d1, d2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    uv_area = .5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    if not np.allclose(uv_area, area, rtol=1e-6, atol=1e-12):
        raise ValueError('Source UV/area mismatch')
    uv_inverse = np.linalg.inv(np.stack((d1, d2), axis=-1))
    rest = a['tri_poses']
    if not np.allclose(rest @ rest.transpose(0, 2, 1),
                       uv_inverse @ uv_inverse.transpose(0, 2, 1), rtol=1e-6, atol=1e-7):
        raise ValueError('Source UV/reference metric mismatch (basis-independent)')
    mass = np.zeros(n)
    for corner in range(3):
        np.add.at(mass, t[:, corner], 2. * area / 3.)
    if not np.allclose(mass, a['particle_mass_kg'], rtol=1e-6, atol=1e-12):
        raise ValueError('Source-area density 2 kg/m2 / mass mismatch')
    return a


def load_input(path):
    with np.load(path, allow_pickle=False) as z:
        return validate_input({k: z[k].copy() for k in z.files})
