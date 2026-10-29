"""Source-rest adapter with explicit simulator checkout and asset roots.

No implicit report imports, USD fallback, randomization, rescaling or rest refit.
The external pinned parent supplies the robot and existing physical defaults.
"""
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
from .inputs import load_input, REST_FIELDS

def check_module_roots(sim1_source):
    """Fail before cached top-level simulator modules can select foreign assets."""
    source = Path(sim1_source).resolve(strict=True)
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in ('envs', 'components', 'scripts'):
            path = getattr(module, '__file__', None)
            if path and not Path(path).resolve().is_relative_to(source):
                raise ValueError('Conflicting already-loaded simulator module: ' + name)


def build_environment(path, *, device='cpu', sim1_source, assets_root):
    import warp as wp
    import newton

    path, sim1_source = Path(path).resolve(), Path(sim1_source).resolve()
    check_module_roots(sim1_source)
    a = load_input(path)
    if os.environ.get('SIM1_TABLE_HALF_XY'):
        raise ValueError('Explicit fixed-table qualification does not accept implicit table overrides')
    assets_root = Path(assets_root).resolve(strict=True)
    if not assets_root.is_dir():
        raise ValueError('Explicit SIM1 asset directory required')
    os.environ['SIM1_ASSETS_ROOT'] = str(assets_root)
    sys.path.insert(0, str(sim1_source))
    parent = importlib.import_module('envs.lift2_short_shirt')
    if Path(parent.__file__).resolve() != sim1_source / 'envs/lift2_short_shirt.py':
        raise RuntimeError('A different SIM1 environment was already imported')
    if not Path(parent._ACONE_URDF_PATH).resolve().is_relative_to(assets_root):
        raise ValueError('Cached robot asset path differs from explicit asset root')

    class SourceRestEnvironment(parent.LiftShortShirt):
        def _create_env(self):
            b = self._env
            b.add_urdf(parent._ACONE_URDF_PATH, floating=False, enable_self_collisions=False,
                       xform=wp.transform(p=wp.vec3(0., 0., self.ROBOT_BASE_HEIGHT)))
            self._identify_robot_joint_indices()
            table_body = b.add_body()
            b.add_joint_fixed(-1, table_body)
            self.table_shape = b.add_shape_box(table_body,
                xform=wp.transform(wp.vec3(.65, 0., .4), wp.quat_identity()),
                hx=.5, hy=1., hz=.4,
                cfg=newton.ModelBuilder.ShapeConfig(mu=1., has_particle_collision=True,
                                                    has_shape_collision=False))
            for i in range(len(b.shape_material_mu)):
                b.shape_material_mu[i] = 1.  # Match EFFECTIVE live settings.
                b.shape_material_ka[i] = .002
                b.shape_is_solid[i] = True
            if b.particle_count or b.tri_count or b.edge_count:
                raise ValueError('Garment must be the only soft body in this adapter')
            q = a['vertices_world_m'].astype(np.float32)
            b.add_cloth_mesh(vertices=q.tolist(), indices=a['triangles'].ravel().tolist(),
                pos=wp.vec3(0.), rot=wp.quat_identity(), scale=1., vel=wp.vec3(0.),
                density=self.CLOTH_DENSITY, tri_ke=self.tri_ke, tri_ka=self.tri_ka,
                tri_kd=self.tri_kd, edge_ke=self.bending_ke, edge_kd=self.bending_kd,
                particle_radius=self.cloth_particle_radius)
            for k, value in [('particle_q', q), ('tri_indices', a['triangles']),
                             ('edge_indices', a['edge_indices'])]:
                if not np.array_equal(np.asarray(getattr(b, k)), value):
                    raise ValueError(f'Builder reordered {k}')
            for field, key in REST_FIELDS.items():
                setattr(b, field, a[key].tolist())
            b.color()
            b.add_ground_plane()
            self.cloth_triangles = a['triangles'].copy()
            self.cloth_asset_name = path.parent.name
            self.cloth_scale_actual = 1.
            self.pos_rand, self.rot_rand = wp.vec3(0.), wp.quat_identity()
            self.cloth_world_pos = np.zeros(3, np.float32)
            self.cloth_world_rot_xyzw = np.array([0., 0., 0., 1.], np.float32)

    with wp.ScopedDevice(device):
        env = SourceRestEnvironment(SimpleNamespace(rand_pos=wp.vec3(0.), rand_rot=wp.quat_identity()))
        model = env.model
    for field, key in REST_FIELDS.items():
        actual = getattr(model, field).numpy()
        if not np.array_equal(actual, a[key].astype(actual.dtype)):
            raise ValueError(f'Finalization changed {field}')
    if model.spring_count or not np.array_equal(model.particle_q.numpy(), a['vertices_world_m'].astype(np.float32)):
        raise ValueError('Unexpected springs or particle placement')
    return env, model, a
