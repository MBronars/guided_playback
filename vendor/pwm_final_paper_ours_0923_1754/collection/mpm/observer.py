"""Read-only observers around the unchanged, instance-local Warp colliders.

Floor and object contact are allowed. Original physical planes remain .03/1.47m
even on a padded computational grid. No force, position, mass or velocity edits.
"""
from pathlib import Path

import numpy as np
import warp as wp

from .walls import POINTS, NORMALS, metadata
from .physics import MPMStateStruct
from .state import MPMModelStruct
from collection.common import digest as sha256

NEVER = 2147483647


@wp.struct
class Observation:
    point: wp.vec3
    normal: wp.vec3
    wall: int
    step: wp.array(dtype=int)
    first_grid: wp.array(dtype=int)
    covered: wp.array(dtype=int)


@wp.kernel
def observe_grid(time: float, dt: float, state: MPMStateStruct,
                 model: MPMModelStruct, param: Observation):
    i, j, k = wp.tid()
    if i == 0 and j == 0 and k == 0:
        param.covered[param.wall] += 1
    offset = wp.vec3(float(i) * model.dx, float(j) * model.dx, float(k) * model.dx) - param.point
    if state.grid_m[i, j, k] > 0.0 and wp.dot(offset, param.normal) < 0.0:
        if wp.dot(state.grid_v_out[i, j, k], param.normal) < 0.0:
            wp.atomic_min(param.first_grid, param.wall, param.step[0])


@wp.kernel
def next_step(step: wp.array(dtype=int)):
    step[0] += 1


@wp.kernel
def observe_particles(x: wp.array(dtype=wp.vec3), points: wp.array(dtype=wp.vec3),
                      normals: wp.array(dtype=wp.vec3), step: wp.array(dtype=int),
                      first: wp.array(dtype=int)):
    particle, wall = wp.tid()
    if wp.dot(x[particle] - points[wall], normals[wall]) <= 0.0:
        wp.atomic_min(first, wall, step[0])


def signature(params):
    return tuple((tuple(p.point), tuple(p.normal), int(p.surface_type), float(p.friction),
                  float(p.start_time), float(p.end_time)) for p in params)


class WallObservedBackend:
    """Fresh passive episodes only; checkpoint continuation is deliberately absent."""
    def __init__(self, backend, clock):
        self.backend = backend
        self.clock = clock
        self.device = backend.device
        s = backend.solver
        if backend.steps or backend.failed or backend.config['dt_s'] != clock.substep_dt_s:
            raise ValueError('Wall observation requires a fresh fixed-clock backend')
        if clock.transitions * clock.substeps_per_transition >= NEVER:
            raise ValueError('Wall step counter would overflow')
        if len(s.collider_params) != 6 or len(s.grid_postprocess) != 6 or s.modify_bc != [None] * 6:
            raise ValueError('Only original floor plus five unmodified colliders supported')
        original = tuple(s.collider_params)
        for p, point, normal in zip(original[1:], POINTS, NORMALS):
            if (not np.array_equal(np.array(p.point), point) or
                    not np.array_equal(np.array(p.normal), normal) or
                    p.surface_type != 2 or p.friction != 0 or p.start_time > 0 or
                    p.end_time <= clock.state_time_s[-1]):
                raise ValueError('Unsupported physical wall configuration')
        floor = original[0]
        if (not np.array_equal(np.array(floor.normal), [0, 0, 1]) or
                not np.array_equal(np.array(floor.point), np.array([0, 0, backend.config['floor_m']], np.float32)) or
                floor.surface_type != 2 or floor.start_time > 0 or floor.end_time <= clock.state_time_s[-1]):
            raise ValueError('Unsupported floor configuration')
        self.original = original
        self.frozen_signature = signature(original)
        self.counter = wp.zeros(1, dtype=int, device=self.device)
        self.first_grid = wp.full(5, NEVER, dtype=int, device=self.device)
        self.first_particle = wp.full(5, NEVER, dtype=int, device=self.device)
        self.covered = wp.zeros(5, dtype=int, device=self.device)
        self.points = wp.array(POINTS, dtype=wp.vec3, device=self.device)
        self.normals = wp.array(NORMALS, dtype=wp.vec3, device=self.device)
        kernels, params, modifiers = [], [], []
        for index, (kernel, param) in enumerate(zip(s.grid_postprocess, original)):
            if index:
                observer = Observation()
                observer.point, observer.normal = param.point, param.normal
                observer.wall = index - 1
                observer.step, observer.first_grid, observer.covered = self.counter, self.first_grid, self.covered
                kernels.append(observe_grid); params.append(observer); modifiers.append(None)
            kernels.append(kernel); params.append(param); modifiers.append(None)
        s.grid_postprocess, s.collider_params, s.modify_bc = kernels, params, modifiers
        self.frozen_topology = (tuple(kernels), tuple(params), tuple(modifiers))
        self.source_sha256 = dict(backend.source_sha256)
        for path in (Path(__file__).resolve(), Path(__file__).with_name('walls.py').resolve()):
            self.source_sha256[str(path)] = sha256(path)
        self.wall_metadata = metadata(sha256(__file__))
        self._particles()

    @property
    def steps(self):
        return self.backend.steps

    @property
    def config(self):
        return self.backend.config

    @property
    def failed(self):
        return self.backend.failed

    @property
    def solver(self):
        return self.backend.solver

    def _unchanged(self):
        s = self.backend.solver
        if ((tuple(s.grid_postprocess), tuple(s.collider_params), tuple(s.modify_bc)) != self.frozen_topology
                or signature(self.original) != self.frozen_signature):
            raise ValueError('Collider configuration changed during wall observation')

    def _particles(self):
        wp.launch(observe_particles, dim=(self.backend.n, 5), inputs=[
            self.backend.solver.mpm_state.particle_x, self.points, self.normals,
            self.counter, self.first_particle], device=self.device)

    def step(self, dt, command, selected):
        if command is not None or np.any(selected):
            raise ValueError('Wall-observed collector currently supports passive episodes only')
        if dt != self.clock.substep_dt_s or self.steps >= self.clock.transitions * self.clock.substeps_per_transition:
            raise ValueError('Wall observation clock exceeded or changed')
        self._unchanged()
        self._particles()
        wp.launch(next_step, dim=1, inputs=[self.counter], device=self.device)
        self.backend.step(dt, None, selected)
        self._particles()

    def read(self):
        self._unchanged()
        return self.backend.read()

    def wall_snapshot(self):
        self._unchanged()
        counter = int(self.counter.numpy()[0])
        covered = self.covered.numpy().astype(np.int64)
        if counter != self.steps or np.any(covered != counter):
            raise ValueError('Wall substep coverage incomplete')
        values = dict(wall_simulation_steps=np.array(counter, np.int64), wall_covered_substeps=covered)
        for name, source in [('wall_first_grid_step', self.first_grid), ('wall_first_particle_step', self.first_particle)]:
            a = source.numpy().astype(np.int64)
            values[name] = np.where(a == NEVER, -1, a)
        return values

    def diagnostics(self):
        # Keep a failed attempt's partial evidence, without certifying coverage.
        if self.failed:
            return dict(self.backend.diagnostics(), wall_observation=self.wall_metadata,
                        source_sha256=self.source_sha256, wall_evidence_complete=False)
        return dict(self.backend.diagnostics(), wall_observation=self.wall_metadata,
                    source_sha256=self.source_sha256,
                    wall_evidence={key: value.tolist() for key, value in self.wall_snapshot().items()})
