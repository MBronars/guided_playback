"""Package-local numerical binding; reference assets are explicitly external.

The reference root contains frozen inputs, not executable modules. Historical
prepared runs must still use their original sources; migration starts new IDs.
"""
import os
from pathlib import Path

from collection.common import digest, relative_path

ROOT = Path(__file__).resolve().parents[2]
STATIC = ('object_ids', 'particle_volume', 'E', 'nu', 'density', 'mat_type', 'yield_stress')


def reference_root():
    value = os.environ.get('PWM_MPM_REFERENCE_ROOT')
    if not value:
        raise ValueError('Set PWM_MPM_REFERENCE_ROOT to the frozen MPM reference assets; no implicit reports path')
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError('Reference root must be a directory')
    return path


def load():
    from . import export, clock, physics, observer, contracts, validate
    return export, clock, physics, observer, contracts, validate


def sources():
    # Domain-specific sources plus shared helpers; unrelated SIM1 edits must not
    # invalidate a frozen MPM worker. Simulator dependencies remain separately
    # fingerprinted by the numerical backend.
    files = list((ROOT / 'collection').glob('*.py')) + list((ROOT / 'collection/mpm').rglob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p)
            for p in sorted(files)}


def verify_sources(expected):
    for name, value in expected.items():
        path = relative_path(ROOT, name)
        if digest(path) != value:
            raise ValueError('Prepared source changed: ' + name)


def backend(initial, manifest, device):
    _, _, physics, observer, contracts, _ = load()
    static = {key: initial[key] for key in STATIC}
    base = physics.CollectionBackend(initial['x'], static, manifest['solver_config'],
                                     initial['v'], device=device)
    physics.install_original_walls_in_padded_grid(base)
    return observer.WallObservedBackend(base, contracts.Clock(**manifest['clock']))
