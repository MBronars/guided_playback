"""Explicit training-checkout selection for independent collection review."""
import importlib
import os
from pathlib import Path
import sys


def import_consumer(root=None):
    root = Path(root or os.environ.get('PWM_MODEL_ROOT') or Path(__file__).resolve().parents[1]).resolve(strict=True)
    expected = root / 'data/unified_dataset.py'
    if not expected.is_file():
        raise ValueError('Model root has no data/unified_dataset.py: ' + str(root))
    for name, module in tuple(sys.modules.items()):
        if name == 'data' or name.startswith('data.'):
            file = getattr(module, '__file__', None)
            if file and not Path(file).resolve().is_relative_to(root / 'data'):
                raise ValueError('Conflicting already-loaded model package: ' + name)
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module('data.unified_dataset')
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != expected:
        raise ValueError('Wrong model adapter import')
    return root, module
