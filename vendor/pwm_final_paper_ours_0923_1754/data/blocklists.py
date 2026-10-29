"""Shared, fail-closed episode exclusions for decoded WebDataset samples.

Absolute entries use complete canonical shard paths. Relative entries are
explicitly relative to a configured dataset root; no basename guessing occurs.
All matching entries are merged, with whole-sample exclusions taking precedence.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_ALIASES = {
    "/data/user_data/robertre/Genesis/3DPCWM_Data":
        "/data/group_data/katefgroup/VLA/3DPCWM",
}
_LEGACY_MANIFESTS = {
    **{f"reports/phantom_blocklists/{name}.json": f"data/manifests/{name}.json"
       for name in ("robert_pickup_data_4success_combined",
                    "robert_pickup_data_place_success_strict_v2")},
    **{f"reports/unified/data_audit_0903/{name}.json": f"data/manifests/{name}.json"
       for name in ("mpm_blocklist_train", "mpm_blocklist_eval",
                    "cloth_blocklist_train", "cloth_blocklist_val")},
}


def manifest_path(path):
    """Resolve bundled manifests independently of cwd, including pinned old configs."""
    name = os.path.normpath(os.fspath(path))
    name = _LEGACY_MANIFESTS.get(name, name)
    if name.startswith("data/manifests/"):
        return REPO_ROOT / name
    return Path(name)


@lru_cache(maxsize=32768)
def canonical_shard(path):
    path = os.path.normpath(os.fspath(path))
    for old, new in DATA_ROOT_ALIASES.items():
        if path == old or path.startswith(old + "/"):
            path = new + path[len(old):]
            break
    return os.path.realpath(path) if os.path.isabs(path) else path


def load_blocklists(paths):
    """Load (shard, key) -> set(row indices) or None; missing files are fatal."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    result = {}
    for path in paths:
        with manifest_path(path).open() as stream:
            data = json.load(stream)
        entries = data["blocklist"] if isinstance(data, dict) else data
        for entry in entries:
            key = (canonical_shard(entry["shard"]), entry["key"])
            elem = entry.get("elem")
            if elem is None:
                result[key] = None
            else:
                if isinstance(elem, bool) or int(elem) != elem or elem < 0:
                    raise ValueError(f"Invalid blocklist row index {elem!r} in {path}")
                rows = result.setdefault(key, set())
                if rows is not None:
                    rows.add(int(elem))
    return result


def filter_blocklisted(samples, blocklist, roots):
    """Drop whole samples or selected rows, without mutating the input dictionary.

    Relative entries apply only below the exact root directory, not a similarly
    named prefix. All batched .pyd sidecars are filtered with the same indices.
    """
    if isinstance(roots, (str, os.PathLike)):
        roots = [roots]
    roots = tuple(canonical_shard(root).rstrip("/") + "/" for root in roots)
    for sample in samples:
        url = canonical_shard(sample.get("__url__", ""))
        key = sample.get("__key__", "")
        candidates = {url, *(url[len(root):] for root in roots if url.startswith(root))}
        matches = [blocklist[(shard, key)] for shard in candidates if (shard, key) in blocklist]
        if not matches:
            yield sample
            continue
        ref = sample.get("particle_pos.pyd")
        if any(rows is None for rows in matches) or ref is None:
            continue
        blocked = set().union(*matches)
        batch_size = int(ref.shape[0])
        if any(i >= batch_size for i in blocked):
            raise ValueError(f"Blocklist row outside batch of {batch_size}: {url}, {key}, {blocked}")
        keep = np.array([i for i in range(batch_size) if i not in blocked], dtype=np.int64)
        if not keep.size:
            continue
        if keep.size < batch_size:
            sample = {name: value[keep] if (
                name.endswith(".pyd") and hasattr(value, "ndim") and value.ndim > 0
                and value.shape[0] == batch_size) else value
                for name, value in sample.items()}
        yield sample
