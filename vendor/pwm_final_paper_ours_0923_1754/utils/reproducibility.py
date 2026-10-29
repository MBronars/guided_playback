"""Small, explicit provenance helpers for evaluation and engineering runs."""
import hashlib
import math
from pathlib import Path
import re
import subprocess
import zlib


def config_for_json(cfg):
    """Copy a config for metadata: encode +/-inf as 'infinity'/'-infinity'.

    NaN remains an error. This is a JSON metadata representation, not a live
    training config; numeric checkpoint configs and caller-owned values are not
    changed. Metrics must still use strict JSON serialization without this helper.
    """
    from omegaconf import OmegaConf

    def encode(value):
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, float):
            if math.isnan(value):
                raise ValueError("NaN is not valid configuration metadata")
            if math.isinf(value):
                return "infinity" if value > 0 else "-infinity"
        return value

    return encode(OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg)


def seed_for_device(seed, device):
    """Seed CPU and only the selected CUDA generator, never other devices.

    Unlike torch.manual_seed this is safe inside a CPU-only or rank-local RNG
    fork: generators outside that fork are neither initialized nor modified.
    """
    import torch
    device = torch.device(device)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def evaluation_seed(base_seed, batch_index, label):
    """crc32-v1: stable uint32 seed, independent of Python's randomized hash()."""
    return (int(base_seed) + 31 * int(batch_index) + zlib.crc32(str(label).encode("utf-8"))) % 2**32


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def working_source_versions():
    """Hash executed repository source for engineering runs with uncommitted edits.

    A receipt alone does not preserve these bytes; collection verification also
    saves a source bundle alongside its results. Dataset files have separate
    identities, and the resolved configuration owns its normalization snapshot.
    """
    root = Path(__file__).resolve().parents[1]
    files = {root / name for name in ("train.py", "inference.py")}
    for folder in ("data", "models", "utils", "diffusion", "guidance", "collection", "scripts/unified"):
        files.update((root / folder).rglob("*.py"))
    files.update((root / "data/embodiments").glob("*.npz"))
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(files)}


def code_revision():
    """Record provenance, not a claim that a dirty checkout is reproducible."""
    root = Path(__file__).resolve().parents[1]
    marker = root / 'SOURCE_REVISION'
    if marker.exists():
        # Archives nested inside a live checkout otherwise inherit that parent
        # checkout's HEAD, even though they execute different frozen files.
        revision = marker.read_text().strip()
        if re.fullmatch(r'[0-9a-f]{40}', revision) is None:
            raise ValueError('SOURCE_REVISION must declare a full lowercase Git commit hash')
        return {'git_commit': revision, 'git_dirty': None,
                'source_snapshot_revision': revision,
                'source_snapshot_integrity': 'not checked by code_revision; verify archive contents separately'}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(status)}
