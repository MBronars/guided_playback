"""Preserve exact executed sources/inputs alongside immutable diagnostic receipts."""
import hashlib
import io
import json
from pathlib import Path
import os
import tarfile
import xml.etree.ElementTree as ET


def robot_assets(assets_root):
    """Pin the declared URDF and all referenced visual/collision mesh files."""
    root = Path(assets_root).resolve(strict=True)
    urdf = root / 'acone/acone.urdf'
    paths = {urdf.resolve(strict=True)}
    for mesh in ET.parse(urdf).iter('mesh'):
        filename = mesh.attrib['filename']
        if '://' in filename or Path(filename).is_absolute():
            raise ValueError('External robot mesh URI needs an explicit asset mapping')
        path = (urdf.parent / filename).resolve(strict=True)
        if not path.is_relative_to(root): raise ValueError('Robot mesh escapes explicit asset root')
        paths.add(path)
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def snapshot_sources(expected, output, *, recovered=None, max_bytes=256*1024**2):
    """Write a verified source archive, never a claim based only on Git HEAD.

    Optional recovered bytes must match the ORIGINAL recorded digest. They are
    only for preserving already completed build-only evidence after local edits.
    Ordinary execution never supplies this recovery argument.
    """
    recovered = recovered or {}
    members = {}
    total = 0
    for name, sha in sorted(expected.items()):
        path = Path(name)
        value = recovered.get(name)
        if value is None:
            if path.stat().st_size + total > max_bytes:
                raise ValueError('Explicit source archive budget exceeded')
            value = path.read_bytes()
        if hashlib.sha256(value).hexdigest() != sha:
            raise ValueError('Source no longer matches recorded execution: ' + name)
        total += len(value)
        if total > max_bytes: raise ValueError('Explicit source archive budget exceeded')
        member = 'sources/' + path.as_posix().lstrip('/')
        if '..' in Path(member).parts or member in members:
            raise ValueError('Ambiguous source archive member')
        members[member] = value
    members['SOURCE_MAP.json'] = json.dumps(expected, sort_keys=True).encode()
    with Path(output).open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            for name, value in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
        stream.flush(); os.fsync(stream.fileno())
    with tarfile.open(output, 'r:gz') as archive:
        if set(archive.getnames()) != set(members): raise ValueError('Archive member mismatch')
        for name, value in members.items():
            if archive.extractfile(name).read() != value: raise ValueError('Archive payload mismatch')
    return dict(files=len(expected), uncompressed_bytes=total,
                sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest())
