"""Small no-clobber publication and relocation helpers for candidate collection."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_once(path, value):
    """Publish complete JSON atomically; never expose partially written claims."""
    payload = (canonical(value) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)  # Only this call's unpublished temporary link.


@contextmanager
def claim(path, value):
    """Exclusive retained claim. Crashed attempts require a new explicit attempt ID."""
    path = Path(path)
    write_once(path, value)
    yield
    # Claims intentionally remain as immutable ownership/provenance records.


def relative_path(root, name):
    """Reject traversal, aliases and symlinks before resolving a manifest member."""
    root = Path(root).resolve()
    name = Path(name)
    if name.is_absolute() or not name.parts or any(p in ("..", ".") for p in name.parts):
        raise ValueError("Manifest member must be a nonempty relative path")
    path = root
    for part in name.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Manifest member is a symlink")
    if not path.is_relative_to(root):
        raise ValueError("Manifest member escapes root")
    return path


def check_hashes(root, hashes):
    for name, expected in hashes.items():
        if digest(relative_path(root, name)) != expected:
            raise ValueError("Artifact changed: " + name)


def rebase_exclusions(blocklist, source_root, destination_root):
    """Explicit canonical prefix only; preserve relative and whole-batch keys."""
    if blocklist is None:
        return None
    old = str(Path(source_root).resolve()).rstrip("/") + "/"
    new = str(Path(destination_root).resolve()).rstrip("/") + "/"
    result = {}
    for (shard, key), rows in blocklist.items():
        target = new + shard[len(old):] if shard.startswith(old) else shard
        pair = target, key
        if pair not in result:
            result[pair] = None if rows is None else set(rows)
        elif rows is None or result[pair] is None:
            result[pair] = None
        else:
            result[pair].update(rows)
    return result


def resolve_release(catalog, release_id, locations):
    """A release identifies data; an explicit location supplies its physical root."""
    release = catalog[release_id]
    if release.get("release_id") != release_id or not release.get("domain"):
        raise ValueError("Release identity/domain missing")
    root = Path(locations[release_id]).resolve()
    canonical_root = Path(release["canonical_root"]).resolve()
    shards = [relative_path(root, name) for name in release["shards"]]
    if not shards or len(shards) != len(set(shards)):
        raise ValueError("Nonempty unique shards required")
    if any(not p.is_file() for p in shards):
        raise ValueError("Missing release shard")
    return {"release_id": release_id, "domain": release["domain"], "root": root,
            "canonical_root": canonical_root, "shards": shards}
