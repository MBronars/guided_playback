"""Verify the complete file set of an rl_push frozen release before training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath

SPLIT_FORMAT, FOLDER_FORMAT = "pwm-data-v1-rl_push-split", "pwm-data-v1-rl_push"


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate release manifest key: {key}")
        obj[key] = value
    return obj


def _split_members(root, files, split, digest):
    """{path inside one split folder of pwm-data-v1/rl_push: metadata}.

    A split manifest lists its object files as `<split>/objects/...`. The dataset folder's manifest
    one level up lists what a split's own manifest cannot: that manifest and the split's INDEX.json.
    The two manifests must agree on every file both of them list."""
    prefix = root.name + "/"
    if split != root.name or not all(name.startswith(prefix) for name in files):
        raise ValueError(f"Split release manifest does not describe its folder: {root}")
    folder = root.parent / "RELEASE.json"
    if not folder.is_file():
        raise ValueError(f"Split release needs the dataset folder's manifest: {folder}")
    value = json.loads(folder.read_bytes(), object_pairs_hook=_unique)
    listed = value.get("files") if value.get("format") == FOLDER_FORMAT else None
    own = listed.get(prefix + "RELEASE.json") if isinstance(listed, dict) else None
    if not isinstance(own, dict) or own.get("sha256") != digest:
        raise ValueError(f"Split release manifest differs from the dataset folder's record: {folder}")
    members = {}
    for name, meta in listed.items():
        if not name.startswith(prefix) or name == prefix + "RELEASE.json":
            continue
        mine = files.get(name, meta)
        if not isinstance(meta, dict) or not isinstance(mine, dict) or any(
                mine.get(key) != meta.get(key) for key in ("bytes", "sha256")):
            raise ValueError(f"Split and dataset-folder release manifests disagree on {name}")
        members[name[len(prefix):]] = mine
    if set(files) - set(listed):
        raise ValueError(f"Split release lists files its dataset folder's manifest does not: "
                         f"{sorted(set(files) - set(listed))[:5]}")
    return members


def verify_rl_push_release(root, expected_sha256=None):
    """Check manifest identity, exact membership, sizes and hashes; return its digest.

    `root` is a frozen release (kind rl_push_frozen_release) or one split folder of
    pwm-data-v1/rl_push (format pwm-data-v1-rl_push-split, checked with the dataset folder's
    manifest one level up). This checks a snapshot at startup; it cannot prevent later
    replacement in writable directories. Training checkpoints bind the returned manifest
    digest for continuation.
    """
    root = Path(root).resolve()
    manifest = root / "RELEASE.json"
    contents = manifest.read_bytes()
    digest = hashlib.sha256(contents).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Frozen release manifest sha256 differs: {manifest}")
    value = json.loads(contents, object_pairs_hook=_unique)
    files = value.get("files")
    frozen, split = value.get("kind") == "rl_push_frozen_release", value.get("format") == SPLIT_FORMAT
    if not (frozen or split) or not isinstance(files, dict) or not files:
        raise ValueError("Expected a nonempty rl_push_frozen_release manifest "
                         f"or a {SPLIT_FORMAT} manifest of a release folder")
    if not frozen:
        files = _split_members(root, files, value.get("split"), digest)
    for name, meta in files.items():
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or str(p) != name or name == "RELEASE.json":
            raise ValueError(f"Invalid release member: {name}")
        if (not isinstance(meta, dict) or not isinstance(meta.get("bytes"), int)
                or isinstance(meta["bytes"], bool) or meta["bytes"] < 0
                or not isinstance(meta.get("sha256"), str) or len(meta["sha256"]) != 64):
            raise ValueError(f"Invalid release metadata: {name}")
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink():
            raise ValueError(f"Frozen release contains a symlink: {p}")
        if p.is_file() and p != manifest:
            actual.add(p.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError(f"Frozen release file set differs: missing={sorted(set(files)-actual)[:5]}, "
                         f"extra={sorted(actual-set(files))[:5]}")
    def check(name):
        path, meta = root / name, files[name]
        return name if path.stat().st_size != meta["bytes"] or _sha256(path) != meta["sha256"] else None
    with ThreadPoolExecutor(max_workers=8) as pool:
        bad = [name for name in pool.map(check, files) if name is not None]
    if bad:
        raise ValueError(f"Frozen release contents differ from RELEASE.json: {bad[:5]}")
    if manifest.read_bytes() != contents:
        raise ValueError("Frozen release manifest changed during verification")
    return digest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--sha256", required=True, help="expected RELEASE.json digest")
    args = parser.parse_args()
    print(verify_rl_push_release(args.root, args.sha256))
