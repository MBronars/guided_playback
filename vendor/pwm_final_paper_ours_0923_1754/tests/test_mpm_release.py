"""scripts/unified/build_mpm_release.py: freeze published MPM campaign chunks into one release.

CPU only, no corpus: chunks are temporary directories of small byte files named like shards.
"""
import hashlib
import json
import os

import pytest

from data.unified_dataset import resolve_data_roots
from scripts.unified.build_mpm_release import build_release


def _chunk(folder, name, shards=2, retained=5):
    """A published campaign chunk: shards plus the MANIFEST.json listing their sha256."""
    chunk = folder / name
    chunk.mkdir(parents=True)
    hashes = {}
    for i in range(shards):
        data = f"{name}-{i}".encode() * 100
        (chunk / f"{i:05d}.tar").write_bytes(data)
        hashes[f"{i:05d}.tar"] = hashlib.sha256(data).hexdigest()
    (chunk / "MANIFEST.json").write_text(json.dumps(
        {"accepted_for_training": False, "chunk_complete_sha256": "c" * 64, "retained": retained,
         "shards": hashes}))
    return chunk


def test_release_hard_links_every_verified_shard_and_lists_it(tmp_path):
    original, continuation = tmp_path / "run/datasets/train", tmp_path / "recovery/datasets/train"
    _chunk(original, "train-84000")
    _chunk(original, "train-84001", shards=3, retained=7)
    _chunk(continuation, "train-84412")
    val = tmp_path / "val_run/datasets/val"
    _chunk(val, "val-90000", retained=4)
    out = tmp_path / "release"

    receipt = build_release(out, {"train": [original, continuation], "val": [val]})

    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.tar")) == [
        "train/train-84000/00000.tar", "train/train-84000/00001.tar", "train/train-84001/00000.tar",
        "train/train-84001/00001.tar", "train/train-84001/00002.tar", "train/train-84412/00000.tar",
        "train/train-84412/00001.tar", "val/val-90000/00000.tar", "val/val-90000/00001.tar"]
    shard = original / "train-84001/00002.tar"
    assert (out / "train/train-84001/00002.tar").stat().st_ino == shard.stat().st_ino   # linked, not copied
    assert json.loads((out / "RELEASE.json").read_text()) == receipt
    train = receipt["splits"]["train"]
    assert (train["chunks"], train["shards"], train["retained"]) == (3, 7, 17)
    assert train["chunk_list"][1]["shards"]["00002.tar"] == hashlib.sha256(shard.read_bytes()).hexdigest()
    assert receipt["splits"]["val"]["retained"] == 4
    assert not (tmp_path / "release.partial").exists()
    roots = resolve_data_roots({"mpm": {"train": str(out / "train"), "val": str(out / "val")}})
    assert roots["mpm"]["train"] == [os.path.realpath(out / "train")]
    assert roots["mpm"]["val"] == [os.path.realpath(out / "val")]


def test_a_train_only_snapshot_records_that_it_has_no_validation_split(tmp_path):
    source = tmp_path / "run/datasets/train"
    _chunk(source, "train-84000")

    receipt = build_release(tmp_path / "snapshot", {"train": [source]})

    assert receipt["splits"]["val"] is None
    assert not (tmp_path / "snapshot/val").exists()


def test_release_refuses_unverifiable_or_ambiguous_chunks_before_writing(tmp_path):
    source = tmp_path / "run/datasets/train"
    _chunk(source, "train-84000")

    changed = tmp_path / "changed/datasets/train"
    (_chunk(changed, "train-84001") / "00000.tar").write_bytes(b"edited after publication")
    unlisted = tmp_path / "unlisted/datasets/train"
    (_chunk(unlisted, "train-84002") / "00009.tar").write_bytes(b"not in the manifest")
    missing = tmp_path / "missing/datasets/train"
    (_chunk(missing, "train-84003") / "00001.tar").unlink()
    unpublished = tmp_path / "unpublished/datasets/train"
    (_chunk(unpublished, "train-84004") / "MANIFEST.json").unlink()
    duplicate = tmp_path / "duplicate/datasets/train"
    _chunk(duplicate, "train-84000")
    empty = tmp_path / "empty/datasets/train"
    empty.mkdir(parents=True)
    releases = tmp_path / "releases"
    releases.mkdir()

    for name, splits, message in [
            ("changed", {"train": [source, changed]}, "sha256"),
            ("unlisted", {"train": [source, unlisted]}, "not listed"),
            ("missing", {"train": [source, missing]}, "missing"),
            ("unpublished", {"train": [source, unpublished]}, "MANIFEST.json"),
            ("twice", {"train": [source, duplicate]}, "more than once"),
            ("leak", {"train": [source], "val": [duplicate]}, "more than once"),
            ("empty", {"train": [empty]}, "no published chunks")]:
        with pytest.raises(ValueError, match=message):
            build_release(releases / name, splits)
    assert not any(releases.iterdir())                  # nothing written, not even a .partial

    (releases / "taken").mkdir()
    with pytest.raises(FileExistsError):
        build_release(releases / "taken", {"train": [source]})
