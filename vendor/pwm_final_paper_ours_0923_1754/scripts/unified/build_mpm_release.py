"""Freeze published MPM campaign chunks into one training release.

`data.roots.mpm` reads one train and one val directory, while a collection campaign publishes
reviewed chunks into several folders (the original run, salvaged chunks, continuations) and
keeps adding more. A release is a fixed selection built from those folders:

    <out>/train/<chunk>/<shard>.tar     hard links to the published shards (no copy)
    <out>/val/<chunk>/<shard>.tar       only when validation folders are given
    <out>/RELEASE.json                  every chunk and shard with its sha256

Every shard must match the sha256 in its chunk's MANIFEST.json, and a chunk may appear only
once across all folders and splits. Nothing is written until every shard verifies; the release
is assembled under <out>.partial and renamed when complete. Campaign folders are never
modified. Hard links need the release on the same filesystem as the campaign.

Usage:
  PYTHONPATH=. python scripts/unified/build_mpm_release.py --out RELEASE_DIR \\
      --train CAMPAIGN/data/collection/datasets/train [--train MORE ...] [--val VAL_FOLDER ...]
"""
import argparse
import errno
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from utils.reproducibility import code_revision


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def published_chunks(folder):
    """(chunk directory, MANIFEST.json contents) for every chunk one campaign split folder published."""
    folder = Path(folder).resolve()
    chunks = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
    if not chunks:
        raise ValueError(f"{folder} has no published chunks")
    listed = []
    for chunk in chunks:
        if not (chunk / "MANIFEST.json").is_file():
            raise ValueError(f"{chunk} has no MANIFEST.json: not a published chunk")
        receipt = json.loads((chunk / "MANIFEST.json").read_text())
        present = {p.name for p in chunk.glob("*.tar")}
        if unlisted := sorted(present - set(receipt["shards"])):
            raise ValueError(f"{chunk}: shards not listed in MANIFEST.json: {unlisted}")
        if absent := sorted(set(receipt["shards"]) - present):
            raise ValueError(f"{chunk}: shards missing from the chunk: {absent}")
        listed.append((chunk, receipt))
    return listed


def build_release(out, splits, jobs=8):
    """Verify, link and list the chunks of {"train": [folders], "val": [folders]}; returns RELEASE.json."""
    out = Path(out).absolute()
    partial = out.with_name(out.name + ".partial")
    for path in (out, partial):
        if path.exists():
            raise FileExistsError(f"{path} already exists; a release is never rebuilt in place")
    if unknown := set(splits) - {"train", "val"}:
        raise ValueError(f"splits must be train and val, got {sorted(unknown)}")
    if not splits.get("train"):
        raise ValueError("a release needs at least one train folder")

    chunks, seen = {"train": [], "val": []}, {}
    for split in chunks:
        for folder in splits.get(split) or ():
            for chunk, receipt in published_chunks(folder):
                if chunk.name in seen:
                    raise ValueError(f"chunk {chunk.name} appears more than once: {seen[chunk.name]} and {chunk}")
                seen[chunk.name] = chunk
                chunks[split].append((chunk, receipt))
    shards = [(chunk / name, digest) for listed in chunks.values() for chunk, receipt in listed
              for name, digest in receipt["shards"].items()]
    with ThreadPoolExecutor(jobs) as pool:
        found = list(pool.map(lambda shard: _sha256(shard[0]), shards))
    if changed := [str(path) for (path, digest), actual in zip(shards, found) if actual != digest]:
        raise ValueError(f"{len(changed)} shards do not match their MANIFEST.json sha256: {changed[:5]}")

    try:
        for split, listed in chunks.items():
            for chunk, receipt in listed:
                target = partial / split / chunk.name
                target.mkdir(parents=True)
                for name in receipt["shards"]:
                    os.link(chunk / name, target / name)
                (target / "MANIFEST.json").write_bytes((chunk / "MANIFEST.json").read_bytes())
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise ValueError(f"{out} must be on the campaign's filesystem: shards are hard-linked") from error
        raise

    def summary(split):
        if not chunks[split]:
            return None
        chunk_list = [dict(name=chunk.name, source=str(chunk), retained=receipt["retained"],
                           chunk_complete_sha256=receipt.get("chunk_complete_sha256"), shards=receipt["shards"],
                           bytes=sum((chunk / name).stat().st_size for name in receipt["shards"]))
                      for chunk, receipt in chunks[split]]
        return dict(sources=[str(Path(folder).resolve()) for folder in splits[split]], chunks=len(chunk_list),
                    shards=sum(len(c["shards"]) for c in chunk_list), retained=sum(c["retained"] for c in chunk_list),
                    bytes=sum(c["bytes"] for c in chunk_list), chunk_list=chunk_list)

    release = dict(format="mpm-release-v1", created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   code=code_revision(), verification="every shard matched its chunk MANIFEST.json sha256",
                   splits={split: summary(split) for split in chunks})
    (partial / "RELEASE.json").write_text(json.dumps(release, indent=2) + "\n")
    os.rename(partial, out)
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="new release directory")
    parser.add_argument("--train", type=Path, action="append", required=True, help="published train folder")
    parser.add_argument("--val", type=Path, action="append", default=[], help="published validation folder")
    parser.add_argument("--jobs", type=int, default=8, help="parallel shard hashes")
    args = parser.parse_args()
    release = build_release(args.out, {"train": args.train, "val": args.val}, jobs=args.jobs)
    for split, summary in release["splits"].items():
        print(f"{split}: " + ("none" if summary is None else
                              f"{summary['chunks']} chunks, {summary['shards']} shards, "
                              f"{summary['retained']} clips, {summary['bytes'] / 1e9:.1f} GB"))
    print(f"release: {args.out}")


if __name__ == "__main__":
    main()
