"""Audit pickup manifest identity and actual tar-header membership (no training).

This checks every referenced shard/key against the corpus without decoding the
large trajectory arrays. Row filtering is separately regression-tested. It does
not infer model exposure or retrospectively repair existing evaluation scores.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tarfile

from data.blocklists import canonical_shard, load_blocklists, manifest_path
from data.unified_dataset import PICKUP_BLOCKLISTS, PICKUP_ROOTS, _split_shards


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    blocked = load_blocklists(PICKUP_BLOCKLISTS)
    manifest_info = []
    literal = set()
    for path in PICKUP_BLOCKLISTS:
        contents = manifest_path(path).read_bytes()
        data = json.loads(contents)
        entries = data["blocklist"] if isinstance(data, dict) else data
        literal.update((e["shard"], e["key"]) for e in entries)
        manifest_info.append({"path": str(manifest_path(path)),
                              "sha256": hashlib.sha256(contents).hexdigest(),
                              "entries": len(entries)})
    split_shards = {}
    for split in ("train", "val"):
        split_shards[split] = {canonical_shard(str(p)) for root in PICKUP_ROOTS
                               for p in _split_shards(sorted(Path(root).rglob("*.tar")), split)}
    by_shard = defaultdict(set)
    for shard, key in blocked:
        by_shard[shard].add(key)
    counts = {s: {"shards": len(shards), "blocked_keys": 0, "blocked_rows": 0}
              for s, shards in split_shards.items()}
    problems = []
    for index, (shard, expected) in enumerate(sorted(by_shard.items())):
        try:
            with tarfile.open(shard) as stream:
                actual = {m.name.removesuffix(".particle_pos.pyd") for m in stream
                          if m.name.endswith(".particle_pos.pyd")}
            missing = expected - actual
            if missing:
                problems.append({"shard": shard, "missing_keys": sorted(missing)})
        except (OSError, tarfile.TarError) as error:
            problems.append({"shard": shard, "error": str(error)})
        for split, shards in split_shards.items():
            if shard in shards:
                counts[split]["blocked_keys"] += len(expected)
                counts[split]["blocked_rows"] += sum(
                    len(blocked[(shard, key)]) if blocked[(shard, key)] is not None else 4
                    for key in expected)
        if index % 100 == 0:
            print(f"Checked {index + 1}/{len(by_shard)} shards", flush=True)
    info = {"manifests": manifest_info, "literal_unique_keys": len(literal),
            "canonical_unique_keys": len(blocked), "referenced_shards": len(by_shard),
            "literal_matches_current_paths": sum((s, k) in literal for s, k in blocked),
            "splits_combined_exclusions": counts, "problems": problems,
            "passed": not problems,
            "notes": ["Row counts assume the documented four-row pickup packaging.",
                      "Combined phantom+holdout exclusions are shown for both shard splits; "
                      "production val applies only the holdout manifest."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
