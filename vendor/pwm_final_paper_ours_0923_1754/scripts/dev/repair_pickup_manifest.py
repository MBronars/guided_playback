"""Recover the 64 malformed place-success exclusion keys with measured evidence.

The historical audit labeled row i with sorted *field* name i, while its data
reader loaded sorted *sample* i. Merely stripping the field suffix would exclude
the WRONG trajectory. Invert the field-name index, then require both the recorded
object path length and maximum frame displacement to match the recovered row.

Writes proposals and a receipt ONLY under --output-dir. It never edits the
bundled manifests or corpus. Historical generator: reference 54be241,
scripts/build_4success_combined_blocklist.py::existing_blocklist_to_entries.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import tarfile

import numpy as np

from data.blocklists import canonical_shard, manifest_path


def recover_key(names, mislabeled_key):
    legacy = sorted({name.rsplit(".", 1)[0] for name in names if name.endswith(".pyd")})
    actual = sorted({name.split(".", 1)[0] for name in names if name.endswith(".pyd")})
    index = legacy.index(mislabeled_key)
    if index >= len(actual):
        raise ValueError(f"Historical index {index} exceeds {len(actual)} real samples")
    return actual[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-ref", default="7a1cd6d",
                        help="Git revision containing the frozen pre-repair manifests")
    args = parser.parse_args()
    place_path = manifest_path("data/manifests/robert_pickup_data_place_success_strict_v2.json")
    combined_path = manifest_path("data/manifests/robert_pickup_data_4success_combined.json")
    def frozen_bytes(path):
        return subprocess.check_output(["git", "show", f"{args.source_ref}:data/manifests/{path.name}"])
    old_place, old_combined = frozen_bytes(place_path), frozen_bytes(combined_path)
    place = json.loads(old_place)
    combined = json.loads(old_combined)
    grouped = defaultdict(list)
    for entry in place["blocklist"]:
        grouped[entry["shard"]].append(entry)
    evidence = []
    recovered = {}
    root = "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success"
    for shard, entries in grouped.items():
        with tarfile.open(f"{root}/{shard}") as stream:
            names = stream.getnames()
            for entry in entries:
                old = entry["key"]
                if "." not in old:
                    raise ValueError("Expected the frozen pre-repair manifest; refusing to reinterpret a repaired key")
                key = recover_key(names, old)
                payload = stream.extractfile(key + ".particle_pos.pyd").read()
                positions = pickle.loads(payload)
                ids = pickle.load(stream.extractfile(key + ".obj_ids.pyd"))
                elem = entry["elem"]
                centroid = positions[elem][:, ids[elem] == 0].mean(axis=1)
                step_mm = np.linalg.norm(np.diff(centroid, axis=0), axis=-1) * 1000
                measured = {"obj_path_mm": float(step_mm.sum()), "max_obj_dt_mm": float(step_mm.max())}
                errors = {name: abs(value - entry[name]) for name, value in measured.items()}
                if max(errors.values()) > 0.02:
                    raise ValueError(f"Recovered trajectory does not match recorded evidence: {shard}, {old}, {errors}")
                evidence.append({"shard": shard, "old_key": old, "key": key, "elem": elem,
                                 "measured": measured, "absolute_error_mm": errors,
                                 "particle_payload_sha256": hashlib.sha256(payload).hexdigest()})
                recovered[(canonical_shard(f"{root}/{shard}"), old, elem)] = key
                entry["key"] = key
        print(f"Verified {shard}: {len(entries)} rows", flush=True)
    changed = 0
    for entry in combined["blocklist"]:
        identity = (canonical_shard(entry["shard"]), entry["key"], entry["elem"])
        if identity in recovered:
            entry["key"] = recovered[identity]
            changed += 1
    assert changed == len(evidence) == 64
    receipt = {"passed": True, "verified_rows": len(evidence),
               "source_ref": args.source_ref,
               "old_place_sha256": hashlib.sha256(old_place).hexdigest(),
               "old_combined_sha256": hashlib.sha256(old_combined).hexdigest(),
               "max_error_mm": max(max(row["absolute_error_mm"].values()) for row in evidence),
               "evidence": evidence}
    provenance = {"date": "2026-09-05", "method": "inverse sorted field index -> sorted sample index",
                  "verified_rows": 64, "max_recomputed_metric_error_mm": receipt["max_error_mm"],
                  "script": "scripts/dev/repair_pickup_manifest.py",
                  "note": "Recorded motion statistics verified against actual recovered rows; not suffix stripping."}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path, contents in ((place_path, place), (combined_path, combined)):
        contents["key_repair"] = provenance
        (args.output_dir / path.name).write_text(json.dumps(contents, indent=2) + "\n")
    (args.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified all {len(evidence)} rows; maximum metric error {receipt['max_error_mm']:.6f} mm", flush=True)


if __name__ == "__main__":
    main()
