#!/usr/bin/env python3
"""Collect raw calibration once, fit both profiles, and audit fixed bounds without refitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.normalization import NORMALIZATION_DOMAINS, load_normalization_artifact
from data.stats_fitting import (audit_caches, collect_domain_cache, collection_coverage, fit_artifact,
                               json_digest, load_collection_index, write_json_new)


def _coverage_text(coverage):
    shards = coverage["shards"]
    return (f"{coverage['records']} records / {coverage['episode_windows']} episode windows; "
            f"recorded shards {shards['visited_recorded']}/{shards['available']}; "
            f"{coverage['skipped_records_recorded']} recorded skips (ledger coverage, not corpus completeness)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect", help="deterministic bounded raw sampling; never fits stats")
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--domains", nargs="+", choices=NORMALIZATION_DOMAINS, default=list(NORMALIZATION_DOMAINS))
    collect.add_argument("--window-policies", nargs="+", choices=("training", "prefix", "uniform"),
                         default=["training", "prefix"])
    collect.add_argument("--split", choices=("train", "val"), default="train",
                         help="val is for audit only and cannot be fitted")
    collect.add_argument("--context-len", type=int, required=True)
    collect.add_argument("--future-len", type=int, required=True)
    collect.add_argument("--seed", type=int, default=0)
    collect.add_argument("--max-records", type=int, default=32,
                         help="finite records per domain/policy; one record may contain multiple episodes")
    collect.add_argument("--reservoir-rows", type=int, default=100000,
                         help="hard row cap per domain/policy/representation, plus bounded sampler working space")
    fit = sub.add_parser("fit", help="derive per-domain20 and max-over-domain global4 from train caches")
    fit.add_argument("--collection", type=Path, required=True)
    fit.add_argument("--out", type=Path, required=True)
    fit.add_argument("--version", required=True, help="new candidate version, never a frozen-* label")
    audit = sub.add_parser("audit", help="physical distributions, clipping and single-transition reconstruction")
    audit.add_argument("--collection", type=Path, required=True)
    audit.add_argument("--artifact", type=Path, help="default: bundled frozen normalization")
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "collect":
        if len(set(args.domains)) != len(args.domains) or len(set(args.window_policies)) != len(args.window_policies):
            parser.error("duplicate domains/window policies are not allowed")
        if args.context_len < 1 or args.future_len < 1 or args.seed < 0 or args.max_records < 1 or args.reservoir_rows < 1:
            parser.error("C/F, record and reservoir bounds must be positive; seed must be nonnegative")
        cache_rows = []
        for domain in args.domains:
            for policy in args.window_policies:
                path, metadata, reused = collect_domain_cache(
                    domain, output_dir=args.output_dir / "cache", context_len=args.context_len,
                    future_len=args.future_len, seed=args.seed, max_records=args.max_records,
                    capacity=args.reservoir_rows, window_policy=policy, split=args.split)
                cache_rows.append(dict(path=str(path.relative_to(args.output_dir)), sha256=metadata["sha256"]))
                print(f"{domain}/{policy}: {'verified cache' if reused else 'collected'} {metadata['retained']}; "
                      f"{_coverage_text(collection_coverage(metadata))}", flush=True)
        index = dict(schema_version=1, caches=cache_rows)
        index["sha256"] = json_digest(index)
        path = args.output_dir / f"collection-{index['sha256']}.json"
        if path.exists():
            if json.loads(path.read_text()) != index:
                raise ValueError("Existing collection index differs; refusing overwrite")
        else:
            write_json_new(path, index)
        print(path)
    elif args.command == "fit":
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite normalization artifact: {args.out}")
        artifact = fit_artifact(load_collection_index(args.collection), artifact_version=args.version)
        write_json_new(args.out, artifact)
        load_normalization_artifact(args.out)
        for row in artifact["provenance"]["caches"]:
            print(f"{row['domain']}/{row['window_policy']}: {_coverage_text(row['coverage'])}")
        print(args.out)
    else:
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite audit: {args.out}")
        report = audit_caches(load_collection_index(args.collection), args.artifact)
        write_json_new(args.out, report)
        for row in report["reports"]:
            print(f"{row['domain']}/{row['window_policy']}: {_coverage_text(row['coverage'])}")
        print(args.out)


if __name__ == "__main__":
    main()
