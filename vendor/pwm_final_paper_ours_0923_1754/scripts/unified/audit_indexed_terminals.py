"""Audit the final C4/F16 window of every episode in frozen index directories.

This checks first-termination boundaries and geometry across an entire indexed
population. It is not a scan of every state and does not fit normalization.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path

import torch

from data.indexed_episodes import IndexedEpisodeDataset, load_index
from scripts.unified.verify_collections import audit_batch
from utils.reproducibility import working_source_versions


@lru_cache(maxsize=3)
def dataset(root, domain):
    torch.set_num_threads(1)
    ds = IndexedEpisodeDataset(root, domain=domain, raw=True, training=False)
    ds.articulated = True
    return ds


def check(task):
    root, domain, row = task
    ds = dataset(root, domain)
    record = ds.records[row]
    result = dict(episode_id=record["episode_id"], split=ds.source_split,
                  teacher=record.get("teacher"))
    try:
        result["audit"] = audit_batch(ds.window(record, record["frames"] - 20))
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--journal", help="Append per-episode results for recovery after interruption")
    parser.add_argument("--resume", action="store_true", help="Resume an explicitly named journal")
    args = parser.parse_args()
    tasks, indices = [], {}
    for path in sorted(Path(args.root).glob("*/INDEX.json")):
        index = load_index(path.parent, args.domain)
        indices[str(path)] = index["index_sha256"]
        tasks += [(str(path.parent), args.domain, i) for i in range(len(index["records"]))]
    if not tasks:
        raise ValueError("No frozen index directories found")
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    sources = working_source_versions()
    header = dict(schema="pwm-terminal-audit-journal-v1", domain=args.domain,
                  indices=indices, source_versions=sources)
    rows, finished, journal = [], set(), None
    if args.resume and not args.journal:
        raise ValueError("--resume requires --journal")
    if args.journal:
        path = Path(args.journal)
        if args.resume:
            with path.open("rb") as stream:
                if json.loads(stream.readline()) != header:
                    raise ValueError("Journal indices or audit source differ; start a new audit")
                good_end = stream.tell()
                for line in stream:
                    if not line.endswith(b"\n"):
                        break  # Only an incomplete final write may be discarded.
                    item = json.loads(line)
                    task = tuple(item["task"])
                    if task in finished:
                        raise ValueError("Duplicate journal task")
                    finished.add(task); rows.append(item["result"])
                    good_end += len(line)
            if not finished <= set(tasks):
                raise ValueError("Journal contains tasks outside the frozen indices")
            with path.open("r+b") as stream:
                stream.truncate(good_end)
            journal = path.open("a")
        else:
            journal = path.open("x")
            journal.write(json.dumps(header) + "\n"); journal.flush()
    pending = [task for task in tasks if task not in finished]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for task, row in zip(pending, pool.map(check, pending, chunksize=16)):
            if journal:
                journal.write(json.dumps(dict(task=task, result=row), allow_nan=False) + "\n")
                journal.flush()
            rows.append(row)
            if len(rows) % 500 == 0:
                try:
                    print(f"{args.domain}: {len(rows)}/{len(tasks)} terminal windows", flush=True)
                except OSError:
                    pass  # A progress-log failure must not discard audited rows.
    if journal:
        journal.close()
    if sources != working_source_versions():
        raise RuntimeError("Source changed during audit; repeat on a stable checkout")
    failed = [r for r in rows if "error" in r]
    successful = [r["audit"] for r in rows if "audit" in r]
    result = dict(domain=args.domain, indices=indices, episodes=len(rows), failed=len(failed),
        source_versions=sources, journal=args.journal,
        coverage="one final C4/F16 window per indexed episode, including validation/test",
        by_teacher=dict(Counter(r["teacher"] for r in rows)),
        maxima={k: max(a[k] for a in successful) for k in successful[0] if k != "shape"} if successful else {},
        failures=failed)
    with Path(args.out).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
