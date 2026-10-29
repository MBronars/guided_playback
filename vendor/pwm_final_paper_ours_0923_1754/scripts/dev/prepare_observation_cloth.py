"""Pin 16 distinct validation cloth episodes at deterministic middle windows.

The earlier geometry fixtures use near-static prefixes. This diagnostic keeps
four spaced validation shards and takes their first four currently accepted
episodes, with a window at half the available start range (no error-based pick).
"""
import argparse
import json
import pickle
import tarfile
from pathlib import Path

import torch

from data.dataset import SkipSample
from data.unified_dataset import UnifiedClothDataset, CLOTH_BLOCKLISTS
from scripts.dev.observation_probe_common import write_json
from utils.checkpoint import read_checkpoint, checkpoint_config
from utils.reproducibility import file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = checkpoint_config(read_checkpoint(a.checkpoint))
    ds = UnifiedClothDataset(["/data/group_data/katefgroup/VLA/3DPCWM/unified_v1/cloth/val"],
                             split="all", training=False, shuffle=False, context_len=4,
                             future_len=16, normalization=cfg.normalization,
                             blocklist_paths=CLOTH_BLOCKLISTS["val"])
    ds.emit_sidecar = True
    ds.articulated = bool(cfg.model.get("articulated_bodies", False))
    old = json.loads(Path("validation/architecture_review_2026-09-12/deep_dive/geometry_results.json").read_text())
    shards = [x["receipt"]["shard"] for x in old["windows"] if x["receipt"]["name"] in
              {"cloth_0", "cloth_1", "cloth_2", "cloth_3"}]
    entries, skipped = [], []
    for shard_index, shard in enumerate(shards):
        accepted = 0
        with tarfile.open(shard) as tar:
            members = tar.getmembers()
            keys = list(dict.fromkeys(m.name.split('.', 1)[0] for m in members))
            for key in keys:
                if accepted >= 4:
                    break
                sample = {"__key__": key, "__url__": shard}
                for m in members:
                    if m.name.startswith(key + '.') and m.name.endswith('.pyd'):
                        sample[m.name[len(key) + 1:]] = pickle.load(tar.extractfile(m))
                if "particle_pos.pyd" not in sample:
                    continue
                from data.blocklists import filter_blocklisted
                sample = next(filter_blocklisted([sample], ds.blocklist, ds.roots), None)
                if sample is None:
                    skipped.append({"shard": shard, "key": key, "reason": "blocklist"})
                    continue
                T = sample["particle_pos.pyd"].shape[0]
                if T < 20:
                    skipped.append({"shard": shard, "key": key, "reason": "short"}); continue
                start = (T - 20) // 2
                sample["particle_pos.pyd"] = sample["particle_pos.pyd"][start:start + 20]
                try:
                    batch = ds._process_sample(sample)
                except SkipSample as error:
                    skipped.append({"shard": shard, "key": key, "reason": str(error)}); continue
                name = f"cloth_middle_{shard_index}_{accepted}"
                path = out / (name + '.pt'); torch.save(batch, path)
                entries.append({"key": name, "domain": "cloth", "fixture": str(path.resolve()),
                                "fixture_sha256": file_sha256(path), "fixture_row": 0,
                                "particles": batch.num_particles,
                                "source": {"shard": shard, "key": key, "source_rows": [0],
                                           "starts": [start], "total_frames": T,
                                           "dt_seconds": float(sample["dt.pyd"]),
                                           "exclusions_sha256": {p: file_sha256(p) for p in CLOTH_BLOCKLISTS["val"]}}})
                print(name, key, start, T, batch.num_particles, flush=True)
                accepted += 1
    assert len(entries) == 16, f"Expected 16 distinct episodes, got {len(entries)}"
    write_json(out / 'manifest.json', {"clips": entries, "skipped": skipped,
                                      "selection": "first four accepted episodes per pinned shard; middle window"})


if __name__ == '__main__':
    main()
