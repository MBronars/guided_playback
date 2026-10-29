"""Per-record first_start in an indexed-episode INDEX.json (data/indexed_episodes.py). Standalone runner:

    python tests/test_indexed_first_start.py

A record without first_start keeps the domain's FIRST_START (yikai: 10) -- the control; a record carrying
first_start 0 draws windows from frame 0; windows below a record's first_start and malformed values are
refused. Uses real yikai_small test segments; skipped when the dataset is not on this machine.
"""
import json, sys, tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.indexed_episodes import IndexedEpisodeDataset

Y = Path("/data/group_data/katefgroup/runs_pwm_mbronars/datasets/yikai_small")
NORM = {"profile": "xy_p99_per_domain", "artifact": str(Y / "normalization_shared_three.json")}
pytestmark = pytest.mark.skipif(not (Y / "test" / "INDEX.json").is_file(), reason="yikai_small not on this machine")


def _dataset(records, **kw):
    index = json.loads((Y / "test" / "INDEX.json").read_text())
    index["records"] = records
    root = Path(tempfile.mkdtemp())
    (root / "INDEX.json").write_text(json.dumps(index))
    return IndexedEpisodeDataset(str(root), domain="yikai", context_len=4, future_len=16,
                                 training=True, shuffle=True, normalization=NORM, **kw)


def _records(n=8):
    return [r for r in json.loads((Y / "test" / "INDEX.json").read_text())["records"] if r["frames"] >= 40][:n]


def _min_start(ds, draws=40):
    torch.manual_seed(0)
    return min(s for r in ds.records for s in ds._draw_starts(r, draws))


def test_without_first_start_the_domain_default_holds():
    ds = _dataset(_records())
    assert ds.first_start == 10 and _min_start(ds) == 10
    with pytest.raises(ValueError, match="Window outside"):
        ds.window(ds.records[0], 9)
    print("ok  control: no per-record first_start -> yikai windows start at frame 10")


def test_record_first_start_overrides_it():
    recs = [dict(r, first_start=0) for r in _records()]
    ds = _dataset(recs)
    assert _min_start(ds) == 0
    assert ds.window(ds.records[0], 0).positions.shape[1] == 20
    mixed = _dataset([dict(r, first_start=(3 if i % 2 else 0)) for i, r in enumerate(_records())])
    torch.manual_seed(0)
    assert all(s >= r["first_start"] for r in mixed.records for s in mixed._draw_starts(r, 40))
    with pytest.raises(ValueError, match="Window outside"):
        mixed.window(mixed.records[1], 2)
    print("ok  per-record first_start: windows from frame 0, never below a record's own start")


@pytest.mark.parametrize("bad", [-1, 1.5, "0", True])
def test_malformed_first_start_is_refused(bad):
    recs = _records(2)
    recs[0] = dict(recs[0], first_start=bad)
    with pytest.raises(ValueError, match="first_start must be a nonnegative integer"):
        _dataset(recs)


if __name__ == "__main__":
    test_without_first_start_the_domain_default_holds()
    test_record_first_start_overrides_it()
    for bad in (-1, 1.5, "0", True):
        test_malformed_first_start_is_refused(bad)
    print("ALL OK")
