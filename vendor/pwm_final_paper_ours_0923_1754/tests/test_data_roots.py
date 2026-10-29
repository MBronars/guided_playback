"""data.roots: explicit per-domain dataset locations for training and validation.

CPU only, no corpus: shard directories are temporary files whose names alone drive
split membership; no tar is ever decoded here.
"""
import json

import pytest
import torch

from data.unified_dataset import build_domain_dataset, resolve_data_roots


def _shards(root, n):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (root / f"{i:05d}.tar").write_bytes(b"\0" * 1024)
    return str(root)


def test_absent_roots_keep_the_registered_corpus():
    assert resolve_data_roots(None) == {}


def test_invalid_specs_fail_before_any_shard_is_opened(tmp_path):
    root = _shards(tmp_path / "push", 3)
    for spec, message in [
        ({"lava": {"train": root}}, "unsupported domain"),
        ({"pushing": {"train": root, "tarin": root}}, "unknown keys"),
        ({"pushing": {"val": root}}, "train is required"),
        ({"pushing": {"train": "relative/path"}}, "absolute"),
        ({"pushing": {"train": str(tmp_path / "missing")}}, "does not exist"),
        ({"pushing": {"train": root, "val": root}}, "overlaps"),
        ({"pushing": {"train": root, "val": str(tmp_path)}}, "overlaps"),
        ({"pushing": {"train": [root, root]}}, "twice"),
        ({"mpm": {"train": root}}, "both"),
        ({"pushing": {"train": root, "exclusions": {"test": []}}}, "exclusions"),
        ({"pushing": {"train": root, "exclusions": {"train": [str(tmp_path / "no.json")]}}}, "not found"),
        ("pushing", "mapping"),
    ]:
        with pytest.raises(ValueError, match=message):
            resolve_data_roots(spec)


def test_override_without_val_uses_the_deterministic_last_ten_percent(tmp_path):
    root = _shards(tmp_path / "push_v3", 10)
    roots = resolve_data_roots({"pushing": {"train": root}})
    train = build_domain_dataset("pushing", "train", raw=True, roots=roots)
    val = build_domain_dataset("pushing", "val", training=False, raw=True, roots=roots)
    assert len(train.shards) == 9 and len(val.shards) == 1
    assert val.shards[0].endswith("00009.tar")
    assert train.roots == [root] and train.roots_override is True
    assert train.blocklist is None and val.blocklist is None      # v1 lists are not carried over


def test_explicit_val_root_reads_both_splits_whole(tmp_path):
    train_root = _shards(tmp_path / "push_v3" / "train", 4)
    val_root = _shards(tmp_path / "push_v3" / "val", 2)
    roots = resolve_data_roots({"pushing": {"train": train_root, "val": val_root}})
    assert len(build_domain_dataset("pushing", "train", raw=True, roots=roots).shards) == 4
    val = build_domain_dataset("pushing", "val", training=False, raw=True, roots=roots)
    assert len(val.shards) == 2 and val.roots == [val_root] and val.source_split == "val"


def test_pickup_multi_root_override_splits_per_root(tmp_path):
    a = _shards(tmp_path / "pick_a", 10)
    b = _shards(tmp_path / "pick_b", 20)
    roots = resolve_data_roots({"pickup": {"train": [a, b]}})
    assert len(build_domain_dataset("pickup", "train", raw=True, roots=roots).shards) == 27
    assert len(build_domain_dataset("pickup", "val", training=False, raw=True, roots=roots).shards) == 3


def test_pickup_shift_is_an_explicit_root_option(tmp_path):
    # pwm-data-v1/pickup states its own canonical translation: the registered -0.10 m leaves the table top at +10 mm.
    from data.unified_dataset import CANONICAL_TRANSLATION
    train, val = _shards(tmp_path / "pickup" / "train", 4), _shards(tmp_path / "pickup" / "test", 2)
    roots = resolve_data_roots({"pickup": {"train": train, "val": val, "shift_m": [-0.45, 0.0, -0.11]}})
    assert roots["pickup"]["shift_m"] == [-0.45, 0.0, -0.11]
    assert resolve_data_roots(roots) == roots                                     # resolving twice changes nothing
    for split in ("train", "val"):
        ds = build_domain_dataset("pickup", split, training=split == "train", raw=True, roots=roots)
        assert torch.equal(ds._shift_t, torch.tensor([-0.45, 0.0, -0.11]))
    plain = resolve_data_roots({"pickup": {"train": train, "val": val}})
    assert "shift_m" not in plain["pickup"]
    registered = build_domain_dataset("pickup", "train", raw=True, roots=plain)
    assert torch.equal(registered._shift_t, torch.tensor(CANONICAL_TRANSLATION["pickup"]))
    for spec, message in [({"pushing": {"train": train, "shift_m": [0, 0, 0]}}, "only supported for pickup"),
                          ({"pickup": {"train": train, "val": val, "shift_m": [0, 0]}}, "three finite numbers"),
                          ({"pickup": {"train": train, "val": val, "shift_m": [0, 0, float("nan")]}}, "three finite numbers"),
                          ({"pickup": {"train": train, "val": val, "shift_m": "down"}}, "three finite numbers")]:
        with pytest.raises(ValueError, match=message):
            resolve_data_roots(spec)


def test_override_exclusions_are_the_only_ones_applied(tmp_path):
    root = _shards(tmp_path / "push_v3", 10)
    blocklist = tmp_path / "exclude.json"
    blocklist.write_text(json.dumps({"blocklist": [{"shard": f"{root}/00000.tar", "key": "k0"}]}))
    roots = resolve_data_roots({"pushing": {"train": root, "exclusions": {"train": [str(blocklist)]}}})
    train = build_domain_dataset("pushing", "train", raw=True, roots=roots)
    assert train.blocklist_paths == (str(blocklist),) and len(train.blocklist) == 1
    assert build_domain_dataset("pushing", "val", training=False, raw=True, roots=roots).blocklist is None


def test_mpm_override_uses_one_directory_per_split(tmp_path):
    train_root = _shards(tmp_path / "mpm_v3" / "train", 3)
    val_root = _shards(tmp_path / "mpm_v3" / "val", 1)
    roots = resolve_data_roots({"mpm": {"train": train_root, "val": val_root}})
    train = build_domain_dataset("mpm", "train", raw=True, roots=roots)
    val = build_domain_dataset("mpm", "val", training=False, raw=True, roots=roots)
    assert train.roots == [train_root] and len(train.shards) == 3
    assert val.roots == [val_root] and len(val.shards) == 1


def test_domains_not_named_are_untouched(tmp_path, monkeypatch):
    from data import unified_dataset
    calls = []
    monkeypatch.setattr(unified_dataset, "_build_override_dataset",
                        lambda domain, *a, **k: calls.append(domain) or object())
    root = _shards(tmp_path / "push_v3", 2)
    roots = resolve_data_roots({"pushing": {"train": root}})
    build_domain_dataset("pushing", "train", raw=True, roots=roots)
    with pytest.raises(ValueError, match="Unsupported unified domain"):
        build_domain_dataset("lava", "train", raw=True, roots=roots)
    assert calls == ["pushing"]


def test_loader_dispatch_forwards_roots_to_every_split(tmp_path, monkeypatch):
    from data import unified_dataset
    from data.dataset import create_dataloader

    class Empty(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())

    seen = []

    def fake(domain, split="train", **kwargs):
        seen.append((domain, split, kwargs.get("roots")))
        return Empty()

    monkeypatch.setattr(unified_dataset, "build_domain_dataset", fake)
    root = _shards(tmp_path / "push_v3", 10)
    spec = {"pushing": {"train": root}}
    create_dataloader("unified://v1", normalization="unified://xy_p99", format="unified",
                      training=False, num_workers=0, domains=["pushing"], roots=spec)
    stream = create_dataloader("unified://v1", normalization="unified://xy_p99", format="unified",
                               training=True, num_workers=0, domains=["pushing"], roots=spec)
    with pytest.raises(ValueError, match="Empty training stream"):
        next(iter(stream))
    assert [s for _, s, _ in seen] == ["val", "train"]
    assert all(r == resolve_data_roots(spec) for _, _, r in seen)


def test_config_validates_roots_before_model_allocation(tmp_path):
    from utils.config import load_config
    root = _shards(tmp_path / "push_v3", 10)
    cfg = load_config("configs/transformer.yaml",
                      [f"data.roots.pushing.train={root}", "data.domains=[pushing]"])
    assert cfg.data.roots.pushing.train == root
    with pytest.raises(ValueError, match="does not exist"):
        load_config("configs/transformer.yaml", [f"data.roots.pushing.train={tmp_path / 'missing'}"])
    with pytest.raises(ValueError, match="Unknown or retired data options"):
        load_config("configs/transformer.yaml", [f"data.rooots.pushing.train={root}"])
    with pytest.raises(ValueError, match="unsupported domain"):
        load_config("configs/transformer.yaml", [f"data.roots.lava.train={root}"])
