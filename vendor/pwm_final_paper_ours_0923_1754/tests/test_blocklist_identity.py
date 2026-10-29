"""Exclusions identify a complete shard, not a spelling or a basename."""
import json

import numpy as np
import pytest

from data.unified_dataset import _blocklist_filter, _load_blocklists


OLD = "/data/user_data/robertre/Genesis/3DPCWM_Data"
NEW = "/data/group_data/katefgroup/VLA/3DPCWM"


def sample(url):
    return {"__url__": url, "__key__": "batch_1",
            "particle_pos.pyd": np.arange(4 * 6).reshape(4, 2, 1, 3),
            "link_data.pyd": np.arange(4), "metadata.pyd": np.arange(7)}


def load(tmp_path, entries):
    path = tmp_path / "block.json"
    path.write_text(json.dumps({"blocklist": entries}))
    return _load_blocklists([str(path)])


def test_relocated_pickup_paths_exclude_the_same_rows(tmp_path):
    suffix = "/robert_pickup_data_place_success/00000.tar"
    blocked = load(tmp_path, [{"shard": OLD + suffix, "key": "batch_1", "elem": i}
                              for i in (0, 1, 3)])
    result = list(_blocklist_filter([sample(NEW + suffix)], blocked, [NEW]))
    assert len(result) == 1
    np.testing.assert_array_equal(result[0]["link_data.pyd"], [2])
    np.testing.assert_array_equal(result[0]["metadata.pyd"], np.arange(7))


def test_symlink_aliases_identify_the_same_shard(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    blocked = load(tmp_path, [{"shard": str(alias / "a.tar"), "key": "batch_1"}])
    assert list(_blocklist_filter([sample(str(actual / "a.tar"))], blocked, [str(actual)])) == []


@pytest.mark.parametrize("source,target", [
    (NEW + "/place/00000.tar", NEW + "/pick/00000.tar"),
    (OLD + "_other/place/00000.tar", NEW + "/place/00000.tar"),
    ("1obj/00000.tar", NEW + "/data_extra/1obj/00000.tar"),
])
def test_similar_names_do_not_collide(tmp_path, source, target):
    blocked = load(tmp_path, [{"shard": source, "key": "batch_1"}])
    result = list(_blocklist_filter([sample(target)], blocked, [NEW + "/data"]))
    assert result[0]["particle_pos.pyd"].shape[0] == 4


def test_all_matching_forms_merge_and_whole_sample_wins(tmp_path):
    suffix = "/place/00000.tar"
    blocked = load(tmp_path, [
        {"shard": OLD + suffix, "key": "batch_1", "elem": 0},
        {"shard": NEW + suffix, "key": "batch_1", "elem": 1},
        {"shard": "00000.tar", "key": "batch_1", "elem": 2},
    ])
    result = list(_blocklist_filter([sample(NEW + suffix)], blocked, [NEW + "/place"]))
    np.testing.assert_array_equal(result[0]["link_data.pyd"], [3])
    blocked[("00000.tar", "batch_1")] = None
    assert list(_blocklist_filter([sample(NEW + suffix)], blocked, [NEW + "/place"])) == []


def test_filter_does_not_mutate_input_rows(tmp_path):
    url = NEW + "/place/00000.tar"
    blocked = load(tmp_path, [{"shard": url, "key": "batch_1", "elem": 0}])
    original = sample(url)
    result = list(_blocklist_filter([original], blocked, [NEW]))
    assert result[0]["link_data.pyd"].shape == (3,)
    assert original["link_data.pyd"].shape == (4,)


def test_missing_exclusions_fail_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_blocklists([str(tmp_path / "missing.json")])


def test_bundled_legacy_manifest_resolves_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old = "./reports/phantom_blocklists/robert_pickup_data_4success_combined.json"
    current = "./data/manifests/robert_pickup_data_4success_combined.json"
    assert _load_blocklists(old) == _load_blocklists(current)


def test_blocklist_bypass_is_rejected_before_loading_data():
    from data.dataset import create_dataloader
    with pytest.raises(ValueError, match="Blocklists are required"):
        create_dataloader("unified://v1", normalization="unified://xy_p99", format="unified",
                          allow_missing_blocklists=True)


@pytest.mark.parametrize("elem", [-1, 0.5, True])
def test_invalid_row_indices_fail_closed(tmp_path, elem):
    with pytest.raises(ValueError, match="Invalid blocklist row"):
        load(tmp_path, [{"shard": "a.tar", "key": "batch_1", "elem": elem}])


def test_shipped_pickup_manifests_use_sample_keys_not_field_names():
    for name in ("robert_pickup_data_4success_combined", "robert_pickup_data_place_success_strict_v2"):
        blocked = _load_blocklists(f"data/manifests/{name}.json")
        assert all("." not in key for _, key in blocked)


def test_legacy_key_recovery_inverts_index_not_string_prefix():
    from scripts.dev.repair_pickup_manifest import recover_key
    names = [f"batch_{i:08d}.{field}.pyd" for i in range(4) for field in ("a", "b")]
    assert recover_key(names, "batch_00000001.b") == "batch_00000003"
