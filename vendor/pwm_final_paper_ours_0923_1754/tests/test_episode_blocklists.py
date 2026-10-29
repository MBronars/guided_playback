"""The mpm episode blocklist (2026-09-03 corpus audit) drops single episodes out of the raw
4-episode samples: every per-episode array shrinks consistently, whole-sample entries drop
the sample, and unlisted samples pass through untouched."""
import json
import numpy as np

from data.unified_dataset import _blocklist_filter, _load_blocklists, MPM_BLOCKLISTS


def _sample(url, key, B=4, T=5, N=6):
    return {"__url__": url, "__key__": key,
            "particle_pos.pyd": np.random.rand(B, T, N, 3).astype(np.float32),
            "valid_particles.pyd": np.ones((B, T, N), dtype=bool),
            "object_ids.pyd": np.zeros((B, N), dtype=np.int32),
            "n_objects.pyd": np.ones(B, dtype=np.int32),
            "material_properties.pyd": np.random.rand(B, N, 5).astype(np.float32),
            "drag_active.pyd": np.zeros((B, T), dtype=bool)}


def test_row_filter_drops_only_the_listed_episode(tmp_path):
    root = "/fake/root"
    bl = tmp_path / "bl.json"
    bl.write_text(json.dumps({"blocklist": [{"shard": f"{root}/1obj/00000.tar", "key": "batch_1", "elem": 2},
                                            {"shard": "1obj/00000.tar", "key": "batch_1", "elem": 2},   # root-relative form
                                            {"shard": f"{root}/1obj/00001.tar", "key": "batch_0"}]}))  # whole sample
    blocked = _load_blocklists([str(bl)])
    s_hit = _sample(f"{root}/1obj/00000.tar", "batch_1")
    s_pass = _sample(f"{root}/1obj/00000.tar", "batch_2")
    s_drop = _sample(f"{root}/1obj/00001.tar", "batch_0")
    orig = s_hit["particle_pos.pyd"].copy()
    out = list(_blocklist_filter(iter([s_hit, s_pass, s_drop]), blocked, [root]))
    assert [o["__key__"] for o in out] == ["batch_1", "batch_2"]
    hit = out[0]
    for k, v in hit.items():
        if k.endswith(".pyd"):
            assert v.shape[0] == 3, k
    assert np.array_equal(hit["particle_pos.pyd"], orig[[0, 1, 3]])
    assert out[1]["particle_pos.pyd"].shape[0] == 4


def test_relative_shard_form_matches_when_url_differs_only_by_root():
    bl = {("1obj/00000.tar", "batch_1"): {0}}
    s = _sample("/other/prefix/1obj/00000.tar", "batch_1")
    out = list(_blocklist_filter(iter([s]), bl, ["/other/prefix"]))
    assert out[0]["particle_pos.pyd"].shape[0] == 3


def test_shipped_mpm_blocklist_parses():
    blocked = _load_blocklists(MPM_BLOCKLISTS)
    eps = sum(len(v) for v in blocked.values() if v)
    assert eps == 128, eps            # 64 episodes x (absolute + root-relative forms)
    assert all(v is not None for v in blocked.values())   # episode-level only, never whole samples


def test_shipped_cloth_blocklists_parse():
    from data.unified_dataset import CLOTH_BLOCKLISTS
    for split, expected in (("train", 418), ("val", 57)):
        blocked = _load_blocklists(CLOTH_BLOCKLISTS[split])
        assert all(v is None for v in blocked.values())          # whole-sample drops (one episode per sample)
        assert len(blocked) == 2 * expected, (split, len(blocked))  # absolute + root-relative forms
