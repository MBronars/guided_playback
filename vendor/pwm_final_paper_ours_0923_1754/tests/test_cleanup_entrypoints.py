"""Retained unified/native/paper entrypoint contracts, not retired specialist CLIs."""
from copy import deepcopy
import json
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from data.normalization import resolve_normalization, domain_normalizers


def _cfg(context=2, future=4, profile="xy_p99_per_domain"):
    return OmegaConf.create(dict(data=dict(format="unified", context_len=context, future_len=future,
                                          domains=["mpm"], num_collate=1),
                                 normalization=resolve_normalization(profile), prediction_target="delta",
                                 diffusion_forcing=True, df_uncertainty_scale=0.7, df_schedule="pyramid",
                                 denoise_method="flow_matching", flow_matching=dict(inference_steps=4)))


@pytest.mark.parametrize("budget,message", [(["--clips", "0"], "--clips"), (["--clips", "-2"], "--clips"),
                                            (["--clips", "all", "--max_batches", "-1"], "retired")])
def test_native_mpm_invalid_budget_precedes_checkpoint(monkeypatch, tmp_path, budget, message):
    from scripts import eval_object_rollout as native
    monkeypatch.setattr(native, "load_model_and_diffusion", lambda *a, **k: pytest.fail("checkpoint reached"))
    monkeypatch.setattr(sys, "argv", ["native", "--checkpoint", "unused", "--wds", "unused",
                                     *budget, "--out", str(tmp_path / "out.json")])
    with pytest.raises(ValueError, match=message):
        native.main()


@pytest.mark.parametrize("key,value", [("context_len", 8), ("future_len", 99),
                                       ("normalization", "xy_p99_global")])
def test_native_mpm_rejects_external_representation(monkeypatch, tmp_path, key, value):
    from scripts import eval_object_rollout as native
    import utils.config
    cfg = _cfg()
    external = deepcopy(cfg)
    if key == "normalization":
        external.normalization = resolve_normalization(value)
    else:
        external.data[key] = value
    monkeypatch.setattr(utils.config, "load_config", lambda path: external)
    monkeypatch.setattr(native, "load_model_and_diffusion", lambda *a, **k: (None, None, cfg, 4))
    monkeypatch.setattr(native, "create_eval_loader", lambda *a, **k: pytest.fail("corpus reached"))
    monkeypatch.setattr(sys, "argv", ["native", "--checkpoint", "unused", "--config", "external.yaml",
                                     "--wds", "unused", "--clips", "all", "--out", str(tmp_path / "out.json")])
    with pytest.raises(ValueError, match=key):
        native.main()


@pytest.mark.parametrize("bad", ["empty", "nan", "zero_valid", None])
@pytest.mark.parametrize("rollout", ["window", "full", "stream"])
def test_native_mpm_metrics_fail_closed_and_weight_episodes(monkeypatch, tmp_path, bad, rollout):
    from scripts import eval_object_rollout as native
    from tests.test_train import create_dummy_batch
    cfg = _cfg()
    batches = [create_dummy_batch(B=b, T=6, N=8, context_len=2) for b in (2, 1)]
    for batch in batches:
        batch.positions.zero_()
        batch.action_mask.zero_()
        batch.object_ids.zero_()
        batch.delta_stats.min_val.fill_(-1)
        batch.delta_stats.max_val.fill_(1)
        if bad == "zero_valid":
            batch.valid_lengths.zero_()
    counter = iter([0.01, 0.03])
    def predictions(batch):
        pred = batch.future.clone()
        pred[..., 0] += next(counter)
        if pred.shape[0] == 2:
            pred[1, ..., 0] += 0.01
        if bad == "nan":
            pred.fill_(float("nan"))
        return pred
    def window(model, batch, diffusion, **kwargs):
        assert kwargs["uncertainty_scale"] == 0.7
        raw = torch.diff(torch.cat([batch.anchor[:, None], predictions(batch)], dim=1), dim=1)
        return batch.delta_stats.normalize(raw)
    def ar(model, diffusion, batch, ar_cfg):
        assert ar_cfg.df_uncertainty_scale == 0.7
        assert ar_cfg.streaming is (rollout == "stream")
        original = batches[0 if batch.positions.shape[0] == 2 else 1]
        return {"predicted_positions": torch.cat([batch.positions[:, :2], predictions(original)], dim=1)}
    monkeypatch.setattr(native, "load_model_and_diffusion", lambda *a, **k: (None, None, cfg, 4))
    monkeypatch.setattr(native, "create_eval_loader", lambda *a, **k: [] if bad == "empty" else batches)
    monkeypatch.setattr(native, "file_sha256", lambda *a: "fixture")
    monkeypatch.setattr(native, "run_diffusion_inference_df", window)
    def windowed(model, diffusion, batch, **kwargs):
        assert kwargs["uncertainty_scale"] == 0.7
        original = batches[0 if batch.positions.shape[0] == 2 else 1]
        return {"positions": torch.cat([batch.positions[:, :2], predictions(original)], dim=1)}
    monkeypatch.setattr(native, "autoregressive_rollout", ar)
    monkeypatch.setattr(native, "windowed_rollout", windowed)
    output = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["native", "--checkpoint", "unused", "--wds", "unused",
                                     "--rollout", rollout, "--clips", "all", "--device", "cpu",
                                     "--out", str(output)])
    if bad:
        with pytest.raises(ValueError, match="No episodes|Non-finite|valid particles"):
            native.main()
        assert not output.exists()
    else:
        native.main()
        result = json.loads(output.read_text())
        assert result["particle_mean_mm"] == pytest.approx(20, abs=0.0001)
        assert result["particle_final_mm"] == pytest.approx(20, abs=0.0001)
        assert result["n_episodes"] == 3 and result["clips_requested"] == "all"
        assert result["particle_per_frame_mm"] == pytest.approx([20]*4, abs=0.0001)
        assert result["per_count"] == {"1": {"n_episodes": 3, "particle_mean_mm": 20.0,
                                             "particle_final_mm": 20.0}}
        assert result["normalization"]["sha256"] == cfg.normalization.sha256
        assert result["shift_m"] == pytest.approx([-0.75, -0.75, -0.10])


@pytest.mark.parametrize("rollout", ["window", "full", "stream"])
def test_native_loader_uses_canonical_adapter_checkpoint_normalization_and_finite_pass(monkeypatch, rollout):
    from scripts import eval_object_rollout as native
    cfg = _cfg(profile="xy_p99_global")
    captured = {}
    class Dataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr(native, "UnifiedMPMDataset", Dataset)
    monkeypatch.setattr(torch.utils.data, "DataLoader", lambda ds, **k: (ds, k))
    ds, loader = native.create_eval_loader("heldout", cfg, rollout=rollout, num_workers=0)
    assert captured["normalization"] == cfg.normalization
    assert "stats_location" not in captured and captured["shuffle"] is False
    assert captured["future_len"] == (4 if rollout == "window" else -1)
    assert captured["context_len"] == 2 and ds.strict_read is True
    assert loader == dict(batch_size=None, num_workers=0)


def test_all_passive_mpm_populations_remain_explicit():
    from scripts.eval_object_rollout import MPM_POPULATIONS
    assert set(MPM_POPULATIONS) == {"v2_seen", "v2_unseen_full", "v2_unseen_validation",
        "v2_unseen_00001plus", "v2_unseen_clean", "v2_ood", "v2_ood_first_tar",
        "v2_viz_pinned", "paper_v1_id", "legacy_v1_test"}
    assert "not independent" in MPM_POPULATIONS["v2_unseen_clean"][1]
    assert "not full OOD" in MPM_POPULATIONS["v2_ood_first_tar"][1]
    assert "selection" in MPM_POPULATIONS["v2_seen"][1]


@pytest.mark.parametrize("context", [2, 4, 8])
@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
def test_paper_eval_defaults_follow_checkpoint(context, profile):
    from scripts.unified import eval_paper_mpm_unified as paper
    cfg = _cfg(context=context, future=16, profile=profile)
    C, F, normalization = paper.resolve_checkpoint_protocol(cfg)
    assert C == context and F == 16
    assert normalization == resolve_normalization(profile)
    with pytest.raises(ValueError, match="match the checkpoint"):
        paper.resolve_checkpoint_protocol(cfg, context_len=context + 1)
    cfg.data.format = "mpm_multi"
    with pytest.raises(ValueError, match="historical specialists"):
        paper.resolve_checkpoint_protocol(cfg)


@pytest.mark.parametrize("context", [2, 4, 8])
def test_paper_scoring_keeps_native_frames(context):
    from scripts.unified.eval_paper_mpm_unified import paper_group_metrics
    crop = 8 - context
    rows = [np.arange(crop, 49, dtype=float), np.arange(crop, 49, dtype=float) + 2]
    result = paper_group_metrics(rows, crop)
    assert result["train_h"] == 16.5
    assert result["long"] == 29.0
    assert result["final"] == 49.0


@pytest.mark.parametrize("rows", [[], [np.zeros(42)], [np.full(43, np.nan)]])
def test_paper_rejects_empty_truncated_or_nonfinite_groups(rows):
    from scripts.unified.eval_paper_mpm_unified import paper_group_metrics
    with pytest.raises(ValueError):
        paper_group_metrics(rows, crop=6)


def test_paper_cloth_uses_checkpoint_windows_and_no_global_dataset_mutation(monkeypatch, tmp_path):
    import inference
    import data.unified_dataset as data
    from scripts.unified import eval_paper_cloth_table3 as paper, eval_unified
    cfg = _cfg(context=4, future=16)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(inference, "load_model_and_diffusion", lambda *a, **k: (torch.nn.Identity(), None, cfg, 4))
    monkeypatch.setattr(paper, "materialize", lambda root, context_len: str(tmp_path / "sidecar.npz"))
    built = []
    def dataset(*a, **k):
        from types import SimpleNamespace
        assert k["context_len"] == 4 and k["future_len"] == 16
        assert k["normalization"]["sha256"] == cfg.normalization.sha256
        assert len(a) == 1
        built.append(SimpleNamespace())
        return built[-1]
    monkeypatch.setattr(data, "UnifiedClothDataset", dataset)
    def evaluate(*a, **k):
        assert k["dataset"] is built[0]
        assert k["context_len"] == 4 and k["future_len"] == 16
        assert k["horizon"] == 1001 and k["chamfer_marks"] == paper.MARKS
        return dict(n_episodes=2, **{f"chamfer_cm_at{m}": 1.0 for m in paper.MARKS})
    monkeypatch.setattr(eval_unified, "rollout_domain", evaluate)
    previous_root = data.CLOTH_WDS_ROOT
    monkeypatch.setattr(sys, "argv", ["paper_cloth", "--checkpoint", str(checkpoint),
                                      "--root", str(tmp_path / "cache"), "--out", str(tmp_path / "out.json")])
    paper.main()
    assert data.CLOTH_WDS_ROOT == previous_root


@pytest.mark.parametrize("context", [2, 4, 8])
def test_paper_cloth_materialization_alignment_and_cache_integrity(monkeypatch, tmp_path, context):
    import pickle
    import tarfile
    from scripts.unified import eval_paper_cloth_table3 as paper, materialize_cloth_wds as materializer
    monkeypatch.setattr(materializer, "N_CLOTH_MAX", 4)
    monkeypatch.setattr(materializer, "N_EE", 4)
    points = np.zeros((2070, 4, 3), np.float32)
    points[:, :, 0] = np.arange(2070)[:, None]
    episodes = []
    for key in ("pants", "shirt"):
        path = tmp_path / f"{key}.npz"
        np.savez(path, cloth_points=points, left_ee_points=points, right_ee_points=points,
                 ee_link_ids=np.arange(4), fps=np.float32(60))
        episodes.append((str(path), key))
    monkeypatch.setattr(paper, "EPISODES", episodes)
    root = tmp_path / "cache"
    sidecar = paper.materialize(str(root), context_len=context)
    shard = root / "val/shard-00000.tar"
    with tarfile.open(shard) as stream:
        stored = pickle.load(stream.extractfile("pants.particle_pos.pyd"))
    assert len(stored) == context + 1001
    assert stored[context, 0, 0] == np.float32(60) + materializer.SHIFT[0]
    for t in paper.PUBLISHED_OURS:
        assert stored[context + t, 0, 0] == np.float32(60 + 2*t) + materializer.SHIFT[0]
    assert paper.materialize(str(root), context_len=context) == sidecar
    with pytest.raises(ValueError, match="inputs changed"):
        paper.materialize(str(root), context_len=context + 1)
    before = shard.read_bytes()
    (root / "cloth_sidecar_table3.npz").write_bytes(b"corrupt cache")
    with pytest.raises(ValueError, match="corrupt"):
        paper.materialize(str(root), context_len=context)
    assert shard.read_bytes() == before


def test_h5_raw_reader_applies_exact_canonical_shift(tmp_path):
    import h5py
    from scripts.inference_mpm.inference_mpm_multi import load_h5_as_long_batch
    path = tmp_path / "passive.h5"
    x = np.arange(5*4*3, dtype=np.float32).reshape(5, 4, 3) * 0.001
    with h5py.File(path, "w") as file:
        file["x"], file["object_ids"] = x, np.array([0, 0, 1, 1])
        for key, values in dict(E=[1000, 2000], nu=[0.2, 0.3], density=[200, 300], mat_type=[0, 2]).items():
            file[key] = values
    delta, pos, _ = domain_normalizers("xy_p99_global", "mpm")
    batch = load_h5_as_long_batch(path, delta, pos)
    torch.testing.assert_close(batch.positions[0], torch.from_numpy(x) + torch.tensor([-0.75, -0.75, -0.10]), atol=0, rtol=0)
    assert batch.E.tolist() == [[1000, 1000, 2000, 2000]]
    assert batch.mat_type.tolist() == [[0, 0, 2, 2]]
    assert not batch.actions.any() and not batch.action_mask.any()



def test_h5_tool_parses_its_arguments_before_loading(monkeypatch, tmp_path):
    from scripts.inference_mpm import inference_mpm_multi as h5_tool

    class Loaded(Exception):
        pass

    def load(*args, **kwargs):
        raise Loaded

    monkeypatch.setattr(h5_tool, "load_model_and_diffusion", load)
    monkeypatch.setattr(sys, "argv", ["h5", "--checkpoint", "unused", "--h5_dir", str(tmp_path), "--device", "cpu"])
    with pytest.raises(Loaded):
        h5_tool.main()

def _mpm_tar(root):
    import io
    import pickle
    import tarfile
    root.mkdir(parents=True, exist_ok=True)
    positions = np.arange(2*49*5*3, dtype=np.float32).reshape(2, 49, 5, 3) * 0.00001
    valid = np.ones((2, 49, 5), dtype=bool)
    valid[:, :, -1] = False
    material = np.zeros((2, 5, 4), np.float32)
    material[..., 0], material[..., 1], material[..., 2] = 1000, 0.3, 400
    values = dict(particle_pos=positions, valid_particles=valid,
                   material_properties=material, object_ids=np.zeros((2, 5), np.int64))
    with tarfile.open(root / "shard-00000.tar", "w") as stream:
        for key, value in values.items():
            encoded = pickle.dumps(value)
            member = tarfile.TarInfo(f"0000.{key}.pyd")
            member.size = len(encoded)
            stream.addfile(member, io.BytesIO(encoded))
    return positions


@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
def test_native_real_wds_decode_shift_mask_and_profile_without_old_assets(tmp_path, profile):
    from scripts.eval_object_rollout import create_eval_loader
    from data.unified_dataset import _shift
    positions = _mpm_tar(tmp_path / "wds")
    cfg = _cfg(context=4, future=16, profile=profile)
    batches = list(create_eval_loader(str(tmp_path / "wds"), cfg, rollout="full", num_workers=0))
    assert len(batches) == 1
    batch = batches[0]
    torch.testing.assert_close(batch.positions, torch.from_numpy(positions) + _shift("mpm"), atol=0, rtol=0)
    assert batch.valid_lengths.tolist() == [4, 4]
    assert batch.context_len == 4 and batch.positions.shape[1] == 49
    delta, position, _ = domain_normalizers(profile, "mpm")
    assert torch.equal(batch.delta_stats.max_val, delta.max_val)
    assert torch.equal(batch.position_stats.max_val, position.max_val)


def test_paper_cli_canonical_shift_crop_padding_and_non_id_groups(monkeypatch, tmp_path):
    from scripts.unified import eval_paper_mpm_unified as paper
    from data.unified_dataset import _shift
    raw = _mpm_tar(tmp_path / "paper" / "1obj")
    _mpm_tar(tmp_path / "paper" / "5obj")
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    cfg = _cfg(context=4, future=16, profile="xy_p99_global")
    monkeypatch.setattr(paper, "load_model_and_diffusion", lambda *a: (torch.nn.Identity(), None, cfg, 4))
    calls = []
    def rollout(*, sequence_data, config, **kwargs):
        calls.append(True)
        assert config.context_len == 4 and config.future_len == 16
        assert sequence_data.valid_lengths.tolist() == [4]
        expected = torch.from_numpy(raw[:1, 4:]) + _shift("mpm")
        torch.testing.assert_close(sequence_data.positions, expected, atol=0, rtol=0)
        pred = sequence_data.positions.clone()
        pred[:, 4:, :, 0] += 0.01
        pred[:, :, 4] = float("nan")  # Invalid padding is never part of the score.
        return dict(predicted_positions=pred, gt_positions=sequence_data.positions)
    monkeypatch.setattr(paper, "autoregressive_rollout", rollout)
    out = tmp_path / "paper.json"
    monkeypatch.setattr(sys, "argv", ["paper", "--checkpoint", str(checkpoint), "--id_wds_root",
        str(tmp_path / "paper"), "--obj_groups", "1", "5", "--max_files", "1", "--device", "cpu", "--out", str(out)])
    paper.main()
    result = json.loads(out.read_text())
    assert len(calls) == 2
    assert result["n_episodes"] == 2 and result["crop"] == 4
    assert "ID_train_h" not in result
    assert result["selected_groups_train_h"] == pytest.approx(1, abs=0.001)
    assert result["normalization"]["sha256"] == cfg.normalization.sha256
    assert result["per_group"]["1"]["episode_ids"] == ["shard-00000_0000_b0"]


@pytest.mark.parametrize("per_frame", [False, True])
@pytest.mark.parametrize("name", ["_gather_knn", "_gather_knn_positions", "_gather_knn_local", "_gather_knn_positions_local"])
def test_transformer_knn_gathers_match_indexing_and_gradients(per_frame, name):
    from models.transformer import TrajectoryTransformer
    torch.manual_seed(901)
    B, T, N, D, L, K = 2, 3, 9, 3, 4, 2
    x = torch.randn(B, T, N, D, requires_grad=True)
    indices = torch.randint(N, (B, T, L, K) if per_frame else (B, L, K))
    expected = torch.stack([torch.stack([x[b, t, indices[b, t] if per_frame else indices[b]]
                                       for t in range(T)]) for b in range(B)])
    observed = getattr(TrajectoryTransformer, name)(None, x, indices)
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)
    expected_grad, = torch.autograd.grad(expected.square().sum(), x, retain_graph=True)
    observed_grad, = torch.autograd.grad(observed.square().sum(), x)
    torch.testing.assert_close(observed_grad, expected_grad, atol=1e-6, rtol=1e-6)
