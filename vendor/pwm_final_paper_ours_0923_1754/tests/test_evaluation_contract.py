"""No empty/non-finite benchmark may look like a valid zero-error result."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from data.normalization import resolve_normalization
import inference
from scripts.unified import eval_unified as evaluation
from tests.test_train import create_dummy_batch


class Dataset:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)



def patch_sampler(monkeypatch, sampler):
    """Fixed windows sample through eval_unified, rollouts through inference.windowed_rollout."""
    monkeypatch.setattr(evaluation, "run_diffusion_inference_df", sampler)
    monkeypatch.setattr(inference, "run_diffusion_inference_df", sampler)

def setup(monkeypatch, batches):
    monkeypatch.setattr(evaluation, "build_domain_dataset", lambda *a, **k: Dataset(batches))
    cfg = OmegaConf.create({"normalization": resolve_normalization(),
                            "data": {"context_len": 2, "future_len": 4},
                            "flow_matching": {"inference_steps": 4}})
    return SimpleNamespace(cfg=SimpleNamespace()), None, cfg, "mpm"


@pytest.mark.parametrize("rollout", [False, True])
def test_empty_evaluation_raises(monkeypatch, rollout):
    args = setup(monkeypatch, [])
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    with pytest.raises(ValueError, match="No episodes"):
        fn(*args, max_clips=1, device=torch.device("cpu"),
           **({"horizon": 4} if rollout else {}))


@pytest.mark.parametrize("budget", [0, -1])
@pytest.mark.parametrize("rollout", [False, True])
def test_invalid_budget_fails_before_corpus_access(monkeypatch, budget, rollout):
    args = setup(monkeypatch, [])
    monkeypatch.setattr(evaluation, "build_domain_dataset",
                        lambda *a, **k: pytest.fail("must validate before corpus I/O"))
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    with pytest.raises(ValueError, match="budget"):
        fn(*args, max_clips=budget, device=torch.device("cpu"),
           **({"horizon": 4} if rollout else {}))


@pytest.mark.parametrize("rollout", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_prediction_raises(monkeypatch, rollout, bad):
    batch = create_dummy_batch(B=1, T=6, N=8, context_len=2)
    args = setup(monkeypatch, [batch])
    patch_sampler(monkeypatch,
                        lambda model, b, *a, **k: torch.full_like(b.prediction_x0, bad))
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    with pytest.raises(ValueError, match="Non-finite"):
        fn(*args, max_clips=1, device=torch.device("cpu"), future_len=4,
           **({"horizon": 4} if rollout else {}))


def test_zero_valid_particles_are_rejected(monkeypatch):
    batch = create_dummy_batch(B=1, T=6, N=8, context_len=2)
    batch.valid_lengths.zero_()
    args = setup(monkeypatch, [batch])
    patch_sampler(monkeypatch,
                        lambda *a, **k: torch.zeros_like(batch.prediction_x0))
    with pytest.raises(ValueError, match="valid particles"):
        evaluation.eval_domain(*args, max_clips=1, device=torch.device("cpu"))


@pytest.mark.parametrize("rollout", [False, True])
def test_exact_target_preserves_valid_metrics(monkeypatch, rollout):
    batch = create_dummy_batch(B=2, T=10, N=8, context_len=2)
    args = setup(monkeypatch, [batch])
    patch_sampler(monkeypatch,
                        lambda model, b, *a, **k: b.prediction_stats.normalize(b.prediction_x0))
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    result = fn(*args, max_clips=None, device=torch.device("cpu"), future_len=4,
                **({"horizon": 7} if rollout else {}))
    assert result["n_episodes"] == 2
    assert result["particle_mean_mm"] < 0.001
    assert result["particle_final_mm"] < 0.001


@pytest.mark.parametrize("horizon", [1, 7])
def test_rollout_dump_preserves_small_motion_in_float32(monkeypatch, tmp_path, horizon):
    batch = create_dummy_batch(B=1, T=10, N=8, context_len=2)
    batch.positions.fill_(1.0)
    batch.positions += torch.arange(10)[None, :, None, None] * 0.00002
    args = setup(monkeypatch, [batch])
    patch_sampler(monkeypatch,
                        lambda model, b, *a, **k: b.prediction_stats.normalize(b.prediction_x0))
    evaluation.rollout_domain(*args, max_clips=1, device=torch.device("cpu"),
                              future_len=4, horizon=horizon, dump_dir=str(tmp_path))
    with np.load(tmp_path / "mpm.npz") as data:
        for field in ("pred", "gt", "context"):
            assert data[field].dtype == np.float32
        np.testing.assert_array_equal(data["gt"], batch.future[0, :horizon].numpy())
        np.testing.assert_allclose(data["pred"], data["gt"], rtol=0, atol=2e-7)
        assert (data["gt"][0] != data["context"][-1]).all()


@pytest.mark.parametrize("rollout", [False, True])
def test_metric_masks_follow_each_batch_element(monkeypatch, rollout):
    from tests.test_unet_std_pose_native import make_batch
    batch = make_batch()
    n = batch.num_particles
    permutation = torch.cat([torch.arange(48, n), torch.arange(48)])
    for name in ("positions", "actions"):
        getattr(batch, name)[1] = getattr(batch, name)[1, :, permutation].clone()
    for name in ("action_mask", "object_ids", "particle_body", "E", "nu", "rho", "mat_type"):
        getattr(batch, name)[1] = getattr(batch, name)[1, permutation].clone()
    args = setup(monkeypatch, [batch])
    args[0].cfg.pose_native = True
    monkeypatch.setattr(Dataset, "_pose_stats", lambda self: None, raising=False)
    def oracle(model, b, *a, **k):
        pred_pos = b.future.clone()
        deform = b.body_type[1, b.particle_body[1]] == 0
        pred_pos[1, :, deform, 0] += 0.001
        delta = torch.diff(torch.cat([b.anchor[:, None], pred_pos], dim=1), dim=1)
        return (b.delta_stats.normalize(delta),
                b.pose_delta_stats.normalize(b.pose_deltas_gt[:, :, b.body_type[0] == 1]))
    patch_sampler(monkeypatch, oracle)
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    result = fn(*args, max_clips=None, device=torch.device("cpu"), context_len=3, future_len=4,
                **({"horizon": 4} if rollout else {}))
    assert result["deform_mean_mm"] == pytest.approx(0.5, abs=0.001)
    assert result["pose_passive_mm"] < 0.002
    assert result["pose_actuated_mm"] < 0.002


@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
@pytest.mark.parametrize("explicit_domains", [False, True])
def test_cli_records_checkpoint_windows_snapshot_and_selected_domain(monkeypatch, tmp_path, profile, explicit_domains):
    from utils.reproducibility import file_sha256
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock model loaded below")
    out = tmp_path / "result.json"
    cfg = OmegaConf.create({"normalization": resolve_normalization(profile),
        "data": {"context_len": 4, "future_len": 16, "domains": ["mpm"]},
        "flow_matching": {"inference_steps": 8}})
    monkeypatch.setattr(evaluation, "load_model_and_diffusion", lambda *a: (torch.nn.Identity(), None, cfg, 8))
    registered = SimpleNamespace(roots=["registered-mpm-val"], blocklist_paths=())
    monkeypatch.setattr(evaluation, "build_domain_dataset", lambda *a, **k: registered)
    calls = []
    def evaluate(*a, **k):
        assert a[3] == "mpm" and a[4] is None      # --clips all: the whole split
        calls.append(k)
        return {"n_episodes": 1, "particle_mean_mm": 0.0}
    monkeypatch.setattr(evaluation, "eval_domain", evaluate)
    monkeypatch.setattr(sys, "argv", ["eval_unified.py", "--checkpoint", str(checkpoint), "--clips", "all",
                                      "--out", str(out)]
                        + (["--domains", "mpm"] if explicit_domains else []))
    evaluation.main()
    result = json.loads(out.read_text())
    assert result["context_len"] == calls[0]["context_len"] == 4
    assert result["future_len"] == calls[0]["future_len"] == 16
    assert result["inference_steps"] == 8
    assert result["checkpoint_sha256"] == file_sha256(checkpoint)
    assert calls[0]["dataset"] is registered
    assert result["inputs"]["mpm"]["roots"] == ["registered-mpm-val"]
    assert result["inputs"]["mpm"]["roots_source"] == "registered"
    assert result["inputs"]["mpm"]["normalization_sha256"] == cfg.normalization.sha256
    assert result["normalization"] == resolve_normalization(profile)
    assert result["domains"] == ["mpm"]
    assert result["clips_requested"] == "all"
    assert result["explicit_cross_domain_selection"] is explicit_domains


@pytest.mark.parametrize("rollout", [False, True])
def test_explicit_dataset_avoids_global_corpus_selection_and_validates_normalization(monkeypatch, rollout):
    batch = create_dummy_batch(B=1, T=6, N=8, context_len=2)
    args = setup(monkeypatch, [])
    ds = Dataset([batch])
    ds.context_len = 2
    ds.normalization = resolve_normalization()
    monkeypatch.setattr(evaluation, "build_domain_dataset", lambda *a, **k: pytest.fail("unexpected corpus access"))
    patch_sampler(monkeypatch,
                        lambda m, b, *a, **k: b.prediction_stats.normalize(b.prediction_x0))
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    kwargs = dict(max_clips=1, device=torch.device("cpu"), context_len=2, future_len=4, dataset=ds)
    if rollout:
        kwargs["horizon"] = 4
    res = fn(*args, **kwargs)
    assert res["n_episodes"] == 1 and res["particle_mean_mm"] < 0.001
    ds.context_len = 4
    with pytest.raises(ValueError, match="context_len"):
        fn(*args, **kwargs)
    ds.context_len, ds.normalization = 2, resolve_normalization("xy_p99_global")
    with pytest.raises(ValueError, match="normalization differs"):
        fn(*args, **kwargs)


def test_explicit_root_construction_does_not_mutate_globals(monkeypatch):
    import data.unified_dataset as data
    cfg = OmegaConf.create({"normalization": resolve_normalization()})
    seen = []
    def build(**kwargs):
        seen.append(kwargs)
        return object()
    monkeypatch.setattr(data, "UnifiedMPMDataset", build)
    before = data.MPM_VAL_ROOT
    evaluation.evaluation_dataset(cfg, "mpm", 4, 16, {"mpm": ["custom-root"]})
    assert data.MPM_VAL_ROOT == before
    assert seen[0]["wds_location"] == "custom-root"
    assert seen[0]["normalization"] == cfg.normalization


def _release_config(tmp_path):
    """A checkpoint config whose MPM data came from its own release (data.roots), with a
    validation exclusion list, instead of the registered unified://v1 corpus."""
    from tests.test_data_roots import _shards
    train, val = _shards(tmp_path / "release" / "train", 3), _shards(tmp_path / "release" / "val", 1)
    exclusions = tmp_path / "release" / "val_exclusions.json"
    exclusions.write_text(json.dumps({"blocklist": [{"shard": f"{val}/00000.tar", "key": "k0"}]}))
    cfg = OmegaConf.create({"normalization": resolve_normalization(),
                            "data": {"context_len": 2, "future_len": 4, "domains": ["mpm"],
                                     "roots": {"mpm": {"train": train, "val": val,
                                                       "exclusions": {"val": [str(exclusions)]}}}},
                            "flow_matching": {"inference_steps": 4}})
    return cfg, val, str(exclusions)


@pytest.mark.parametrize("rollout", [False, True])
def test_evaluators_score_the_release_a_checkpoint_saved(monkeypatch, tmp_path, rollout):
    from data.unified_dataset import resolve_data_roots
    cfg, val, exclusions = _release_config(tmp_path)
    dataset = evaluation.evaluation_dataset(cfg, "mpm", 2, 4)
    assert dataset.roots == [val] and dataset.blocklist_paths == (exclusions,)
    batch = create_dummy_batch(B=1, T=6, N=8, context_len=2)
    seen = []
    monkeypatch.setattr(evaluation, "build_domain_dataset", lambda *a, **k: seen.append(k) or Dataset([batch]))
    patch_sampler(monkeypatch, lambda m, b, *a, **k: b.prediction_stats.normalize(b.prediction_x0))
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    fn(SimpleNamespace(cfg=SimpleNamespace()), None, cfg, "mpm", max_clips=1, device=torch.device("cpu"),
       context_len=2, future_len=4, **({"horizon": 4} if rollout else {}))
    assert seen[0]["roots"] == resolve_data_roots(OmegaConf.to_container(cfg.data.roots))


def test_receipt_describes_the_release_that_was_scored(monkeypatch, tmp_path):
    from utils.reproducibility import file_sha256
    cfg, val, exclusions = _release_config(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock model loaded below")
    out = tmp_path / "result.json"
    monkeypatch.setattr(evaluation, "load_model_and_diffusion", lambda *a: (torch.nn.Identity(), None, cfg, 4))
    scored = []
    monkeypatch.setattr(evaluation, "eval_domain",
                        lambda *a, **k: scored.append(k["dataset"]) or {"n_episodes": 1, "particle_mean_mm": 0.0})
    argv = ["eval_unified.py", "--checkpoint", str(checkpoint), "--clips", "all", "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluation.main()
    inputs = json.loads(out.read_text())["inputs"]["mpm"]
    assert inputs["roots"] == scored[0].roots == [val]
    assert inputs["roots_source"] == "checkpoint data.roots"
    assert inputs["blocklists_sha256"] == {exclusions: file_sha256(exclusions)}
    # --mpm_val_root re-points the registered corpus; it cannot replace a saved release.
    monkeypatch.setattr(sys, "argv", argv + ["--mpm_val_root", str(tmp_path)])
    with pytest.raises(ValueError, match="own data.roots"):
        evaluation.main()


def test_batch_seed_is_a_function_of_only_the_seed_and_the_batch_index():
    seeds = {evaluation._batch_seed(s, i) for s in range(4) for i in range(40)}
    assert len(seeds) == 160                      # no collision between seeds and positions
    assert evaluation._batch_seed(0, 0) != 54321  # never the train.validate_sampler stream


@pytest.mark.parametrize("rollout", [False, True])
@pytest.mark.parametrize("seed", [0, 7])
def test_sampler_noise_is_reseeded_per_batch_not_per_process(monkeypatch, rollout, seed):
    """A domain's number must not depend on what was evaluated before it.

    The sampler draws from the global generator, so one process-level seed made
    pickup read 16.038 mm after mpm and pushing and 15.694 mm alone.
    """
    batches = [create_dummy_batch(B=1, T=6, N=8, context_len=2) for _ in range(3)]
    args = setup(monkeypatch, batches)
    patch_sampler(monkeypatch,
                        lambda m, b, *a, **k: b.prediction_stats.normalize(b.prediction_x0))
    drawn = []
    monkeypatch.setattr(evaluation, "seed_for_device", lambda s, d: drawn.append(s))
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    fn(*args, max_clips=3, device=torch.device("cpu"), seed=seed,
       **({"horizon": 4} if rollout else {}))
    assert drawn == [evaluation._batch_seed(seed, i) for i in range(3)]


@pytest.mark.parametrize("rollout", [False, True])
def test_receipt_records_the_shards_the_scored_clips_came_from(monkeypatch, rollout):
    batch = create_dummy_batch(B=2, T=6, N=8, context_len=2)
    batch.source_shard = "root/place/00057.tar"
    args = setup(monkeypatch, [batch])
    patch_sampler(monkeypatch, lambda m, b, *a, **k: b.prediction_stats.normalize(b.prediction_x0))
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    result = fn(*args, max_clips=None, device=torch.device("cpu"),
                **({"horizon": 4} if rollout else {}))
    assert result["shards_read"] == 1
    assert result["clips_per_shard_read"] == {"root/place/00057.tar": 2}
    json.dumps(result, allow_nan=False)


def test_clip_budget_is_exact_and_reads_no_record_past_it():
    from utils.evaluation import iter_clips
    pulled = []

    def records():
        for index in range(3):
            pulled.append(index)
            yield create_dummy_batch(B=2, T=6, N=8, context_len=2)

    assert [batch.batch_size for batch, _ in iter_clips(records(), 3)] == [2, 1]
    assert pulled == [0, 1]
    assert sum(batch.batch_size for batch, _ in iter_clips(records(), None)) == 6


def test_clips_flag_and_retired_max_batches(monkeypatch, tmp_path):
    from utils.evaluation import parse_clips
    assert parse_clips("all") is None and parse_clips("ALL") is None and parse_clips("12") == 12
    for bad in ("0", "-3", "1.5", "many", ""):
        with pytest.raises(ValueError, match="--clips"):
            parse_clips(bad)
    monkeypatch.setattr(evaluation, "load_model_and_diffusion", lambda *a: pytest.fail("checkpoint reached"))
    monkeypatch.setattr(sys, "argv", ["eval_unified.py", "--checkpoint", "unused", "--clips", "all",
                                      "--max_batches", "40", "--out", str(tmp_path / "out.json")])
    with pytest.raises(ValueError, match="retired"):
        evaluation.main()


def test_scores_average_within_each_clip_then_across_clips():
    """Clip-weighted, not particle-pooled: a 2-particle clip counts as much as an 8-particle one."""
    from utils.evaluation import TrajectoryScores
    batch = create_dummy_batch(B=2, T=6, N=8, context_len=2)
    batch.valid_lengths = torch.tensor([8, 2])
    pred = batch.future.clone()
    pred[0, ..., 0] += 0.001
    pred[1, ..., 0] += 0.003
    scores = TrajectoryScores(per_frame=True)
    scores.add(batch, pred, batch.future)
    result = scores.result()
    assert result["n_episodes"] == 2
    assert result["particle_mean_mm"] == pytest.approx(2.0, abs=1e-3)      # pooled would read 1.4
    assert result["particle_final_mm"] == pytest.approx(2.0, abs=1e-3)
    assert result["particle_per_frame_mm"] == pytest.approx([2.0] * 4, abs=1e-3)
    assert result["horizon_frames_min"] == result["horizon_frames_max"] == 4
    first_clip = torch.zeros(2, 8, dtype=torch.bool)
    first_clip[0] = True
    subset = TrajectoryScores()
    subset.add(batch, pred, batch.future, rows=first_clip)
    assert subset.result()["n_episodes"] == 1
    assert subset.result()["particle_mean_mm"] == pytest.approx(1.0, abs=1e-3)
    assert "particle_mse_mm2" not in subset.result() and "chamfer_mean_mm" not in subset.result()   # opt-in keys


def test_per_frame_curves_share_one_curve_across_clip_lengths_with_mse_and_chamfer():
    """The evaluation protocol's receipt keys: at frame h the mean over the clips that reach h, and how many do."""
    from utils.evaluation import TrajectoryScores, chamfer_per_frame
    scores = TrajectoryScores(per_frame=True, chamfer=True)
    for frames, shift in ((6, 0.001), (4, 0.003)):                        # 4 and 2 scored frames, 1 mm and 3 mm off
        batch = create_dummy_batch(B=1, T=frames, N=8, context_len=2)
        pred = batch.future.clone()
        pred[..., 0] += shift
        scores.add(batch, pred, batch.future)
        assert scores.last_curves["particle"] == pytest.approx([shift * 1000.0] * (frames - 2), abs=1e-3)
    result = scores.result()
    assert result["particle_per_frame_mm"] == pytest.approx([2.0, 2.0, 1.0, 1.0], abs=1e-3)
    assert result["particle_per_frame_n"] == [2, 2, 1, 1]
    assert result["particle_mse_mm2"] == pytest.approx(5.0, abs=1e-2)             # (1 + 9) / 2
    assert result["particle_mse_per_frame_mm2"] == pytest.approx([5.0, 5.0, 1.0, 1.0], abs=1e-2)
    assert (result["horizon_frames_min"], result["horizon_frames_max"]) == (2, 4)
    # Chamfer needs no correspondence: a permutation of the predicted points leaves it unchanged, and it is at most the
    # matched distance (twice the shift here: both directions are summed).
    assert len(result["chamfer_per_frame_mm"]) == 4 and 0 < result["chamfer_mean_mm"] <= 2 * 2.0 + 1e-3
    points = torch.rand(3, 8, 3)
    moved = points + torch.tensor([0.002, 0.0, 0.0])
    assert torch.allclose(chamfer_per_frame(moved, points), chamfer_per_frame(moved[:, torch.randperm(8)], points))
    assert torch.allclose(chamfer_per_frame(moved, points, budget=1), chamfer_per_frame(moved, points))   # frame chunks
    assert float(chamfer_per_frame(points, points).abs().max()) == 0.0


@pytest.mark.parametrize("context,future,horizon,expected", [
    (4, 16, 36, [(4, 16, 0), (20, 16, 0), (36, 4, 0)]),       # pickup: 40-frame clips
    (4, 16, 45, [(4, 16, 0), (20, 16, 0), (36, 13, 0)]),      # mpm: 49-frame clips
    (4, 16, 5, [(4, 5, 0)]),                                   # a horizon shorter than one window
    (4, 38, 48, [(4, 38, 0), (42, 10, 0)]),
])
def test_rollout_windows_advance_from_latest_context_and_end_on_horizon(context, future, horizon, expected):
    committed, target, windows = context, context + horizon, []
    while committed < target:
        start, length, skip = inference.window_bounds(committed, target, future, context)
        windows.append((start, length, skip))
        committed += length - skip
    assert windows == expected


def test_windowed_rollout_commits_every_frame_once_from_its_own_output(monkeypatch):
    batch = create_dummy_batch(B=1, T=10, N=8, context_len=2)
    seen = []

    def oracle(model, b, *a, **k):
        seen.append(b.positions.shape[1])
        return b.prediction_stats.normalize(b.prediction_x0)

    monkeypatch.setattr(inference, "run_diffusion_inference_df", oracle)
    out = inference.windowed_rollout(SimpleNamespace(cfg=SimpleNamespace()), None, batch, horizon=7, future_len=4)
    assert out["windows"] == [(2, 4, 0), (6, 3, 0)] and out["horizon"] == 7 and out["R"] is None
    assert seen == [6, 5]
    torch.testing.assert_close(out["positions"], batch.positions[:, :9], atol=1e-6, rtol=0)


def test_shard_groups_nest_roots_then_directories():
    from data.dataset import stratified_shard_groups
    shards = ["/r/place/gpu0/1.tar", "/r/place/gpu0/2.tar", "/r/place/gpu1/1.tar", "/r/pick/gpu0/1.tar"]
    assert stratified_shard_groups(shards, ["/r/place", "/r/pick/"]) == [
        [["/r/place/gpu0/1.tar", "/r/place/gpu0/2.tar"], ["/r/place/gpu1/1.tar"]],
        [["/r/pick/gpu0/1.tar"]]]


def _mpm_shard(path, records, frames=49, particles=5):
    """A small passive-MPM shard holding `records` records of two clips each."""
    import io
    import pickle
    import tarfile
    path.parent.mkdir(parents=True, exist_ok=True)
    material = np.zeros((2, particles, 4), np.float32)
    material[..., 0], material[..., 1], material[..., 2] = 1000, 0.3, 400
    values = dict(particle_pos=np.zeros((2, frames, particles, 3), np.float32),
                  valid_particles=np.ones((2, frames, particles), bool),
                  material_properties=material, object_ids=np.zeros((2, particles), np.int64))
    with tarfile.open(path, "w") as stream:
        for key in range(records):
            for name, value in values.items():
                encoded = pickle.dumps(value)
                member = tarfile.TarInfo(f"{key:04d}.{name}.pyd")
                member.size = len(encoded)
                stream.addfile(member, io.BytesIO(encoded))


def test_evaluation_reads_alternate_directories_and_keep_every_record(tmp_path):
    """Any prefix of an evaluation read covers every directory; training reads keep sorted order."""
    from data.unified_dataset import UnifiedMPMDataset
    for records, shard in ((2, "1obj/00000.tar"), (1, "1obj/00001.tar"), (2, "2obj/00000.tar")):
        _mpm_shard(tmp_path / "wds" / shard, records)

    def order(training, stratify=None):
        ds = UnifiedMPMDataset(wds_location=str(tmp_path / "wds"), context_len=4, future_len=16,
                               training=training, shuffle=False, normalization=resolve_normalization())
        if stratify is not None:
            ds.stratify_shards = stratify
        return [Path(batch.source_shard).relative_to(tmp_path / "wds").as_posix() for batch in ds]

    in_sorted_order = ["1obj/00000.tar", "1obj/00000.tar", "1obj/00001.tar", "2obj/00000.tar", "2obj/00000.tar"]
    assert order(training=False) == ["1obj/00000.tar", "2obj/00000.tar", "1obj/00000.tar",
                                     "2obj/00000.tar", "1obj/00001.tar"]
    assert order(training=False, stratify=False) == in_sorted_order
    assert order(training=True) == in_sorted_order


@pytest.mark.parametrize("rollout", [False, True])
@pytest.mark.parametrize("schedule,expected", [(None, "pyramid"), ("full_sequence", "full_sequence")])
def test_schedule_override_reaches_every_sampled_window(monkeypatch, rollout, schedule, expected):
    batch = create_dummy_batch(B=1, T=10, N=8, context_len=2)
    args = setup(monkeypatch, [batch])
    used = []

    def oracle(model, b, *a, **k):
        used.append(k["schedule"])
        return b.prediction_stats.normalize(b.prediction_x0)

    patch_sampler(monkeypatch, oracle)
    fn = evaluation.rollout_domain if rollout else evaluation.eval_domain
    fn(*args, max_clips=None, device=torch.device("cpu"), future_len=4, schedule=schedule,
       **({"horizon": 7} if rollout else {}))
    assert len(used) == (2 if rollout else 1) and set(used) == {expected}


@pytest.mark.parametrize("flag,used", [([], "pyramid"), (["--schedule", "full_sequence"], "full_sequence")])
def test_cli_schedule_defaults_to_the_checkpoint_and_records_an_override(monkeypatch, tmp_path, flag, used):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mock model loaded below")
    out = tmp_path / "result.json"
    cfg = OmegaConf.create({"normalization": resolve_normalization(), "df_schedule": "pyramid",
                            "data": {"context_len": 2, "future_len": 4, "domains": ["mpm"]},
                            "flow_matching": {"inference_steps": 4}})
    monkeypatch.setattr(evaluation, "load_model_and_diffusion", lambda *a: (torch.nn.Identity(), None, cfg, 4))
    monkeypatch.setattr(evaluation, "build_domain_dataset",
                        lambda *a, **k: SimpleNamespace(roots=["registered-mpm-val"], blocklist_paths=()))
    calls = []
    monkeypatch.setattr(evaluation, "eval_domain",
                        lambda *a, **k: calls.append(k) or {"n_episodes": 1, "particle_mean_mm": 0.0})
    monkeypatch.setattr(sys, "argv", ["eval_unified.py", "--checkpoint", str(checkpoint), "--clips", "all",
                                      *flag, "--out", str(out)])
    evaluation.main()
    result = json.loads(out.read_text())
    assert calls[0]["schedule"] == result["df_schedule"] == used
    assert result["checkpoint_df_schedule"] == "pyramid"
