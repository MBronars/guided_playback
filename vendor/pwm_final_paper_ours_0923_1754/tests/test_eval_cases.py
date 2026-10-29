"""Frozen evaluation cases: row sets, trivial references, selection rules and the freezer's fidelity to the loaders."""
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.eval_cases import (FIELDS, MOVING_MM, active_start, case_numbers, hash_order, load_cases, motion_profile,
                              predicted_rows, quantile_starts, scored_rows, span_scores, trivial_predictions)
from scripts.unified.freeze_eval_cases import (PUSHING_ROLLOUT_CLIPS, build_manifest, freeze_indexed, freeze_pickup,
                                               freeze_pushing, freeze_rl_push, summarize, write_manifest)
from tests.test_rl_push import _dataset, release  # noqa: F401  (pytest fixture)

ARGS = dict(limit=0, rollouts=0, rollout_frames=32, verify_every=1, verify_stream=2)
COLUMN = {name: i for i, name in enumerate(FIELDS)}


def sidecar_batch():
    from tests.test_unet_std_pose_native import make_batch
    return make_batch()


def test_predicted_rows_follow_the_overfit_harness_rule():
    from scripts.unified.overfit_prod import primary_mask
    from tests.test_train import create_dummy_batch
    batch = sidecar_batch()
    assert batch.action_mask.any() and torch.equal(predicted_rows(batch), batch.get_valid_mask() & ~batch.action_mask)
    assert torch.equal(predicted_rows(batch), primary_mask(batch))
    plain = create_dummy_batch(B=1, T=8, N=16, context_len=2)          # no rigid sidecar: commanded particles stay
    assert plain.body_type is None and torch.equal(predicted_rows(plain), plain.get_valid_mask())
    assert torch.equal(predicted_rows(plain), primary_mask(plain))


def test_moving_rows_need_five_millimetres_from_the_anchor():
    batch = sidecar_batch()
    C = batch.context_len
    positions = batch.positions[:1, :C + 4].clone()
    positions[:] = positions[:, C - 1:C]                               # everything rests at the anchor
    free = predicted_rows(batch)[0].nonzero()[:, 0]
    positions[0, C + 1, free[0], 0] += (MOVING_MM + .1) / 1000          # leaves and returns: still counts as moving
    positions[0, C:, free[1], 1] += (MOVING_MM - .1) / 1000
    actuated = batch.action_mask[0].nonzero()[0, 0]
    positions[0, C:, actuated, 2] += .05                               # an actuator is never a predicted row
    rows = scored_rows(batch, positions)
    assert rows["moving"][0].nonzero()[:, 0].tolist() == [int(free[0])]
    assert torch.equal(rows["moving"] | rows["still"], rows["predicted"]) and not (rows["moving"] & rows["still"]).any()
    assert not rows["predicted"][0, actuated] and rows["all"][0, actuated]


def test_case_numbers_are_the_scorer_on_the_trivial_predictions():
    from utils.evaluation import TrajectoryScores
    batch = sidecar_batch()
    C = batch.context_len
    zero, hold = trivial_predictions(batch.positions, C)
    numbers = case_numbers(batch)
    assert len(numbers) == batch.positions.shape[0] and len(numbers[0]) == len(FIELDS) - 5
    mean = lambda name: float(np.mean([row[COLUMN[name] - 5] for row in numbers]))
    for pred, rows, key, final in ((zero, "all", "zero_all_mm", None), (zero, "predicted", "zero_predicted_mm", "zero_predicted_final_mm"),
                                   (hold, "predicted", "hold_predicted_mm", "hold_predicted_final_mm"),
                                   (zero, "moving", "zero_moving_mm", None)):
        scores = TrajectoryScores()
        scores.add(batch, pred, batch.positions[:, C:], rows=scored_rows(batch)[rows], poses=False)
        result = scores.result()
        assert mean(key) == pytest.approx(result["particle_mean_mm"], abs=2e-3)
        if final:
            assert mean(final) == pytest.approx(result["particle_final_mm"], abs=2e-3)
    scores = TrajectoryScores()                                       # pose references: the scorer's (clip, body) pair average
    scores.add(batch, zero, batch.positions[:, C:])
    result = scores.result()
    bodies = [row[COLUMN["bodies_passive"] - 5] for row in numbers]
    assert bodies == [1] * len(numbers)                                # the rigid object; the gripper links are actuated
    for key, scored in (("zero_pose_passive_mm", "pose_passive_mm"), ("zero_pose_passive_deg", "pose_passive_deg")):
        ours = np.average([row[COLUMN[key] - 5] for row in numbers], weights=bodies)
        assert ours == pytest.approx(result[scored], abs=5e-3) and ours > 0
    step = (batch.positions[0, C:] - batch.positions[0, C - 1:-1]).norm(dim=-1)[:, predicted_rows(batch)[0]].mean() * 1000
    assert numbers[0][COLUMN["motion_mm_per_frame"] - 5] == pytest.approx(float(step), abs=1e-3)
    assert numbers[0][0] == int(batch.get_valid_mask()[0].sum()) and numbers[0][1] == int(predicted_rows(batch)[0].sum())


def test_case_numbers_score_a_slice_and_refuse_unscorable_clips():
    batch = sidecar_batch()
    C = batch.context_len
    part = batch.positions[:, 1:C + 3]
    sliced = case_numbers(batch, 1, C + 2)
    zero, _ = trivial_predictions(part, C)
    rows = predicted_rows(batch)[0]
    expected = (zero[0][:, rows] - part[0, C:][:, rows]).norm(dim=-1).mean() * 1000
    assert sliced[0][COLUMN["zero_predicted_mm"] - 5] == pytest.approx(float(expected), abs=1e-3)
    with pytest.raises(ValueError, match="scored frame"):
        case_numbers(batch, 0, C)
    # A span's pose reference is the pose change inside the span, whichever frame the templates came from.
    from utils.evaluation import TrajectoryScores
    scores = TrajectoryScores()
    scores.add(batch, zero, part[:, C:], poses=False)
    from utils.evaluation import rigid_pose_errors
    errors = [(deg.mean(), mm.mean()) for _, _, actuated, deg, mm in
              rigid_pose_errors(batch, zero, batch.body_poses[:, 1 + C:1 + C + 2]) if not actuated]
    assert sliced[0][COLUMN["zero_pose_passive_deg"] - 5] == pytest.approx(float(errors[0][0]), abs=5e-3)
    assert sliced[0][COLUMN["zero_pose_passive_mm"] - 5] == pytest.approx(float(errors[0][1]), abs=5e-3)
    still = batch.slice(1)
    still.positions = still.positions[:, :1].expand(-1, C + 2, -1, -1).clone()
    numbers = case_numbers(still)[0]
    assert numbers[COLUMN["zero_moving_mm"] - 5] is None and numbers[COLUMN["rows_moving"] - 5] == 0     # no number, not zero


def test_span_scores_are_the_loaders_window_motion_rule():
    rng = np.random.default_rng(0)
    profile, first, C, F = rng.random(60) * .01, 10, 4, 16
    starts, score = span_scores(profile, first, C, F)
    frames = first + len(profile) + 1
    assert starts.tolist() == list(range(first, frames - C - F + 1))
    for s, value in zip(starts, score):                                # the F transitions into the window's future frames
        assert value == pytest.approx(profile[s - first + C - 1:s - first + C + F - 1].mean(), abs=1e-7)
    with pytest.raises(ValueError, match="too short"):
        span_scores(profile[:10], first, C, F)


def test_motion_selection_is_deterministic_and_prefers_the_earliest_tie():
    profile = np.zeros(200)
    profile[100:140] = .002                                            # 40 moving transitions
    assert active_start(profile, 0, 4, 32) == 97                       # first start whose 32 scored transitions all move
    assert active_start(np.zeros(50), 5, 4, 16) == 5
    picks = quantile_starts(profile, 0, 4, 16)
    assert len(picks) == 4 and picks == quantile_starts(profile.copy(), 0, 4, 16)
    _, score = span_scores(profile, 0, 4, 16)
    chosen = [score[s] for s in picks]
    assert chosen == sorted(chosen) and chosen[0] == 0 and chosen[-1] > 0
    names = [f"shard-{i}" for i in range(8)]
    assert hash_order(names, "salt:") == hash_order(names[::-1], "salt:") != sorted(names)
    assert hash_order(names, "salt:") != hash_order(names, "other:") and sorted(hash_order(names, "salt:")) == names


def test_motion_profile_is_the_mean_step_of_the_predicted_rows():
    batch = sidecar_batch()
    with pytest.raises(ValueError, match="one clip"):
        motion_profile(batch)
    one = batch.slice(1)
    rows = predicted_rows(one)[0]
    expected = (one.positions[0, 1:, rows] - one.positions[0, :-1, rows]).norm(dim=-1).mean(dim=1)
    profile = motion_profile(one)
    assert profile.shape == (one.positions.shape[1] - 1,) and profile.dtype == np.float64
    np.testing.assert_allclose(profile, expected.numpy(), atol=1e-9)


def _long_yikai(tmp_path):
    """The yikai fixture stretched to 200 frames; its cup slides only over the transitions 99->100 .. 138->139,
    so the earliest 32-frame span that moves throughout starts at frame 96 (scored transitions 99..130)."""
    from scripts.unified.prepare_indexed_datasets import file_identity
    from tests.test_indexed_episodes import index, yikai_source
    record, names, pid = yikai_source(tmp_path, frames=200)
    with np.load(record["file"]) as source:
        arrays = {key: source[key] for key in source.files}
    pos = arrays["state/particles"]
    cup = pid == names.index("cup_0001")
    pos[:, cup] = pos[:1, cup]
    pos[100:, cup, 0] += (np.minimum(np.arange(100, 200), 139) - 99)[:, None].astype(np.float32) * .002
    np.savez(record["file"], **arrays)
    record.update(file_identity(record["file"]))
    return index(tmp_path / "test", [record], "yikai", split="test"), record


def test_indexed_freezer_pins_the_loaders_window_and_motion_selected_cases(tmp_path, monkeypatch):
    from data.indexed_episodes import IndexedEpisodeDataset
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    root, record = _long_yikai(tmp_path)
    ds = IndexedEpisodeDataset(root, domain="yikai", raw=True, training=False)
    ds.emit_sidecar = True
    sets = freeze_indexed(ds, "yikai", Namespace(**ARGS), {record["episode_id"]: [[150, 160]]})
    assert set(sets) == {"windows", "windows_motion", "rollouts_first", "rollouts_active"}
    next(iter(ds))                                                     # the loader's own evaluation window
    assert sets["windows"]["cases"][0][:3] == [record["episode_id"], ds.last_windows[0]["start"], 16]
    assert sets["windows"]["cases"][0][3] == "gen" and sets["windows"]["evaluator"].startswith("eval_unified.py --clips all")
    assert sets["rollouts_active"]["evaluator"] == ("eval_unified.py --cases <this manifest> --case_set rollouts_active "
                                                    "--rollout --rollout_frames 32 --clips all")
    motion = sets["windows_motion"]["cases"]
    assert [case[4].split(",")[0] for case in motion] == ["q1", "q2", "q3", "q4"]
    assert motion[0][COLUMN["motion_mm_per_frame"]] == 0 and motion[0][COLUMN["rows_moving"]] == 0
    assert 84 <= motion[3][1] <= 135 and motion[3][COLUMN["rows_moving"]] == 32      # the cup's 32 points, mid-slide
    first, active = sets["rollouts_first"]["cases"][0], sets["rollouts_active"]["cases"][0]
    assert first[1:3] == [10, 32] and first[COLUMN["rows_moving"]] == 0
    assert active[1:3] == [96, 32] and active[COLUMN["zero_moving_mm"]] > first[COLUMN["zero_predicted_mm"]]
    beyond = {name: [("beyond" in case[4].split(",")) for case in entry["cases"]] for name, entry in sets.items()}
    assert beyond["rollouts_first"] == [False] and beyond["rollouts_active"] == [False]
    for case in motion + sets["windows"]["cases"]:                     # frames 150..160 are tagged, nothing else
        assert ("beyond" in case[4].split(",")) == (case[1] <= 160 and case[1] + 19 >= 150)


def test_indexed_freezer_adds_whole_episode_rollouts_on_request(tmp_path, monkeypatch):
    """--full_rollouts: the rollout episodes from their first allowed frame to their last; every other set is unchanged,
    so a manifest frozen before the option existed still reproduces."""
    from data.indexed_episodes import IndexedEpisodeDataset
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    root, record = _long_yikai(tmp_path)
    frozen = {}
    for full in (False, True):
        ds = IndexedEpisodeDataset(root, domain="yikai", raw=True, training=False)
        ds.emit_sidecar = True
        frozen[full] = freeze_indexed(ds, "yikai", Namespace(**ARGS, full_rollouts=full), {})
    assert set(frozen[True]) == set(frozen[False]) | {"rollouts_full"}
    assert all(frozen[True][name] == frozen[False][name] for name in frozen[False])
    whole = frozen[True]["rollouts_full"]
    assert [case[:3] for case in whole["cases"]] == [[record["episode_id"], 10, 200 - 10 - 4]]   # frames 10..199, 4 of them context
    assert whole["evaluator"] == "eval_unified.py --cases <this manifest> --case_set rollouts_full --rollout --rollout_frames 186 --clips all"
    assert whole["cases"][0][COLUMN["rows_moving"]] == 32                # the cup slides within the whole episode
    first = frozen[True]["rollouts_first"]["cases"][0]
    assert whole["cases"][0][COLUMN["zero_predicted_final_mm"]] > first[COLUMN["zero_predicted_final_mm"]]


def test_indexed_freezer_handles_short_passive_episodes(tmp_path, monkeypatch):
    """MPM clips (the drafts of docs/EVAL_PROTOCOLS.md): no rigid sidecar, every particle predicted, whole-clip
    rollouts from frame 0 and no motion-selected sets."""
    from data.indexed_episodes import IndexedEpisodeDataset
    from tests.test_indexed_episodes import index, mpm_source
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    records, _ = mpm_source(tmp_path, episodes=3, moving_from=30)
    records = [dict(record, evaluation_category="passive_unseen") for record in records]      # the MPM suite's record field
    ds = IndexedEpisodeDataset(index(tmp_path / "test", records, "mpm_fresh", split="test"), domain="mpm_fresh", raw=True, training=False)
    ds.emit_sidecar = True
    sets = freeze_indexed(ds, "mpm_fresh", Namespace(**{**ARGS, "rollout_frames": 128, "rollouts": 2}), {})
    assert set(sets) == {"windows", "rollouts_first"}
    assert len(sets["windows"]["cases"]) == 3 and len(sets["rollouts_first"]["cases"]) == 2
    assert {case[COLUMN["stratum"]] for case in sets["windows"]["cases"]} == {"passive_unseen"}
    # a report folder's manifest names the option that scores it, in every command of every set
    from scripts.unified.freeze_eval_cases import name_test_folder
    named = name_test_folder(sets, "mpm_fresh", tmp_path / "test")
    assert all(entry["evaluator"].count(f"--test_folder mpm_fresh={(tmp_path / 'test').resolve()} ") == entry["evaluator"].count("eval_unified.py ") >= 1
               for entry in named.values())
    rollout = sets["rollouts_first"]["cases"][0]
    assert rollout[1:3] == [0, 45] and rollout[COLUMN["rows_all"]] == rollout[COLUMN["rows_predicted"]] == 8
    assert rollout[COLUMN["bodies_passive"]] == 0 and rollout[COLUMN["zero_pose_passive_mm"]] is None
    # the scene rests until frame 30 and then moves 1 mm per frame: 19 moving transitions by the last frame
    assert rollout[COLUMN["zero_predicted_final_mm"]] == pytest.approx(19.0, abs=1e-2)


def test_indexed_freezer_takes_dex_numbers_from_the_adapters_own_window(tmp_path, monkeypatch):
    """Dex positions are computed per window, so a slice of the episode may differ in the last float32 bit; the frozen
    numbers must be those of the clip the reader yields."""
    from data.indexed_episodes import IndexedEpisodeDataset
    from tests.test_indexed_episodes import dex_source, index
    from utils.eval_cases import CaseDataset
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "test", [dex_source(tmp_path)], "dex", split="test"), domain="dex", raw=True,
                               training=False)
    ds.emit_sidecar = True
    calls = []
    real = ds.window
    monkeypatch.setattr(ds, "window", lambda rec, start: calls.append((start, ds.future_len)) or real(rec, start))
    sets = freeze_indexed(ds, "dex", Namespace(**{**ARGS, "verify_stream": 0}), {})
    case = sets["windows"]["cases"][0]
    assert (case[1], 16) in calls and ds.future_len == 16             # the case's own window was built, and the length restored
    assert case[COLUMN["rows_predicted"]] < case[COLUMN["rows_all"]] and case[COLUMN["bodies_passive"]] == 1   # the object; the hand is given
    path = _manifest(tmp_path / "dex_v1.json", sets, ds, "dex")
    for name in sets:
        for frozen, clip in zip(load_cases(path, name), CaseDataset(ds, path, name)):
            ours, theirs = case_numbers(clip)[0], [frozen[key] for key in FIELDS[5:]]
            assert ours[:3] == theirs[:3] and ours[10] == theirs[10]                   # row and body counts exactly
            assert ours == pytest.approx(theirs, abs=2e-4)                              # numbers to the stored precision


def test_indexed_freezer_refuses_a_window_that_differs_from_its_episode(tmp_path, monkeypatch):
    from data.indexed_episodes import IndexedEpisodeDataset
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    root, _ = _long_yikai(tmp_path)
    ds = IndexedEpisodeDataset(root, domain="yikai", raw=True, training=False)
    ds.emit_sidecar = True
    real = ds.window

    def shifted(record, start):
        batch = real(record, start)
        if batch.positions.shape[1] == 20:
            batch.positions = batch.positions + .001
        return batch
    monkeypatch.setattr(ds, "window", shifted)
    with pytest.raises(ValueError, match="differs from the slice"):
        freeze_indexed(ds, "yikai", Namespace(**{**ARGS, "verify_stream": 0}), {})


def test_rl_push_freezer_pins_the_loaders_windows_and_spreads_rollouts(release):  # noqa: F811
    ds = _dataset(release, training=False)
    ds.emit_sidecar = True
    sets = freeze_rl_push(ds, Namespace(**{**ARGS, "rollouts": 4}))
    windows = sets["windows"]["cases"]
    ds.batch_size = 1
    read = list(ds)                                                    # the evaluator's clips, in its order
    assert len(windows) == len(read) == 6
    for case, batch in zip(windows, read):
        assert case[0].split(":")[0] == batch.source_shard and case[2] == 16
        assert case[5:] == pytest.approx(case_numbers(batch)[0], abs=2e-3) and case[COLUMN["bodies_passive"]] == 1
    rollouts = sets["rollouts_first"]["cases"]
    objects = [case[0].split(":")[0] for case in rollouts]
    assert len(rollouts) == 4 and len(set(objects[:3])) == 3           # one episode of every object before a second
    assert all(case[1] == ds.first_start and case[2] >= 16 for case in rollouts)
    assert {case[0] for case in rollouts} <= {case[0] for case in windows}
    assert freeze_rl_push(ds, Namespace(**{**ARGS, "rollouts": 4}))["rollouts_first"]["cases"] == rollouts


def _pushing_shards(root, shards=3, records=3):
    from tests.test_source_io import write_shard
    rng = np.random.default_rng(5)
    root.mkdir()
    for s in range(shards):
        rows = []
        for r in range(records):
            base = rng.normal(size=(2, 1, 8, 3)).astype(np.float32) / 50
            slide = np.arange(40, dtype=np.float32)[None, :, None, None] * np.array([.002, 0, 0], np.float32)
            rows.append((f"rec{r:03d}", dict(particle_pos=base + slide, obj_ids=np.tile(np.array([0] * 4 + [1] * 4), (2, 1)),
                                             material_properties=np.ones((2, 8, 4), dtype=np.float32))))
        write_shard(root / f"{s:05d}.tar", rows)
    return root


def test_pushing_freezer_pins_prefix_windows_and_whole_clip_rollouts(tmp_path):
    from data.unified_dataset import UnifiedPushingDataset
    root = _pushing_shards(tmp_path / "test")
    ds = UnifiedPushingDataset([str(root)], raw=True, context_len=4, future_len=16, split="all", training=False, shuffle=False)
    ds.emit_sidecar = True
    sets = freeze_pushing(ds, Namespace(**ARGS))
    windows, rollouts = sets["windows"]["cases"], sets["rollouts_first"]["cases"]
    assert [case[0] for case in windows[:3]] == ["00000.tar:rec000:0", "00000.tar:rec000:1", "00000.tar:rec001:0"]
    assert len(windows) == 3 * 3 * 2 and all(case[1:3] == [0, 16] for case in windows)
    assert ds.future_len == 16                                          # the rollout convention is undone
    read = [batch for batch in ds]
    assert sum(batch.batch_size for batch in read) == len(windows) and read[0].positions.shape[1] == 20
    # The prefix case of a 40-frame record is the 20-frame window the loader yields, pose reference included.
    assert windows[0][5:] == pytest.approx(case_numbers(read[0])[0], abs=2e-3) and windows[0][COLUMN["bodies_passive"]] == 1
    assert windows[1][5:] == pytest.approx(case_numbers(read[0])[1], abs=2e-3)
    assert windows[0][COLUMN["zero_pose_passive_mm"]] == pytest.approx(2.0 * 17 / 2, abs=1e-2)   # 2 mm per frame, frames 1..16
    assert len(rollouts) == 3 * min(PUSHING_ROLLOUT_CLIPS, 6) and all(case[1:3] == [0, 36] for case in rollouts)
    assert {case[0] for case in rollouts} <= {case[0] for case in windows}
    assert windows[0][COLUMN["rows_predicted"]] == 4 and windows[0][COLUMN["rows_all"]] == 8      # the rod is an input


def _pickup_shards(root, collections=("pick", "place"), records=2):
    """Two test shards named after their collection, as pwm-data-v1/pickup/test has them: gripper rows first (hand 6, fingers 4 + 4),
    then 6 object rows; every body slides rigidly, the object slower than the gripper."""
    from tests.test_source_io import write_shard
    rng = np.random.default_rng(11)
    root.mkdir()
    npts = np.zeros(14, dtype=np.int32)
    npts[8:11] = (6, 4, 4)
    quat = np.tile(np.array([1, 0, 0, 0], np.float32), (2, 40, 14, 1))
    for name in collections:
        rows = []
        for r in range(records):
            base = rng.normal(size=(2, 1, 20, 3)).astype(np.float32) / 50
            speed = np.concatenate([np.full(14, .003, np.float32), np.full(6, .001, np.float32)])
            slide = np.arange(40, dtype=np.float32)[None, :, None, None] * speed[None, None, :, None] * np.array([1, 0, 0], np.float32)
            rows.append((f"batch_{r:08d}", dict(particle_pos=base + slide, obj_ids=np.tile(np.array([1] * 14 + [0] * 6), (2, 1)),
                                                material_properties=np.ones((2, 20, 4), dtype=np.float32),
                                                link_local_npts=npts, link_quat=quat)))
        write_shard(root / f"{name}.tar", rows)
    return root


def test_pickup_freezer_pins_every_episode_as_window_and_rollout_with_its_collection(tmp_path):
    from data.unified_dataset import UnifiedPickupDataset
    from utils.eval_cases import CaseDataset
    root = _pickup_shards(tmp_path / "test")
    (root / "INDEX.json").write_text(json.dumps(dict(schema="pwm-shard-index-v1", domain="pickup", split="test")))
    ds = UnifiedPickupDataset([str(root)], raw=True, context_len=4, future_len=16, split="all", training=False, shuffle=False)
    ds.emit_sidecar = True
    sets = freeze_pickup(ds, Namespace(**ARGS))
    windows, rollouts = sets["windows"]["cases"], sets["rollouts_first"]["cases"]
    assert len(windows) == len(rollouts) == 2 * 2 * 2 and ds.future_len == 16
    assert [case[0] for case in windows] == [case[0] for case in rollouts]            # every episode is rolled out
    assert {case[3] for case in windows} == {"pick", "place"} and windows[0][0] == "pick.tar:batch_00000000:0"
    assert all(case[1:3] == [0, 16] for case in windows) and all(case[1:3] == [0, 36] for case in rollouts)
    assert windows[0][COLUMN["rows_predicted"]] == 6 and windows[0][COLUMN["rows_all"]] == 20       # the gripper is an input
    assert windows[0][COLUMN["zero_pose_passive_mm"]] == pytest.approx(1.0 * 17 / 2, abs=1e-2)   # 1 mm per frame, frames 1..16
    path = _manifest(tmp_path / "pickup_v1.json", sets, ds, "pickup")
    for name, frames in (("windows", 20), ("rollouts_first", 40)):
        cases, clips = load_cases(path, name), list(CaseDataset(ds, path, name))
        assert len(clips) == len(cases) == 8 and ds.future_len == 16 and len(ds.shards) == 2
        for case, clip in zip(cases, clips):
            assert clip.positions.shape[:2] == (1, frames) and Path(clip.source_shard).name == case["id"].split(":")[0]
            assert case_numbers(clip)[0] == pytest.approx([case[key] for key in FIELDS[5:]], abs=2e-3)


def test_manifest_round_trip_and_reader_checks(tmp_path):
    case = ["episode", 3, 16, "gen", "q1", 10, 8, 2, .5, 1., 2., 3., 4., 5., 6., 3, 9., 1.5]
    other = ["episode", 9, 16, "fail", "", 10, 8, 0, 0., 0., 0., 0., None, 0., 0., 1, 1., .5]
    sets = {"windows": dict(selection="rule", evaluator="command", cases=[case, other])}
    manifest = build_manifest(sets, dataset="yikai", domain="yikai", inputs={"roots": ["/x"]}, code={"git_commit": "c"},
                              window={"context_len": 4, "future_len": 16, "first_start": 10}, recipe={"path": "r", "sha256": "s"})
    summary = manifest["sets"]["windows"]["summary"]
    assert summary["cases"] == 2 and summary["episodes"] == 1 and summary["cases_with_moving_rows"] == 1
    assert summary["zero_moving_mm"] == 4.0 and summary["zero_predicted_mm"] == 1.0          # None is skipped, not zero
    assert summary["zero_pose_passive_mm"] == 5.0 and summary["zero_pose_passive_pairs_mm"] == 7.0   # (3 * 9 + 1 * 1) / 4
    assert summary["zero_pose_passive_pairs_deg"] == 1.25 and "bodies_passive" not in summary
    none = summarize([case[:15] + [0, None, None]])
    assert none["zero_pose_passive_pairs_mm"] is None and none["zero_pose_passive_mm"] is None
    assert set(summary["by_stratum"]) == {"fail", "gen"} and set(summary["by_tag"]) == {"q1"}
    assert summarize([case])["horizon_max"] == 16 and "by_stratum" not in summarize([case])
    path = tmp_path / "cases" / "yikai_v1.json"
    write_manifest(path, manifest)
    assert path.read_text().count("\n") == 3                           # one case per line
    assert load_cases(path) == json.loads(json.dumps(manifest))
    assert load_cases(path, "windows")[0]["start"] == 3 and load_cases(path, "windows")[1]["zero_moving_mm"] is None
    with pytest.raises(ValueError, match="no case set"):
        load_cases(path, "rollouts")
    for damage in (dict(kind="other"), dict(version=2), dict(sets={"windows": dict(fields=list(FIELDS), cases=[case, case])}),
                   dict(sets={"windows": dict(fields=list(FIELDS), cases=[case[:-1]])}),
                   dict(sets={"windows": dict(fields=list(FIELDS), cases=[])})):
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({**manifest, **damage}))
        with pytest.raises(ValueError):
            load_cases(broken)



def _manifest(path, sets, ds, domain):
    from scripts.unified.eval_unified import _input_provenance
    inputs = _input_provenance(ds, {"sha256": "test"})
    if "episode_index_sha256" not in inputs:
        from utils.reproducibility import file_sha256
        inputs["folder_index_sha256"] = {str(Path(root) / "INDEX.json"): file_sha256(Path(root) / "INDEX.json")
                                         for root in inputs["roots"] if (Path(root) / "INDEX.json").is_file()}
    write_manifest(path, build_manifest(sets, dataset=domain, domain=domain, inputs=inputs, code={"git_commit": "c", "git_dirty": False},
                                        window={"context_len": 4, "future_len": 16, "first_start": int(getattr(ds, "first_start", 0))},
                                        recipe={"path": "r", "sha256": "s"}))
    return path


def test_case_reader_yields_the_frozen_clips_of_an_indexed_dataset(tmp_path, monkeypatch):
    from data.indexed_episodes import IndexedEpisodeDataset
    from utils.eval_cases import CaseDataset
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    root, record = _long_yikai(tmp_path)
    ds = IndexedEpisodeDataset(root, domain="yikai", raw=True, training=False)
    ds.emit_sidecar = True
    path = _manifest(tmp_path / "yikai_v1.json", freeze_indexed(ds, "yikai", Namespace(**ARGS), {}), ds, "yikai")
    for name in ("windows", "windows_motion", "rollouts_active"):
        reader = CaseDataset(ds, path, name)
        reader.future_len = 10_000                                     # the evaluator's rollout convention is ignored
        clips = list(reader)
        assert len(clips) == len(reader) == len(load_cases(path, name)) and ds.future_len == 16
        for case, clip in zip(load_cases(path, name), clips):
            assert clip.positions.shape[1] == 4 + case["horizon"] and clip.source_episode_id == case["id"]
            assert case_numbers(clip)[0] == pytest.approx([case[key] for key in FIELDS[5:]], abs=2e-3)
    active = CaseDataset(ds, path, "windows_motion", tag="q4", stratum="gen")
    assert [case["tag"] for case in active.cases] == ["q4"] and len(list(active)) == 1
    with pytest.raises(ValueError, match="No case"):
        CaseDataset(ds, path, "windows_motion", stratum="fail")
    reader.emit_sidecar = False                                        # switches reach the dataset that builds the batches
    assert ds.emit_sidecar is False and reader.domain == "yikai"
    value = json.loads(path.read_text())
    for damage in (dict(partial=True), dict(domain="folding"), dict(window=dict(value["window"], context_len=2)),
                   dict(inputs=dict(value["inputs"], episode_index_sha256="0" * 64))):
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps({**value, **damage}))
        with pytest.raises(ValueError):
            CaseDataset(ds, broken, "windows")


def test_case_reader_yields_the_frozen_clips_of_pushing_records(tmp_path):
    from data.unified_dataset import UnifiedPushingDataset
    from utils.eval_cases import CaseDataset
    root = _pushing_shards(tmp_path / "test")
    (root / "INDEX.json").write_text(json.dumps(dict(schema="pwm-shard-index-v1", domain="pushing", split="test")))
    ds = UnifiedPushingDataset([str(root)], raw=True, context_len=4, future_len=16, split="all", training=False, shuffle=False)
    ds.emit_sidecar = True
    path = _manifest(tmp_path / "pushing_v1.json", freeze_pushing(ds, Namespace(**ARGS)), ds, "pushing")
    for name, frames in (("windows", 20), ("rollouts_first", 40)):
        cases, clips = load_cases(path, name), list(CaseDataset(ds, path, name))
        assert len(clips) == len(cases) and ds.future_len == 16 and len(ds.shards) == 3
        for case, clip in zip(cases, clips):
            assert clip.positions.shape[:2] == (1, frames) and Path(clip.source_shard).name == case["id"].split(":")[0]
            assert case_numbers(clip)[0] == pytest.approx([case[key] for key in FIELDS[5:]], abs=2e-3)
    (root / "INDEX.json").write_text(json.dumps(dict(schema="pwm-shard-index-v1", domain="pushing", split="test", note="rebuilt")))
    with pytest.raises(ValueError, match="another test folder"):
        CaseDataset(ds, path, "windows")


@pytest.mark.parametrize("rollout", [False, True])
def test_a_model_that_predicts_no_motion_scores_the_frozen_references(release, tmp_path, monkeypatch, rollout):  # noqa: F811
    """End to end through the evaluator: the case reader, the row sets and the scorer reproduce the manifest's numbers."""
    from types import SimpleNamespace
    import inference
    from omegaconf import OmegaConf
    from data.normalization import resolve_normalization
    from scripts.unified import eval_unified as evaluation
    from utils.eval_cases import CaseDataset
    ds = _dataset(release, training=False)
    ds.emit_sidecar = True
    sets = freeze_rl_push(ds, Namespace(**{**ARGS, "rollouts": 0}))
    path = _manifest(tmp_path / "rl_push_v1.json", sets, ds, "rl_push")
    name = "rollouts_first" if rollout else "windows"
    still = lambda model, batch, *a, **k: torch.zeros_like(batch.prediction_x0)
    monkeypatch.setattr(evaluation, "run_diffusion_inference_df", still)
    monkeypatch.setattr(inference, "run_diffusion_inference_df", still)
    cfg = OmegaConf.create({"normalization": resolve_normalization(release["spec"]), "data": {"context_len": 4, "future_len": 16},
                            "flow_matching": {"inference_steps": 4}})
    model = SimpleNamespace(cfg=SimpleNamespace())
    reader = CaseDataset(_dataset(release, training=False), path, name)
    rows = ("predicted", "moving", "still")
    if rollout:
        result = evaluation.rollout_domain(model, None, cfg, "rl_push", None, torch.device("cpu"), horizon=200, future_len=16,
                                           context_len=4, dataset=reader, row_sets=rows)
    else:
        result = evaluation.eval_domain(model, None, cfg, "rl_push", None, torch.device("cpu"), future_len=16, context_len=4,
                                        dataset=reader, row_sets=rows)
    summary = sets[name]["cases"] and summarize(sets[name]["cases"])
    assert result["n_episodes"] == summary["cases"] == 6
    assert result["rows"]["predicted"]["particle_mean_mm"] == pytest.approx(summary["zero_predicted_mm"], abs=5e-3)
    assert result["rows"]["predicted"]["particle_final_mm"] == pytest.approx(summary["zero_predicted_final_mm"], abs=5e-3)
    assert result["rows"]["moving"]["particle_mean_mm"] == pytest.approx(summary["zero_moving_mm"], abs=5e-3)
    assert result["rows"]["moving"]["n_episodes"] == summary["cases_with_moving_rows"]
    assert result["pose_passive_mm"] == pytest.approx(summary["zero_pose_passive_pairs_mm"], abs=5e-3)
    assert result["pose_passive_deg"] == pytest.approx(summary["zero_pose_passive_pairs_deg"], abs=5e-3)
    assert (result["horizon_frames_min"], result["horizon_frames_max"]) == (summary["horizon_min"], summary["horizon_max"])
    if not rollout:                                                    # a window leaves the actuators where they were too
        assert result["particle_mean_mm"] == pytest.approx(summary["zero_all_mm"], abs=5e-3)


@pytest.mark.parametrize("rollout", [False, True])
def test_a_case_set_adds_curves_trivial_references_case_records_and_saved_predictions(release, tmp_path, monkeypatch, rollout):  # noqa: F811
    """What --cases adds to a receipt (eval_unified.CaseExtras), checked with a model that predicts no motion: its numbers
    are the zero reference's, every case's curve averages to the case's frozen number, and the saved prediction is the
    last context frame held."""
    from types import SimpleNamespace
    import inference
    from omegaconf import OmegaConf
    from data.normalization import resolve_normalization
    from scripts.unified import eval_unified as evaluation
    from utils.eval_cases import CaseDataset
    ds = _dataset(release, training=False)
    ds.emit_sidecar = True
    sets = freeze_rl_push(ds, Namespace(**{**ARGS, "rollouts": 0}))
    path = _manifest(tmp_path / "rl_push_v1.json", sets, ds, "rl_push")
    name = "rollouts_first" if rollout else "windows"
    still = lambda model, batch, *a, **k: torch.zeros_like(batch.prediction_x0)
    monkeypatch.setattr(evaluation, "run_diffusion_inference_df", still)
    monkeypatch.setattr(inference, "run_diffusion_inference_df", still)
    cfg = OmegaConf.create({"normalization": resolve_normalization(release["spec"]), "data": {"context_len": 4, "future_len": 16},
                            "flow_matching": {"inference_steps": 4}})
    reader = CaseDataset(_dataset(release, training=False), path, name)
    rows = ("predicted", "moving", "still")
    extras = evaluation.CaseExtras(rows, reader.cases, tmp_path / "predictions")
    score = evaluation.rollout_domain if rollout else evaluation.eval_domain
    result = score(SimpleNamespace(cfg=SimpleNamespace()), None, cfg, "rl_push", None, torch.device("cpu"), future_len=16,
                   context_len=4, dataset=reader, row_sets=rows, case_extras=extras, **({"horizon": 200} if rollout else {}))
    model, zero = result["rows"]["predicted"], result["reference"]["zero"]["rows"]["predicted"]
    summary = summarize(sets[name]["cases"])
    for key in ("particle_mean_mm", "particle_final_mm", "chamfer_mean_mm", "particle_per_frame_mm", "chamfer_per_frame_mm"):
        assert model[key] == pytest.approx(zero[key], abs=5e-3), key   # a model that predicts no motion IS the reference
    assert model["particle_per_frame_n"] == zero["particle_per_frame_n"]
    assert model["particle_mse_mm2"] == pytest.approx(zero["particle_mse_mm2"], rel=1e-3)
    assert zero["particle_mean_mm"] == pytest.approx(summary["zero_predicted_mm"], abs=5e-3)
    assert result["reference"]["hold"]["rows"]["predicted"]["particle_mean_mm"] == pytest.approx(summary["hold_predicted_mm"], abs=5e-3)
    assert result["reference"]["zero"]["pose_passive_mm"] == pytest.approx(summary["zero_pose_passive_pairs_mm"], abs=5e-3)
    assert len(model["particle_per_frame_mm"]) == summary["horizon_max"] and model["particle_per_frame_n"][0] == 6
    assert model["particle_per_frame_n"][-1] == sum(1 for case in reader.cases if case["horizon"] == summary["horizon_max"])
    assert "_per_frame_" not in json.dumps(evaluation._without_curves(result))      # the printed line stays short
    assert [record["id"] for record in extras.records] == [case["id"] for case in reader.cases]
    for k, (record, case) in enumerate(zip(extras.records, reader.cases)):
        assert len(record["predicted_mm"]) == case["horizon"] == len(record["zero_predicted_mm"])
        assert np.mean(record["predicted_mm"]) == pytest.approx(case["zero_predicted_mm"], abs=5e-3)
        assert record["predicted_mm"][-1] == pytest.approx(case["zero_predicted_final_mm"], abs=5e-3)
        assert np.mean(record["hold_predicted_mm"]) == pytest.approx(case["hold_predicted_mm"], abs=5e-3)
        assert (record["moving_mm"] is None) == (case["rows_moving"] == 0)
        with np.load(tmp_path / "predictions" / f"{k:05d}.npz") as saved:
            assert str(saved["id"]) == case["id"] and int(saved["start"]) == case["start"] and int(saved["horizon"]) == case["horizon"]
            assert saved["pred"].shape == (case["horizon"], case["rows_all"], 3) and saved["pred"].dtype == np.float32
            assert int(saved["predicted"].sum()) == case["rows_predicted"] and len(saved["rows"]) == case["rows_all"]
            held = saved["pred"][:, saved["predicted"]]
            assert np.abs(held - held[:1]).max() < 1e-6                # no motion: every frame is the last context frame


FROZEN = {  # a frozen list is never edited: a changed rule or test folder gets a new version file
    "folding_8mm_v1.json": ("8d6539522293a85e055050569f980dd4a69a3cb311bed16db18dd2ba24344514",
                            dict(windows=236, windows_motion=944, rollouts_first=236, rollouts_active=236)),
    "yikai_v1.json": ("d0a44905ad63a2e6b477351b9f9a0b503a1130101f85cc6c187c48fe2e9cee6d",
                      dict(windows=548, windows_motion=2192, rollouts_first=256, rollouts_active=256)),
    "rl_push_v1.json": ("3674b6a9e970c42434b0f680ec653303481bfcc82fbec17ef2e2de0440441643",
                        dict(windows=5534, rollouts_first=256)),
    "dex_v2_v1.json": ("246a16bc438ccfc17a47329797f00d227373a4375b3daa48fca6921441b55941",
                       dict(windows=194, windows_motion=760, rollouts_first=194, rollouts_active=190)),
    "dex_v2_filtered_v1.json": ("6afcc9b8c502c8067e9f88056ec68cd135d24279e3b604f78c60876c997d19c2",
                                dict(windows=190, windows_motion=744, rollouts_first=190, rollouts_active=186)),
    "mpm_fresh_v1.json": ("ac5becd2110b4e739c177a1a91151aab1106ae137fe551b3ecae28eed42784db",
        dict(windows=100, rollouts_first=100)),
    "mpm_fresh_test_seen_v1.json": ("f70c8af1a1879e4bc15691d68b9dda9cfaf63f67a25d98d93316d8c3a5d7f1d5",
        dict(windows=100, rollouts_first=100)),
    "mpm_fresh_test_count_unseen_v1.json": ("4d1b135396db032f0994e5ec6b6181b443f9d795e851d95a918515fc6ac89ed0",
        dict(windows=100, rollouts_first=100)),
    "mpm_fresh_test_count_seen_v1.json": ("8f3332e330b9474cc70e30d762c92c0eb5cb4808c9f6fa56ecaff0b32b1d14a3",
        dict(windows=100, rollouts_first=100)),
    "mpm_drag_v1.json": ("772542710356be53f69fd1394311b9a5e4848f0acde19341f6f8652f707137d9",
        dict(windows=100, rollouts_first=100)),
    "mpm_drag_test_seen_v1.json": ("7c480a007297d75758d7a5a191d8e196c0bef0c49547cbce504db3e66822af1c",
        dict(windows=100, rollouts_first=100)),
    "dex_v2_filtered2_v1.json": ("c0851e88f03b78c1e3a9435ea6f9767cb4dec1560b52982cf13d948809a6e396",
        dict(windows=174, windows_motion=680, rollouts_first=174, rollouts_active=170)),
    "pushing_v1.json": ("34b9833fc8366ae661ebfc55878274abd3ed2849d14afa9b141fb2cfeb32d87e",
                        dict(windows=10148, rollouts_first=256)),
    # 2026-09-19: the small test folders (the v1 lists above describe the complete folders, now `test_full`) and pickup
    "pushing_v2.json": ("0b39060482b1f8f0370ad11a620a7275423e13271ec6be1c8b49f58dbaac5877",
                        dict(windows=100, rollouts_first=100)),
    "rl_push_v2.json": ("472315b83aee760ad794a6050a12d0452f3e9ee42ec84c0121f779a9109c0dd4",
                        dict(windows=200, rollouts_first=200)),
    "yikai_v2.json": ("3193b64f18d4763d9957345874c8618bc2d8f607eaf777970c45aa8827a57d3c",
        dict(windows=200, windows_motion=800, rollouts_first=200, rollouts_active=200)),
    "pickup_v1.json": ("90e6b4e5eb4aaf1e1a9c1013f38aaa0a82c72b0d5cd96481e2165a9c4fedbf40",
                       dict(windows=200, rollouts_first=200)),
    # 2026-09-19: the long recordings also rolled out to their last frame (rollouts_full, frozen at e366c0c). The other sets
    # pin the clips of the previous version; a few stored means moved by 0.0001 mm on the machine of this freeze.
    "yikai_v3.json": ("c0d0223d041910065322dd8c06148aa5305e6cf140b9c1b3495dd0acfbd30eca",
        dict(windows=200, windows_motion=800, rollouts_first=200, rollouts_active=200, rollouts_full=200)),
    "dex_v2_filtered2_v2.json": ("4a62f090da50b3ab842e1c612e871ea8860687114d916ff7207608a91f29460e",
        dict(windows=174, windows_motion=680, rollouts_first=174, rollouts_active=170, rollouts_full=174)),
    "folding_8mm_v2.json": ("ea7809eb2d828ebdfcba9c8d72d1fbd529c6379fe1a138cebb27933c0c6255c2",
        dict(windows=236, windows_motion=944, rollouts_first=236, rollouts_active=236, rollouts_full=236)),
    # 2026-09-20: folding rolls out 50 episodes (a whole clip is 898 frames): the first 50 of v2's rollout cases, ten per fold
    # action, frozen at 71e93a2; the window sets pin v2's clips.
    "folding_8mm_v3.json": ("827cb6b455c1f0234f369f3437dc80512162cae16e2160e8234a08c85d265efc",
        dict(windows=236, windows_motion=944, rollouts_first=50, rollouts_active=50, rollouts_full=50)),
}


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_shipped_manifests_are_the_frozen_ones(name):
    import hashlib
    from pathlib import Path
    path = Path("data/manifests/eval_cases") / name
    digest, sizes = FROZEN[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    manifest = load_cases(path)
    assert not manifest.get("partial") and not manifest["code"]["git_dirty"]
    assert manifest["window"]["context_len"] == 4 and manifest["window"]["future_len"] == 16
    assert {key: len(entry["cases"]) for key, entry in manifest["sets"].items()} == sizes
    for key, entry in manifest["sets"].items():
        assert entry["fields"] == list(FIELDS) and entry["summary"]["cases"] == sizes[key]
        assert all(case[1] >= manifest["window"]["first_start"] and case[2] >= 16 for case in entry["cases"])
    windows = load_cases(path, "windows")
    assert len({case["id"] for case in windows}) == len(windows)          # one window per test episode
    rollouts = {case["id"] for case in load_cases(path, "rollouts_first")}
    assert rollouts <= {case["id"] for case in windows}
    if "rollouts_active" in manifest["sets"]:                          # only episodes long enough to offer a choice of spans
        active = {case["id"] for case in load_cases(path, "rollouts_active")}
        long_episodes = {case["id"] for case in load_cases(path, "windows_motion")}
        assert active == rollouts & long_episodes
    if "rollouts_full" in manifest["sets"]:                            # the same episodes from the same frame, to their end
        first, whole = load_cases(path, "rollouts_first"), load_cases(path, "rollouts_full")
        assert [(c["id"], c["start"]) for c in whole] == [(c["id"], c["start"]) for c in first]
        assert all(w["horizon"] >= f["horizon"] for w, f in zip(whole, first))
