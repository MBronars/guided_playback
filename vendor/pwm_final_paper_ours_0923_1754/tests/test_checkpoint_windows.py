"""Inference CLIs use checkpoint windows through sampling and JSON receipts.

Synthetic native episodes and oracle predictions isolate window selection from
checkpoint quality. The evaluator and both rollout drivers execute normally.
"""
import json
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

import inference
from data import unified_dataset
from data.native_episodes import NativeEpisodeDataset
from data.normalization import resolve_normalization
from diffusion.flow_matching import TrajectoryFlowMatching
from guidance import run_guidance
from guidance.plan import plan_from_batch
from guidance.planners import PlanResult, window_nfe
from scripts.dev import native_eval_unified, native_zero_shot
from scripts.unified import eval_unified
from tests.test_native_episodes import _synthetic_episode


WINDOWS = pytest.mark.parametrize("saved,expected", [
    (16, 16), (38, 38), (8, 8), (None, 38), (-1, 38), (0, 38), ("missing", 38),
], ids=["f16", "f38", "f8", "null", "per-domain", "zero", "missing"])


@pytest.fixture(scope="module")
def episode_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("window_episode") / "trajectory.npz"
    _synthetic_episode(path, T=52)
    return path


def _checkpoint(monkeypatch, tmp_path, saved):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"model and configuration supplied by the test loader")
    data = dict(context_len=4)
    if saved != "missing":
        data["future_len"] = saved
    # a real checkpoint always carries its train block; the guidance CLI reads train.action_dropout from it
    cfg = OmegaConf.create(dict(data=data, normalization=resolve_normalization(), train=dict(action_dropout=0.0),
                               flow_matching=dict(inference_steps=2)))
    model = torch.nn.Identity()
    model.cfg = SimpleNamespace(pose_native=True, articulated_bodies=False)
    diffusion = TrajectoryFlowMatching()
    loader = lambda *args: (model, diffusion, cfg, 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for module in (inference, native_eval_unified, native_zero_shot):
        monkeypatch.setattr(module, "load_model_and_diffusion", loader)
    return path, cfg


def _dataset(path, cfg, context_len, future_len):
    dataset = NativeEpisodeDataset([path], context_len=context_len, future_len=future_len,
                                    windows_per_episode=1, normalization=cfg.normalization)
    # Keep enough recorded frames for the same 48-frame rollout in every case.
    dataset.items = [(0, 0)]
    return dataset


def _prediction(batch):
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    poses = batch.body_poses[:, batch.context_len:, ridx]
    return dict(positions=batch.future, R=poses[..., :3, :3], t=poses[..., :3, 3]), ridx


@WINDOWS
@pytest.mark.parametrize("rollout", [False, True], ids=["window", "rollout"])
def test_native_evaluator_checkpoint_window_and_receipt(monkeypatch, tmp_path, episode_path,
                                                       saved, expected, rollout):
    checkpoint, cfg = _checkpoint(monkeypatch, tmp_path, saved)
    datasets, samples = [], []
    def dataset(paths, **kwargs):
        datasets.append((kwargs["context_len"], kwargs["future_len"]))
        return _dataset(paths[0], cfg, *datasets[-1])
    def sampler(model, batch, *args, **kwargs):
        samples.append(batch.num_future_frames)
        rigid = batch.body_type[0] == 1
        return (batch.delta_stats.normalize(batch.future_deltas, clip=False),
                batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, rigid], clip=False))
    monkeypatch.setattr(native_eval_unified, "NativeEpisodeDataset", dataset)
    monkeypatch.setattr(eval_unified, "run_diffusion_inference_df", sampler)    # fixed windows
    monkeypatch.setattr(inference, "run_diffusion_inference_df", sampler)       # rollout windows
    out = tmp_path / "evaluation.json"
    monkeypatch.setattr(sys, "argv", ["native_eval_unified", "--checkpoint", str(checkpoint),
        "--episodes", str(episode_path), "--out", str(out), "--windows_per_episode", "1"]
        + (["--rollout", "--rollout_frames", "48"] if rollout else []))
    native_eval_unified.main()
    receipt = json.loads(out.read_text())
    assert datasets == [(4, expected)]
    lengths = [min(expected, 48 - start) for start in range(0, 48, expected)]
    assert samples == (lengths if rollout else [expected])
    assert receipt["context_len"] == 4 and receipt["future_len"] == expected
    if rollout:
        assert receipt["rollout_frames_requested"] == 48


@WINDOWS
def test_native_zero_shot_checkpoint_windows_include_pickup_control(monkeypatch, tmp_path,
                                                                  episode_path, saved, expected):
    checkpoint, cfg = _checkpoint(monkeypatch, tmp_path, saved)
    samples, controls = [], []
    def sampler(model, diffusion, batch, *args):
        samples.append(batch.num_future_frames)
        pred, ridx = _prediction(batch)
        return pred["positions"], pred["R"], pred["t"], ridx
    def control(config, domain, context_len, future_len):
        assert domain == "pickup" and config is cfg
        controls.append((context_len, future_len))
        return _dataset(episode_path, cfg, *controls[-1])
    def rollout_sampler(model, batch, *args, **kwargs):
        samples.append(batch.num_future_frames)
        rigid = batch.body_type[0] == 1
        return (batch.delta_stats.normalize(batch.future_deltas, clip=False),
                batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, rigid], clip=False))
    monkeypatch.setattr(native_zero_shot, "sample_window", sampler)
    monkeypatch.setattr(inference, "run_diffusion_inference_df", rollout_sampler)
    monkeypatch.setattr(native_zero_shot, "select_windows", lambda *args: [0])
    monkeypatch.setattr(native_zero_shot, "render", lambda *args: None)
    monkeypatch.setattr(native_zero_shot, "evaluation_dataset", control)
    monkeypatch.setattr(sys, "argv", ["native_zero_shot", "--checkpoint", str(checkpoint),
        "--episodes", str(episode_path), "--out", str(tmp_path), "--layouts", "pickup",
        "--windows", "1", "--rollout_frames", "48", "--pickup_control_batches", "1"])
    native_zero_shot.main()
    receipt = json.loads((tmp_path / "zero_shot.json").read_text())
    lengths = [min(expected, 48 - start) for start in range(0, 48, expected)]
    assert samples == [expected, *lengths, expected]
    assert controls == [(4, expected)]
    assert receipt["context_len"] == 4 and receipt["future_len"] == expected
    assert next(iter(receipt["episodes"].values()))["rollout"]["frames"] == 48
    assert len(receipt["pickup_control"]) == 1


@pytest.mark.parametrize("dropout,hidden", [(0.0, False), (1.0, True)], ids=["given", "hidden"])
def test_native_zero_shot_follows_the_checkpoint_action_policy(monkeypatch, tmp_path, episode_path,
                                                               dropout, hidden):
    """A checkpoint trained with every action hidden (train.action_dropout=1) is sampled without
    them in the native window, every rollout window and the pickup control (ISSUES.md ISSUE-004)."""
    checkpoint, cfg = _checkpoint(monkeypatch, tmp_path, 16)
    cfg.train = {"action_dropout": dropout}
    given = []
    def sampler(model, batch, *args, **kwargs):
        given.append(bool(batch.pose_action_mask.any()))
        rigid = batch.body_type[0] == 1
        return (batch.delta_stats.normalize(batch.future_deltas, clip=False),
                batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, rigid], clip=False))
    monkeypatch.setattr(native_zero_shot, "run_diffusion_inference_df", sampler)   # windows and control
    monkeypatch.setattr(inference, "run_diffusion_inference_df", sampler)          # rollout windows
    monkeypatch.setattr(native_zero_shot, "select_windows", lambda *args: [0])
    monkeypatch.setattr(native_zero_shot, "render", lambda *args: None)
    monkeypatch.setattr(native_zero_shot, "evaluation_dataset",
                        lambda config, domain, context_len, future_len: _dataset(episode_path, cfg, context_len, future_len))
    monkeypatch.setattr(sys, "argv", ["native_zero_shot", "--checkpoint", str(checkpoint),
        "--episodes", str(episode_path), "--out", str(tmp_path), "--layouts", "pickup",
        "--windows", "1", "--rollout_frames", "48", "--pickup_control_batches", "1"])
    native_zero_shot.main()
    receipt = json.loads((tmp_path / "zero_shot.json").read_text())
    assert given == [not hidden] * 5            # native window, three rollout windows, pickup control
    assert receipt["actions"] == ("hidden" if hidden else "given")


@WINDOWS
@pytest.mark.parametrize("domain", ["pickup", "pushing"])
@pytest.mark.parametrize("horizon", [None, 5, 48], ids=["default-horizon", "short-horizon", "long-horizon"])
def test_guidance_checkpoint_window_horizon_and_cost_receipt(monkeypatch, tmp_path, episode_path,
                                                           saved, expected, domain, horizon):
    checkpoint, cfg = _checkpoint(monkeypatch, tmp_path, saved)
    if saved in (None, -1, 0, "missing"):
        expected = unified_dataset.DOMAIN_FUTURE_LEN[domain]
    datasets, samples, costs = [], [], []
    def dataset(config, selected_domain, context_len, future_len):
        assert selected_domain == domain
        datasets.append((context_len, future_len))
        return _dataset(episode_path, config, context_len, future_len)
    def planner(model, diffusion, batch, objective, **kwargs):
        samples.append(batch.num_future_frames)
        nfe = window_nfe(diffusion, batch, kwargs["steps"], kwargs["schedule"], kwargs["uncertainty_scale"])
        costs.append(nfe)
        pred, _ = _prediction(batch)
        return PlanResult(plan_from_batch(batch), (), pred, 0.0, {}, nfe)
    monkeypatch.setattr(eval_unified, "evaluation_dataset", dataset)
    monkeypatch.setitem(run_guidance.planners.PLANNERS, "best_of_k", planner)
    out = tmp_path / "guidance.json"
    monkeypatch.setattr(sys, "argv", ["guidance", "--checkpoint", str(checkpoint), "--domain", domain,
        "--out", str(out), "--max_batches", "1", "--phantom_weight", "0"]
        + (["--horizon", str(horizon)] if horizon is not None else []))
    run_guidance.main()
    receipt = json.loads(out.read_text())
    requested_horizon = expected if horizon is None else horizon
    assert datasets == [(4, expected)]
    lengths = [min(expected, requested_horizon - start)
               for start in range(0, requested_horizon, expected)]
    assert samples == lengths
    assert receipt["context_len"] == 4 and receipt["future_len"] == expected
    assert receipt["horizon"] == receipt["stride"] == requested_horizon
    # With two base pyramid steps, a full F-frame window takes F + 1 calls.
    # This nominal cost remains distinct from the actual shorter-horizon cost.
    assert receipt["nfe_per_window"] == expected + 1
    assert costs == [length + 1 for length in lengths]
    assert receipt["summary"]["nfe_total"] == sum(costs)
    assert receipt["episodes"][0]["windows"] == len(samples)
