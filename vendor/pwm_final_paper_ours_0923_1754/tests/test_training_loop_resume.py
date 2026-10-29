"""Exercise the actual training loop, including validation and checkpoint timing."""
import copy
from pathlib import Path

import pytest
import torch

from tests.test_train import create_dummy_batch


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
@pytest.mark.parametrize("rejected_step", [None, 2, 3])
def test_particle_loop_resume_preserves_the_next_update_and_validation_state(tmp_path, monkeypatch, architecture, profile, rejected_step):
    import train
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    for name in ("init_wandb", "log_metrics", "finish_wandb"):
        monkeypatch.setattr(train, name, lambda *args, **kwargs: None)
    torch.manual_seed(2468)
    batch = create_dummy_batch(B=1, T=4, N=8, context_len=2)
    calls = []

    def loader(**kwargs):
        calls.append(kwargs)
        return [copy.deepcopy(batch)]

    monkeypatch.setattr(train, "create_dataloader", loader)
    from utils.config import load_config
    from data.normalization import domain_normalizers
    cfg = load_config(f"configs/{architecture}.yaml", [f"normalization.profile={profile}"])
    if architecture == "transformer":
        cfg.model.update(dict(dim=32, depth=1, heads=4, num_latents=8, k_neighbors=4,
                              gradient_checkpointing=False))
    else:
        cfg.model.update(dict(unet_strides="1", unet_widths="16,32", unet_enc_depths="1,1",
            unet_dec_depths="1", unet_head_dim=8, unet_t_dim=32, unet_l1_by_mat="0:4,1:4,2:4",
            unet_pool_k=4, unet_register_from_level=-1, unet_ckpt_from_level=-1,
            unet_ckpt_seams=False, unet_ckpt_ingraph=False))
    cfg.model.dropout, cfg.model.context_len = 0.2, 2
    batch.domain = "mpm"
    batch.object_ids.zero_()
    batch.action_mask.zero_()
    batch.actions.zero_()
    batch.mat_type = torch.zeros_like(batch.object_ids)
    batch.delta_stats, batch.position_stats, _ = domain_normalizers(cfg.normalization, "mpm")
    cfg.data.domains = ["mpm"]
    cfg.data.context_len, cfg.data.future_len = 2, 2
    cfg.data.seed, cfg.data.domain_weights = 77, {"mpm": 1.0}
    accumulation = 1 if rejected_step is None else 2
    cfg.train.update(dict(steps=6, warmup_steps=1, overfit_batches=1, max_batch_size=-1,
                          mixed_precision="no", compile=False, grad_accum=accumulation, ema_warmup=True,
                          stop_after_step=0))
    cfg.log.update(dict(dir=str(tmp_path), every=1, val_every=2, val_batches=1,
                        sampler_batches=0, best_metric="loss", save_every=2,
                        sample_every=0, project=None))
    cfg.run_name = "continuous"
    cfg.resume = None
    production_step, microstep, rejected = train.train_step, 0, []

    def reject_one_microbatch(*args, **kwargs):
        nonlocal microstep
        group = microstep // accumulation
        # Inject a deterministic finite loss spike in the first microbatch.
        # Production accumulation/rejection decides whether to discard the group.
        if group == rejected_step and microstep % accumulation == 0:
            kwargs["loss_ema"], kwargs["loss_spike_threshold"] = 1e-20, 0.0
        microstep += 1
        result = production_step(*args, **kwargs)
        if result[1]["sync_gradients"] and result[1]["skipped"]:
            rejected.append(group)
        return result

    monkeypatch.setattr(train, "train_step", reject_one_microbatch)
    train.train(cfg)
    checkpoint = tmp_path / "continuous/checkpoints/step_2.pt"
    partial = torch.load(checkpoint, weights_only=True)
    assert partial["step"] == 2 and partial["next_step"] == 3
    assert rejected == ([] if rejected_step is None else [rejected_step])
    assert partial["ema_meta"]["updates"] == 3 - int(rejected_step == 2)
    assert partial["scheduler"]["last_epoch"] == partial["ema_meta"]["updates"]
    assert partial["runtime"]["ranks"][0]["loop"]["best_val"] < float("inf")
    cfg.run_name, cfg.resume = "resumed", str(checkpoint)
    microstep = partial["next_step"] * accumulation
    rejected.clear()
    train.train(cfg)
    expected = torch.load(tmp_path / "continuous/checkpoints/final.pt", weights_only=True)
    observed = torch.load(tmp_path / "resumed/checkpoints/final.pt", weights_only=True)
    assert expected["next_step"] == observed["next_step"] == 6
    assert rejected == ([3] if rejected_step == 3 else [])
    assert expected["ema_meta"]["updates"] == 6 - int(rejected_step is not None)
    assert expected["scheduler"]["last_epoch"] == expected["ema_meta"]["updates"]
    for field in ("model", "ema", "optimizer"):
        torch.testing.assert_close(observed[field], expected[field], rtol=0, atol=0)
    assert observed["scheduler"] == expected["scheduler"]
    assert observed["ema_meta"] == expected["ema_meta"]
    assert observed["runtime"]["ranks"][0]["loop"] == expected["runtime"]["ranks"][0]["loop"]
    assert partial["normalization"]["profile"] == profile
    assert expected["normalization"] == observed["normalization"] == partial["normalization"]
    assert all(call["seed"] == 77 and dict(call["domain_weights"]) == {"mpm": 1.0}
               and list(call["domains"]) == ["mpm"]
               and call["normalization"]["sha256"] == partial["normalization"]["sha256"] for call in calls)


@pytest.mark.parametrize("val_every,sample_every,expected_loaders", [(0, 0, 1), (2, 0, 3), (0, 2, 5)])
def test_disabled_diagnostics_do_not_construct_extra_loaders(tmp_path, monkeypatch, val_every, sample_every, expected_loaders):
    import train
    from utils.config import load_config
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    for name in ("init_wandb", "log_metrics", "finish_wandb"):
        monkeypatch.setattr(train, name, lambda *args, **kwargs: None)
    batch = create_dummy_batch(B=1, T=4, N=8, context_len=2)
    calls = []
    def loader(**kwargs):
        calls.append(kwargs)
        return [copy.deepcopy(batch)]
    monkeypatch.setattr(train, "create_dataloader", loader)
    cfg = load_config("configs/transformer.yaml")
    cfg.model.update(dict(dim=32, depth=1, heads=4, num_latents=8, k_neighbors=4,
                         context_len=2, gradient_checkpointing=False))
    cfg.data.context_len, cfg.data.future_len, cfg.data.domains = 2, 2, ["mpm"]
    cfg.train.update(dict(steps=1, warmup_steps=0, mixed_precision="no", compile=False))
    cfg.log.update(dict(dir=str(tmp_path), project=None, val_every=val_every, sampler_batches=0,
                        sample_every=sample_every, save_every=0, save_best=False, save_final=False))
    train.train(cfg)
    assert len(calls) == expected_loaders
    assert all(list(call["domains"]) == ["mpm"] for call in calls)
    assert all(call["emit_sidecar"] and call["bucket_max_items"] == 64 for call in calls)
    assert all(call["normalization"]["sha256"] == cfg.normalization.sha256 for call in calls)
    if sample_every:
        assert all(call["no_shard_split"] and call["num_workers"] == 0 for call in calls[-3:])


def _tiny_unet_cfg(tmp_path, run_name, resume=None, steps=4, stop_after=0, train_overrides=None,
                   **log_overrides):
    """The smallest real UNet recipe the loop will run, on one cached batch."""
    from utils.config import load_config
    cfg = load_config("configs/unet.yaml")
    cfg.model.update(dict(unet_strides="1", unet_widths="16,32", unet_enc_depths="1,1",
        unet_dec_depths="1", unet_head_dim=8, unet_t_dim=32, unet_l1_by_mat="0:4,1:4,2:4",
        unet_pool_k=4, unet_register_from_level=-1, unet_ckpt_from_level=-1,
        unet_ckpt_seams=False, unet_ckpt_ingraph=False, dropout=0.0, context_len=2))
    cfg.data.context_len, cfg.data.future_len, cfg.data.domains = 2, 2, ["mpm"]
    cfg.data.domain_weights = {"mpm": 1.0}
    cfg.train.update(dict(steps=steps, warmup_steps=1, overfit_batches=1, mixed_precision="no",
                          compile=False, grad_accum=1, stop_after_step=stop_after))
    cfg.train.update(train_overrides or {})
    cfg.log.update(dict(dir=str(tmp_path), every=1, val_every=1, val_batches=1,
                        sampler_batches=1, save_every=0, sample_every=0, project=None,
                        **log_overrides))
    cfg.run_name, cfg.resume = run_name, resume
    return cfg


def _scripted_loop(tmp_path, monkeypatch, sample_mm, run_name="gated", resume=None,
                   steps=4, stop_after=0, train_overrides=None, **log_overrides):
    """Run the loop with validate/validate_sampler scripted; return (cfg, saved names).

    The scripted val loss falls at EVERY validation while sample_mm does not, so
    the two gates are distinguishable by how often best.pt is written.
    """
    import train
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    for name in ("init_wandb", "log_metrics", "finish_wandb"):
        monkeypatch.setattr(train, name, lambda *args, **kwargs: None)
    torch.manual_seed(1357)
    batch = create_dummy_batch(B=1, T=4, N=8, context_len=2)
    batch.domain = "mpm"
    batch.object_ids.zero_()
    batch.action_mask.zero_()
    batch.actions.zero_()
    batch.mat_type = torch.zeros_like(batch.object_ids)
    cfg = _tiny_unet_cfg(tmp_path, run_name, resume=resume, steps=steps,
                         stop_after=stop_after, train_overrides=train_overrides, **log_overrides)
    from data.normalization import domain_normalizers
    batch.delta_stats, batch.position_stats, _ = domain_normalizers(cfg.normalization, "mpm")
    monkeypatch.setattr(train, "create_dataloader", lambda **kwargs: [copy.deepcopy(batch)])
    losses = iter([9.0, 9.0, 8.0, 8.0, 7.0, 7.0] * 4)      # raw and ema per validation
    monkeypatch.setattr(train, "validate", lambda **kwargs: {"loss": next(losses)})
    samples = iter(list(sample_mm) * 4)
    monkeypatch.setattr(train, "validate_sampler", lambda **kwargs: {"sample_mm": next(samples)})
    saved = []
    production = train.save_checkpoint

    def record(path, *args, **kwargs):
        saved.append(Path(path).name)
        return production(path, *args, **kwargs)

    monkeypatch.setattr(train, "save_checkpoint", record)
    train.train(cfg)
    return cfg, saved


def _loop_state(tmp_path, run_name):
    state = torch.load(tmp_path / f"{run_name}/checkpoints/final.pt", weights_only=True)
    return state["runtime"]["ranks"][0]["loop"]


def test_best_checkpoint_follows_the_full_sampler_error_not_the_one_pass_loss(tmp_path, monkeypatch):
    """log.best_metric=sample_mm (the default) crowns best.pt on generation quality."""
    _, saved = _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0])
    # Validations at steps 1, 2, 3: sample_mm improves at 1 and 2 but not at 3,
    # while the scripted val loss improves at every one of them.
    assert saved.count("best.pt") == 2
    loop = _loop_state(tmp_path, "gated")
    assert loop["best_metric"] == "sample_mm"
    assert loop["best_val"] == pytest.approx(3.0)


def test_legacy_loss_gate_remains_available_and_records_its_metric(tmp_path, monkeypatch):
    _, saved = _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0],
                              run_name="legacy", best_metric="loss")
    assert saved.count("best.pt") == 3
    loop = _loop_state(tmp_path, "legacy")
    assert loop["best_metric"] == "loss" and loop["best_val"] == pytest.approx(7.0)


def test_changing_the_gate_metric_restarts_the_best_search_on_resume(tmp_path, monkeypatch):
    """A stored best_val in millimetres must never be compared against a loss."""
    _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="mm", steps=8, stop_after=3)
    checkpoint = tmp_path / "mm/checkpoints/final.pt"
    assert _loop_state(tmp_path, "mm")["best_val"] == pytest.approx(3.0)
    _, saved = _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="regated",
                              resume=str(checkpoint), steps=8, best_metric="loss")
    # 3.0 mm carried in the checkpoint does not gate a loss of 9.0.
    assert saved.count("best.pt") >= 1
    assert _loop_state(tmp_path, "regated")["best_metric"] == "loss"


def test_same_gate_metric_keeps_the_running_best_across_resume(tmp_path, monkeypatch):
    _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="keep", steps=8, stop_after=3)
    checkpoint = tmp_path / "keep/checkpoints/final.pt"
    # Resuming with the same metric must NOT accept a worse sampler error as best.
    _, saved = _scripted_loop(tmp_path, monkeypatch, [9.0, 9.0, 9.0], run_name="kept",
                              resume=str(checkpoint), steps=8)
    assert saved.count("best.pt") == 0
    assert _loop_state(tmp_path, "kept")["best_val"] == pytest.approx(3.0)


def test_checkpoints_record_exclusion_contents_and_resume_refuses_an_edit(tmp_path, monkeypatch):
    """Every checkpoint stores the exclusion-file hashes the run started with; resuming after
    one of those files changed in place is refused before any state is loaded."""
    import json
    import data.unified_dataset as unified
    from utils.checkpoint import corpus_identity
    exclusions = tmp_path / "mpm_blocklist_train.json"
    exclusions.write_text(json.dumps({"blocklist": []}))
    monkeypatch.setattr(unified, "MPM_BLOCKLISTS", [str(exclusions)])
    cfg, _ = _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="hashed", steps=8, stop_after=3)
    checkpoint = tmp_path / "hashed/checkpoints/final.pt"
    corpus = torch.load(checkpoint, weights_only=True)["corpus"]
    assert corpus == corpus_identity(cfg) and str(exclusions.resolve()) in corpus["train"]["mpm"]
    exclusions.write_text(json.dumps({"blocklist": [{"shard": "00000.tar", "key": "k0"}]}))
    with pytest.raises(ValueError, match="corpus files differ"):
        _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="refused",
                       resume=str(checkpoint), steps=8)


def test_init_from_starts_a_new_run_from_another_runs_weights_and_resume_supersedes_it(tmp_path, monkeypatch):
    """train.init_from loads the source's EMA weights before the new EMA and optimizer exist
    (fresh schedule, step 0); on a resubmission the resumed state wins, not the source."""
    import train
    _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="source")
    source = tmp_path / "source/checkpoints/final.pt"
    source_ema = torch.load(source, weights_only=True)["ema"]
    starts = []

    class RecordingEMA(train.EMA):
        def __init__(self, model, *args, **kwargs):
            starts.append({n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad})
            super().__init__(model, *args, **kwargs)

    monkeypatch.setattr(train, "EMA", RecordingEMA)
    init = {"init_from": str(source), "action_dropout": 1.0}
    _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="tuned", steps=8, stop_after=3,
                   train_overrides=init)
    assert starts[-1] and set(starts[-1]) <= set(source_ema)
    for name, value in starts[-1].items():
        torch.testing.assert_close(value, source_ema[name], rtol=0, atol=0)
    tuned = torch.load(tmp_path / "tuned/checkpoints/final.pt", weights_only=True)
    assert any(not torch.equal(tuned["model"][n], source_ema[n]) for n in starts[-1])   # it trained
    _scripted_loop(tmp_path, monkeypatch, [5.0, 3.0, 4.0], run_name="resumed", steps=8,
                   resume=str(tmp_path / "tuned/checkpoints/final.pt"), train_overrides=init)
    for name, value in starts[-1].items():
        torch.testing.assert_close(value, tuned["model"][name], rtol=0, atol=0)
