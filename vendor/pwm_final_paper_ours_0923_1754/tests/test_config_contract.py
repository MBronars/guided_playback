"""Strict recipes: bad overrides fail before normalization or corpus access."""
import pytest

from utils.config import load_config, resolve_config


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
@pytest.mark.parametrize("domains", ["[mpm]", "[pickup]", "[cloth]", "[pushing]", "[cloth,mpm]",
                                    "[mpm,pickup,cloth,pushing]"])
def test_primary_choices_resolve_without_opening_any_corpus(monkeypatch, architecture, profile, domains):
    from data import unified_dataset
    monkeypatch.setattr(unified_dataset, "build_domain_dataset", lambda *a, **k: pytest.fail("unexpected corpus I/O"))
    cfg = load_config(f"configs/{architecture}.yaml", [f"normalization.profile={profile}", f"data.domains={domains}"])
    assert cfg.model.architecture == architecture
    assert cfg.normalization.profile == profile
    assert len(cfg.normalization.sha256) == 64
    assert cfg.model.context_len == cfg.data.context_len == 4
    assert resolve_config(cfg) == cfg


@pytest.mark.parametrize("override", [
    "trani.lr=0.01", "train.leanring_rate=0.01", "log.val_baches=4", "data.typo=false",
    "model.num_lantents=8", "flow_matching.inference_step=10", "train.aux_losses.typo=1",
    "device=cuda", "data.stats_location=unified://xy_p99", "model.fps_random_start=true",
    "train.lambda_rigid_rot=0", "train.supervised_points=0", "model.per_object_noise=false",
])
def test_unknown_and_retired_keys_fail_before_normalization(monkeypatch, override):
    from data import normalization
    monkeypatch.setattr(normalization, "resolve_normalization", lambda *a, **k: pytest.fail("unexpected stats access"))
    with pytest.raises(ValueError, match="Unknown|retired"):
        load_config("configs/transformer.yaml", [override])


@pytest.mark.parametrize("overrides", [
    ["seed=-1"], ["data.context_len=1"], ["data.future_len=0"], ["data.num_workers=-1"],
    ["data.batch_size=0"], ["data.batch_size=1.5"], ["data.batch_size=true"],
    ["data.num_collate=2"], ["data.max_objects=3"], ["data.domains=[]"], ["data.domains=[drag]"],
    ["data.domains=[mpm]", "data.domain_weights.mpm=0"], ["data.eval_domains=[]"],
    ["train.steps=0"], ["train.steps=10"], ["train.grad_accum=0"], ["train.lr=-0.1"],
    ["train.weight_decay=-1"], ["train.ema_decay=1"], ["train.ema_decay=.nan"],
    ["train.max_batch_size=0"], ["train.overfit_batches=0"], ["train.mixed_precision=fp32"],
    ["train.adam_betas=[0.9,1]"], ["train.adam_betas=[0.9]"], ["train.adam_eps=0"],
    ["train.stop_after_step=200000"], ["train.loss_spike_threshold=0"],
    ["log.every=0"], ["log.val_batches=0"], ["log.best_gate=sampler"],
    ["diffusion_forcing=1"], ["df_schedule=typo"], ["df_uncertainty_scale=0"],
    ["flow_matching.num_train_timesteps=1"], ["flow_matching.shift=3"],
    ["flow_matching.prediction_type=epsilon"], ["flow_matching.time_sampler=typo"],
    ["flow_matching.beta_alpha=2"], ["flow_matching.time_sampler=beta", "flow_matching.beta_alpha=0"],
    ["flow_matching.time_sampler=beta", "flow_matching.beta_t_min=1"],
    ["train.compile_regional=true"], ["train.compile_mode=reduce-overhead"],
    ["train.rot_aug=true"], ["train.ddp_static_graph=true"], ["train.lambda_mesh=1"],
    ["train.aux_losses.floor=1"], ["train.aux_losses.floor=1", "train.aux_losses.floor_z=0.091"],
    ["train.aux_losses.accel=1", "data.future_len=1"], ["train.aux_losses.vel=1"],
    ["train.aux_losses.pos_cum=-1"], ["train.aux_losses=[]"], ["train.light_pose_noise_std=-0.1"],
    ["train.light_pose_noise_anchor=true"],          # the flag alone would be a silent no-op
    ["model.dim=33"], ["model.dropout=1"],
    ["model.gradient_checkpointing=notreally"], ["model.num_mat_types=3"],
    ["model.unet_actions=false"], ["run_name=../escape"], ["model.architecture=classic"],
])
def test_invalid_contracts_fail_before_stats(monkeypatch, overrides):
    from data import normalization
    monkeypatch.setattr(normalization, "resolve_normalization", lambda *a, **k: pytest.fail("unexpected stats access"))
    with pytest.raises(ValueError):
        load_config("configs/transformer.yaml", overrides)


@pytest.mark.parametrize("overrides", [
    ["model.dim=32"], ["model.unet_widths=127,256,512,1024"], ["model.unet_strides=4,0,2"],
    ["model.unet_widths=[128,256,512,1024]"], ["model.unet_enc_depths=1,1"],
    ["model.unet_dec_depths=1,1,1"], ["model.unet_pool_mode=typo"], ["model.unet_geo_r_m=0"],
    ["model.unet_l1_by_mat=0:256"], ["model.unet_register_from_level=0"],
])
def test_invalid_unet_geometry_and_policy_fail_before_stats(monkeypatch, overrides):
    from data import normalization
    monkeypatch.setattr(normalization, "resolve_normalization", lambda *a, **k: pytest.fail("unexpected stats access"))
    with pytest.raises(ValueError):
        load_config("configs/unet.yaml", overrides)


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("overrides", [
    ["train.noise_context=true"], ["train.noise_context=true", "train.light_context_noise_std=0.1"],
    ["train.supervise_context=true"],
])
def test_noisy_context_is_refused_for_new_runs(monkeypatch, architecture, overrides):
    """Independent per-particle history noise put rigid bodies' particles away from their pose
    (ISSUES.md ISSUE-001). Checkpoints trained with it still load; new runs cannot enable it."""
    from data import normalization
    monkeypatch.setattr(normalization, "resolve_normalization", lambda *a, **k: pytest.fail("unexpected stats access"))
    with pytest.raises(ValueError, match="is retired"):
        load_config(f"configs/{architecture}.yaml", overrides)


def test_meaningful_optional_ablations_remain_supported():
    cfg = load_config("configs/transformer.yaml", ["data.domains=[mpm]", "train.rot_aug=true",
        "train.ddp_static_graph=true", "train.noise_context=false", "train.clean_ctx_timesteps=true",
        "flow_matching.time_sampler=beta", "flow_matching.beta_alpha=1.5",
        "train.adam_betas=[0.85,0.95]", "train.lr_schedule=constant"])
    assert cfg.train.rot_aug and cfg.train.clean_ctx_timesteps
    pickup = load_config("configs/unet.yaml", ["data.domains=[pickup]", "train.light_pose_noise_std=0.05"])
    assert pickup.train.light_pose_noise_std == 0.05
    restore = load_config("configs/unet.yaml", ["data.domains=[pickup]", "train.light_pose_noise_std=0.03",
                                              "train.light_pose_noise_anchor=true", "train.clip_pose_deltas=false"])
    assert restore.train.light_pose_noise_anchor and restore.train.light_pose_noise_std == 0.03
    cloth = load_config("configs/transformer.yaml", ["data.domains=[cloth]", "train.lambda_mesh=1",
        "train.aux_losses.floor=1", "train.aux_losses.floor_z=0"])
    assert cloth.train.aux_losses.floor_z == 0


def test_checkpoint_aliases_are_not_fresh_recipe_aliases():
    from models.factory import parse_model_config
    for alias, expected in (("classic", "transformer"), ("unet_std", "unet")):
        assert parse_model_config({"architecture": alias}, checkpoint=True)[0] == expected
        with pytest.raises(ValueError, match="architecture"):
            load_config("configs/transformer.yaml", [f"model.architecture={alias}"])


def test_historical_primary_numerics_and_budget_confound_remain_explicit():
    tf, unet = load_config("configs/transformer.yaml"), load_config("configs/unet.yaml")
    for cfg in (tf, unet):
        assert (cfg.data.context_len, cfg.data.future_len, cfg.data.batch_size, cfg.seed) == (4, 16, 4, 42)
        assert (cfg.train.steps, cfg.train.warmup_steps, cfg.train.lr, cfg.train.weight_decay) == (200000, 5000, 1e-4, .001)
        assert (cfg.train.grad_clip, cfg.train.ema_decay, cfg.train.lambda_pose_native) == (1, .99, 1)
        assert (cfg.flow_matching.num_train_timesteps, cfg.flow_matching.inference_steps) == (1000, 25)
    assert (tf.model.dim, tf.model.depth, tf.model.heads, tf.model.num_latents) == (768, 15, 12, 512)
    assert (tf.train.stop_after_step, unet.train.stop_after_step) == (0, 40000)
    assert tf.train.compile is False and unet.train.compile is True


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("term", ["strain_inc", "strain_cum"])
@pytest.mark.parametrize("selection", [
    ["data.domains=[cloth]"], ["data.domains=[mpm,cloth]"],
    ["data.domains=[pickup]"], ["data.domains=[pushing]"],
    ["data.domains=[mpm]", "data.eval_domains=[cloth]"],
    ["data.domains=[mpm]", "data.eval_domains=[pickup]"],
    ["data.domains=[mpm]", "data.eval_domains=[mpm,pushing]"],
])
def test_strain_layout_requires_mpm_only_before_stats(monkeypatch, architecture, term, selection):
    from data import normalization
    monkeypatch.setattr(normalization, "resolve_normalization", lambda *a, **k: pytest.fail("unexpected stats access"))
    with pytest.raises(ValueError, match="Strain auxiliaries require MPM-only"):
        load_config(f"configs/{architecture}.yaml", [*selection, f"train.aux_losses.{term}=1"])


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("term", ["strain_inc", "strain_cum"])
@pytest.mark.parametrize("selection", [
    ["data.domains=[mpm]"],
    ["data.domains=[mpm]", "data.eval_domains=[mpm]"],
    ["data.domains=[mpm,cloth]", "data.domain_weights={mpm: 1, cloth: 0}"],
])
def test_mpm_strain_ablation_is_valid(architecture, term, selection):
    cfg = load_config(f"configs/{architecture}.yaml", [*selection, f"train.aux_losses.{term}=1"])
    assert cfg.train.aux_losses[term] == 1
