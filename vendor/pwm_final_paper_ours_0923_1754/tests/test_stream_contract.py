"""Weighted interleave must reject bad recipes before I/O and cannot spin empty."""
import pytest
import torch

from data import unified_dataset as unified


@pytest.mark.parametrize("weights", [{}, {"mpm": 0}, {"mpm": -1},
    {"mpm": float("nan")}, {"mpm": float("inf")}, {"mpm": 1, "typo": 1}])
def test_invalid_domain_weights_fail_before_data_access(monkeypatch, weights):
    def no_io(*args, **kwargs):
        pytest.fail("Invalid domain weights must be rejected before loading any corpus")

    monkeypatch.setattr(unified, "build_domain_dataset", no_io)
    with pytest.raises(ValueError, match="(weight|domain)"):
        unified.create_unified_dataloader(domains=("mpm",), weights=weights, num_workers=0)


def test_zero_weight_domain_is_not_opened(monkeypatch):
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            from types import SimpleNamespace
            yield SimpleNamespace()

    opened = []

    def build(domain, **kwargs):
        opened.append(domain)
        return Stream()

    monkeypatch.setattr(unified, "build_domain_dataset", build)
    stream = unified.create_unified_dataloader(domains=("mpm", "cloth"),
        weights={"mpm": 1, "cloth": 0}, num_workers=0)
    assert next(stream).domain == "mpm"
    assert opened == ["mpm"]


@pytest.mark.parametrize("domains", [[], ["mpm", "mpm"], ["drag"], "mpm"])
@pytest.mark.parametrize("training", [True, False])
def test_bad_domain_selection_fails_before_assets(monkeypatch, domains, training):
    from data.dataset import create_dataloader
    monkeypatch.setattr(unified, "build_domain_dataset", lambda *a, **k: pytest.fail("unexpected corpus access"))
    with pytest.raises(ValueError, match="domain"):
        create_dataloader("unified://v1", format="unified", domains=domains, training=training)


@pytest.mark.parametrize("domains", [["mpm"], ["pickup"], ["cloth"], ["pushing"],
                                      ["cloth", "mpm"], list(unified.DOMAIN_ORDER)])
def test_validation_opens_only_selected_domains(monkeypatch, domains):
    from data.dataset import create_dataloader
    class Empty(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())
    opened = []
    def build(domain, **kwargs):
        opened.append(domain)
        return Empty()
    monkeypatch.setattr(unified, "build_domain_dataset", build)
    loader = create_dataloader("unified://v1", format="unified", domains=domains, training=False)
    expected = tuple(d for d in unified.DOMAIN_ORDER if d in domains)
    assert tuple(opened) == expected and loader.DOMAINS == expected


@pytest.mark.parametrize("override", [None, ["pickup", "pushing"]])
def test_zero_weights_also_disable_default_validation(monkeypatch, override):
    from data.dataset import create_dataloader
    class Empty(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())
    opened = []
    def build(domain, **kwargs):
        opened.append(domain)
        return Empty()
    monkeypatch.setattr(unified, "build_domain_dataset", build)
    create_dataloader("unified://v1", format="unified", domains=["mpm", "cloth"],
        domain_weights={"mpm": 1, "cloth": 0}, eval_domains=override, training=False)
    assert opened == (["mpm"] if override is None else override)


def test_rank_domain_draw_is_independent_of_input_order(monkeypatch):
    from itertools import islice
    from types import SimpleNamespace
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            for _ in range(100):
                yield SimpleNamespace()
    monkeypatch.setattr(unified, "build_domain_dataset", lambda *a, **k: Stream())
    def sequence(domains):
        return [b.domain for b in islice(unified.create_unified_dataloader(
            domains=domains, weights={"mpm": 1, "pushing": 3, "cloth": 2},
            seed=31, num_workers=0), 60)]
    assert sequence(["mpm", "cloth", "pushing"]) == sequence(["pushing", "mpm", "cloth"])


def test_empty_training_stream_fails_after_one_restart(monkeypatch):
    class Empty(torch.utils.data.IterableDataset):
        calls = 0

        def __iter__(self):
            self.calls += 1
            if self.calls > 2:
                pytest.fail("Empty training loader was restarted indefinitely")
            return iter(())

    dataset = Empty()
    monkeypatch.setattr(unified, "build_domain_dataset", lambda *args, **kwargs: dataset)
    stream = unified.create_unified_dataloader(domains=("mpm",), num_workers=0)
    with pytest.raises(ValueError, match="Empty training stream"):
        next(stream)
    assert dataset.calls == 2


def test_finite_loader_repeats_without_caching_or_reordering():
    from itertools import islice
    from utils.iteration import repeat_nonempty

    assert list(islice(repeat_nonempty([1, 2, 3]), 7)) == [1, 2, 3, 1, 2, 3, 1]


@pytest.mark.parametrize("one_shot", [False, True])
def test_exhausted_loader_fails_instead_of_busy_spinning(one_shot):
    from utils.iteration import repeat_nonempty

    stream = repeat_nonempty(iter([42]) if one_shot else [])
    if one_shot:
        assert next(stream) == 42
    with pytest.raises(ValueError, match="Empty training loader"):
        next(stream)


def test_empty_particle_overfit_cache_fails_before_model_allocation(monkeypatch):
    import train
    from models import factory
    from utils.config import load_config

    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    monkeypatch.setattr(train, "init_wandb", lambda **kwargs: None)
    monkeypatch.setattr(train, "create_dataloader", lambda **kwargs: [])

    def no_model(*args, **kwargs):
        pytest.fail("An empty overfit cache must be rejected before allocating the model")

    monkeypatch.setattr(factory, "build_particle_model", no_model)
    cfg = load_config("configs/unet.yaml")
    cfg.train.overfit_batches, cfg.train.mixed_precision = 1, "no"
    with pytest.raises(ValueError, match="Empty training loader"):
        train.train(cfg)
