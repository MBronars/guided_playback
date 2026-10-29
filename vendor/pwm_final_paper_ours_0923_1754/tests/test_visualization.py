"""Headless visualization smoke tests using synthetic supported trajectories."""
import numpy as np
from PIL import Image
import pytest
import torch

from utils.visualization import create_trajectory_comparison_gif


@pytest.mark.parametrize("show_gt", [False, True])
def test_trajectory_gif_contains_all_frames(tmp_path, show_gt):
    rng = np.random.default_rng(42)
    points = rng.normal(size=(8, 3)).astype(np.float32) * 0.01
    positions = points[None] + np.arange(3, dtype=np.float32)[:, None, None] * 0.02
    mask = np.arange(8) < 2
    output = tmp_path / "trajectory.gif"
    gt = torch.from_numpy(positions)[None]
    create_trajectory_comparison_gif(
        gt, gt + 0.01, torch.from_numpy(mask)[None], str(output),
        figsize=(4, 2), fps=5, show_gt=show_gt)
    with Image.open(output) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 3
        assert min(gif.size) >= 100
        frames = []
        for i in range(gif.n_frames):
            gif.seek(i)
            assert gif.info["duration"] == 200
            frames.append(np.array(gif.convert("RGB")))
        assert not np.array_equal(frames[0], frames[-1])


def test_comparison_uses_explicit_bounds_without_an_empty_legend(tmp_path, monkeypatch, caplog):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    positions = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3) / 100
    limits = ((-0.2, 0.5), (-0.3, 0.6), (-0.4, 0.7))
    closed = []
    close = plt.close

    def record_close(figure=None):
        if isinstance(figure, Figure):
            for axis in figure.axes:
                closed.append((axis.get_xlim(), axis.get_ylim(), axis.get_zlim()))
        close(figure)

    monkeypatch.setattr(plt, "close", record_close)
    create_trajectory_comparison_gif(positions, positions + 0.01,
        torch.zeros(1, 3, dtype=torch.bool), str(tmp_path / "bounded.gif"),
        object_ids=torch.zeros(1, 3, dtype=torch.long), axis_limits=limits, figsize=(4, 2))
    assert len(closed) == 4  # two panels, two frames
    np.testing.assert_allclose(closed, np.broadcast_to(limits, (4, 3, 2)))
    assert "No artists with labels" not in caplog.text


@pytest.mark.parametrize("limits", [((0, 1),), ((0, 1), (2, 1), (0, 1)),
                                    ((0, 1), (0, float("nan")), (0, 1))])
def test_comparison_rejects_invalid_explicit_bounds(tmp_path, limits):
    positions = torch.zeros(1, 1, 2, 3)
    output = tmp_path / "invalid.gif"
    with pytest.raises(ValueError, match="axis_limits"):
        create_trajectory_comparison_gif(positions, positions,
            torch.zeros(1, 2, dtype=torch.bool), str(output), axis_limits=limits)
    assert not output.exists()


@pytest.mark.parametrize("runs", [1, 2, 3])
@pytest.mark.parametrize("show_gt", [False, True])
@pytest.mark.parametrize("multi_view", [False, True])
def test_ensemble_panels_are_allocated_and_annotated(tmp_path, monkeypatch, runs, show_gt, multi_view):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    positions = torch.arange(24, dtype=torch.float32).reshape(1, 2, 4, 3) / 100
    predictions = torch.stack([positions + .01 * (run + 1) for run in range(runs)])
    masks = torch.tensor([[False, False, False, True]])
    limits = ((-.1, .5), (-.1, .5), (-.1, .5))
    panels = []
    close = plt.close

    def record_close(figure=None):
        if isinstance(figure, Figure):
            panels.append([dict(title=axis.get_title(), xlabel=axis.get_xlabel(),
                                ylabel=axis.get_ylabel(), zlabel=axis.get_zlabel(),
                                limits=(axis.get_xlim(), axis.get_ylim(), axis.get_zlim()))
                           for axis in figure.axes])
        close(figure)

    monkeypatch.setattr(plt, "close", record_close)
    output = tmp_path / "ensemble.gif"
    create_trajectory_comparison_gif(positions, predictions, masks, str(output),
        actions=torch.ones_like(positions) * .001, figsize=(9, 5),
        show_gt=show_gt, multi_view=multi_view, axis_limits=limits)
    expected_count = 3 * (1 + int(show_gt)) if multi_view else runs + int(show_gt)
    assert len(panels) == 2
    for frame in panels:
        assert len(frame) == expected_count
        for panel in frame:
            assert panel["title"]
            assert (panel["xlabel"], panel["ylabel"], panel["zlabel"]) == ("X", "Y", "Z")
            np.testing.assert_allclose(panel["limits"], limits)
        if not multi_view and runs > 1:
            for run in range(runs):
                assert frame[run + int(show_gt)]["title"].startswith(f"Run {run} ")
    with Image.open(output) as gif:
        assert gif.n_frames == 2


def test_empty_ensemble_fails_before_creating_figures(tmp_path):
    import matplotlib.pyplot as plt

    positions = torch.zeros(1, 2, 4, 3)
    before = plt.get_fignums()
    output = tmp_path / "empty.gif"
    with pytest.raises(ValueError, match="at least one ensemble run"):
        create_trajectory_comparison_gif(positions, torch.empty(0, 1, 2, 4, 3),
            torch.zeros(1, 4, dtype=torch.bool), str(output))
    assert plt.get_fignums() == before
    assert not output.exists()
