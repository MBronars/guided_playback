"""Minimal wandb logging utilities."""

from typing import Dict, Any, Optional
import wandb


def init_wandb(
    project: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    entity: Optional[str] = None,
) -> None:
    """
    Initialize wandb run.

    Args:
        project: Wandb project name
        config: Config dict to log
        run_name: Optional run name
        entity: Optional wandb entity/team
    """
    wandb.init(
        project=project,
        name=run_name,
        entity=entity,
        config=config,
    )


def log_metrics(metrics: Dict[str, float], step: int) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dict of metric names to values
        step: Training step
    """
    wandb.log(metrics, step=step)


def log_video(key: str, path: str, step: int, fps: int = 10) -> None:
    """
    Log a video/GIF to wandb.

    Args:
        key: Metric key name
        path: Path to video/GIF file
        step: Training step
        fps: Frames per second for display
    """
    wandb.log({key: wandb.Video(path, fps=fps, format="gif")}, step=step)


def finish() -> None:
    """Finish wandb run."""
    wandb.finish()
