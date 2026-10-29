"""launch/: the pwm-data-v1 templates and submit scripts run from any account, each run in the calling account's own folders."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCH = Path(__file__).resolve().parents[1] / "launch"
SUBMIT = sorted(LAUNCH.glob("submit_*.sh"))
RECIPE = Path("/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1/configs/unet_pushing_100m.yaml")


def _code(path):
    """The lines bash or SLURM act on: #SBATCH directives and everything that is not a comment."""
    return [line for line in path.read_text().splitlines() if line.startswith("#SBATCH") or not line.lstrip().startswith("#")]


def test_the_templates_read_the_conda_installation_and_the_run_folders_from_the_calling_account():
    for name in ("train_pwm_data_v1.sbatch", "smoke_pwm_data_v1.sbatch", "overfit_pwm_data_v1.sbatch"):
        code = _code(LAUNCH / name)
        assert not [line for line in code if "/home/" in line and "PWM_CONDA_SH" not in line], name     # nobody's private conda
        assert 'source "$PWM_CONDA_SH"' in code and 'conda activate "${PWM_CONDA_ENV:-pwm_blackwell}"' in code, name
    code = _code(LAUNCH / "train_pwm_data_v1.sbatch")
    assert any(line.startswith("ME=$(id -un)") for line in code)
    assert "LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$ME}" in code
    assert "SCRATCH=${SCRATCH:-/data/user_data/$ME/scratchpad/launch_pwm_data_v1}" in code
    # One account is named twice: the shared recipe folder, and the log path SLURM reads before any variable exists
    # (every submit script passes --output under LOG_DIR).
    assert [line.split("=")[0] for line in code if "robertre" in line] == ["#SBATCH --output", "ROOT"]


def _stub_id(folder, account):
    folder.mkdir(exist_ok=True)
    (folder / "id").write_text(f"#!/bin/bash\necho {account}\n")
    (folder / "id").chmod(0o755)
    return folder


@pytest.mark.parametrize("account", ["robertre", "labmate"])
def test_the_training_template_expands_its_locations_for_the_calling_account(tmp_path, account):
    """For robertre the four values are the ones the template held as constants until 05e8978, so those runs resume where they were."""
    names = ("ME", "LOG_DIR", "SCRATCH", "PWM_CONDA_SH")
    lines = [line for line in _code(LAUNCH / "train_pwm_data_v1.sbatch")                    # the four assignments of a default
             if line.startswith("ME=") or any(line.startswith(f"{name}=${{{name}:-") for name in names)]
    assert [line.split("=")[0] for line in lines] == list(names)
    inherited = {k: v for k, v in os.environ.items() if k not in names}
    result = subprocess.run(["bash", "-c", "\n".join(["set -euo pipefail", *lines, 'echo "$LOG_DIR|$SCRATCH|$PWM_CONDA_SH"'])],
                            capture_output=True, text=True, env={**inherited, "HOME": f"/home/{account}",
                                                                 "PATH": f"{_stub_id(tmp_path / 'bin', account)}:{inherited['PATH']}"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == (f"/data/group_data/katefgroup/runs_pwm_{account}|/data/user_data/{account}/scratchpad/launch_pwm_data_v1|"
                                     f"/home/{account}/miniconda3/etc/profile.d/conda.sh")


def test_every_submit_script_exports_the_run_folder_and_logs_under_it():
    assert len(SUBMIT) == 17
    for path in SUBMIT:
        code = _code(path)
        # The L40S jobs export OVERRIDES and PWM_CONDA_ENV beside the two required names.
        assert "LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$(id -un)}" in code \
            and any(line.startswith("export ROOT LOG_DIR") for line in code), path.name
        # the L40S jobs spell their resources out, so CMD=(sbatch ...) spans several lines
        start = next(i for i, line in enumerate(code) if line.startswith("CMD=(sbatch"))
        cmd = code[start]
        while ")" not in cmd:
            start += 1
            cmd += " " + code[start].strip()
        assert '--output="$LOG_DIR/slurm/%x_%j.log"' in cmd, path.name
        printed = next(i for i, line in enumerate(code) if '"--print"' in line and "exit 0" in line)
        assert code.index('mkdir -p "$LOG_DIR/slurm"') > printed, path.name             # --print creates nothing


@pytest.fixture()
def checkout(tmp_path):
    """A clean one-commit checkout of the launch folder, a recipe folder with the pushing recipe, and an `id` that answers `labmate`."""
    if not RECIPE.is_file():
        pytest.skip("the launch recipes are not on this machine")
    repo, root, stubs = tmp_path / "repo", tmp_path / "root", tmp_path / "bin"
    shutil.copytree(LAUNCH, repo / "launch")
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-q", "--no-verify", "-m", "launch"], check=True)
    (root / "configs").mkdir(parents=True)
    shutil.copy(RECIPE, root / "configs")
    _stub_id(stubs, "labmate")
    (stubs / "sbatch").write_text("#!/bin/bash\necho SUBMITTED; exit 1\n")              # --print never reaches it
    (stubs / "sbatch").chmod(0o755)
    return repo, root, stubs


def _print(checkout, **env):
    repo, root, stubs = checkout
    inherited = {k: v for k, v in os.environ.items() if k not in ("LOG_DIR", "ROOT")}
    result = subprocess.run(["bash", str(repo / "launch" / "submit_pushing_100m.sh"), "--print"], capture_output=True, text=True,
                            env={**inherited, "PATH": f"{stubs}:{inherited['PATH']}", "ROOT": str(root), **env})
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_a_submit_script_puts_the_run_under_the_calling_account(checkout):
    line = _print(checkout)
    assert 'LOG_DIR="/data/group_data/katefgroup/runs_pwm_labmate"' in line and "SUBMITTED" not in line
    assert "--output=/data/group_data/katefgroup/runs_pwm_labmate/slurm/%x_%j.log" in line
    assert not Path("/data/group_data/katefgroup/runs_pwm_labmate").exists()            # --print creates nothing
    assert "--output=/somewhere/else/slurm/%x_%j.log" in _print(checkout, LOG_DIR="/somewhere/else")
    (checkout[0] / "launch" / "README.md").write_text("changed")                        # a checkout with local changes is refused
    with pytest.raises(AssertionError, match="local changes"):
        _print(checkout)
