import json
from pathlib import Path
import pytest

from collection.mpm import evaluation_supervisor as supervisor


def failed_case(tmp_path, message=supervisor.RESOURCE_ERRORS[0], phase='simulate'):
    attempt = tmp_path / 'attempt-0000'
    attempt.mkdir()
    (attempt / f'FAILURE-{phase}.json').write_text(json.dumps(
        dict(phase=phase, terminal_candidate=False)))
    (attempt / 'simulate.log').write_text('traceback\n' + message + '\n')
    return attempt


@pytest.mark.parametrize('message', supervisor.RESOURCE_ERRORS)
def test_only_known_contention_retry(tmp_path, message):
    failed_case(tmp_path, message)
    assert supervisor.retry_decision(tmp_path) is True


@pytest.mark.parametrize('message', [
    'RuntimeError: CUDA out of memory',
    'RuntimeError: Nonfinite particles',
    'ValueError: Changed bound artifact',
    supervisor.RESOURCE_ERRORS[0] + '\nRuntimeError: numerical failure',
    '',
])
def test_unknown_errors_not_retried(tmp_path, message):
    failed_case(tmp_path, message)
    with pytest.raises(RuntimeError, match='Not an approved'):
        supervisor.retry_decision(tmp_path)


def test_new_candidate(tmp_path):
    assert supervisor.retry_decision(tmp_path) is False


@pytest.mark.parametrize('file', ['REVIEW.json', 'SIMULATION.json'])
def test_no_resimulation_of_completed_work(tmp_path, file):
    attempt = failed_case(tmp_path)
    (attempt / file).write_text('{}')
    assert supervisor.retry_decision(tmp_path) is False


@pytest.mark.parametrize('phase', ['prepare', 'review'])
def test_other_phase_failure_blocks(tmp_path, phase):
    failed_case(tmp_path, phase=phase)
    with pytest.raises(RuntimeError, match='Non-simulation'):
        supervisor.retry_decision(tmp_path)


def test_ambiguous_interruption_blocks(tmp_path):
    (tmp_path / 'attempt-0000').mkdir()
    with pytest.raises(RuntimeError, match='Unclassified'):
        supervisor.retry_decision(tmp_path)


def test_retry_budget(tmp_path):
    failed_case(tmp_path)
    with pytest.raises(RuntimeError, match='budget exhausted'):
        supervisor.retry_decision(tmp_path, max_attempts=1)


def test_terminal_failure_not_retryable(tmp_path):
    attempt = failed_case(tmp_path)
    (attempt/'FAILURE-simulate.json').write_text(json.dumps(
        dict(phase='simulate', terminal_candidate=True)))
    with pytest.raises(RuntimeError, match='Invalid resource'):
        supervisor.retry_decision(tmp_path)


def test_claim_excludes_and_releases(tmp_path):
    path = tmp_path / 'lock'
    with supervisor.claim(path) as first:
        assert first
        with supervisor.claim(path) as second:
            assert not second
    with supervisor.claim(path) as third:
        assert third


@pytest.mark.parametrize('memory,utilization,processes,idle', [
    (0, 0, '', True), (32, 0, '', True), (33, 0, '', False),
    (0, 100, '', False), (0, 0, 'GPU-test, 123\n', False),
    (0, 0, 'GPU-other, 123\n', True),
])
def test_gpu_preflight_unchanged(monkeypatch, memory, utilization, processes, idle):
    answers = iter([f'GPU-test, {memory}, {utilization}\n', processes])
    monkeypatch.setattr(supervisor.subprocess, 'check_output', lambda *a, **k: next(answers))
    assert supervisor.gpu_idle('GPU-test')[0] is idle


def test_missing_gpu_blocks(monkeypatch):
    monkeypatch.setattr(supervisor.subprocess, 'check_output', lambda *a, **k: 'GPU-other, 0, 0\n')
    with pytest.raises(RuntimeError, match='missing or ambiguous'):
        supervisor.gpu_idle('GPU-test')


@pytest.mark.parametrize('message', supervisor.RESOURCE_ERRORS)
def test_wrapped_passive_resource_failure_requires_corresponding_result(tmp_path, message):
    reasons = [message.replace('RuntimeError: ', 'RuntimeError:', 1)]
    line = 'RuntimeError: Simulation failed; preserve and diagnose: ' + json.dumps(reasons)
    attempt = failed_case(tmp_path, line)
    with pytest.raises(RuntimeError, match='Not an approved'):
        supervisor.retry_decision(tmp_path)
    result = attempt/'input/cases/example/attempts/attempt-00000/RESULT.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps(dict(status='failed', reasons=reasons)))
    assert supervisor.retry_decision(tmp_path)
    result.write_text(json.dumps(dict(status='complete', reasons=reasons)))
    with pytest.raises(RuntimeError, match='Not an approved'):
        supervisor.retry_decision(tmp_path)


@pytest.mark.parametrize('reasons', [[], ['RuntimeError:Nonfinite particles'],
    ['RuntimeError:Foreign process on selected GPU; owned worker stops', 'ValueError:Invalid deformation'],
    [42], {'error': 'GPU idle/memory guard failed'}])
def test_wrapped_unknown_or_mixed_failure_blocks(tmp_path, reasons):
    line = 'RuntimeError: Simulation failed; preserve and diagnose: ' + json.dumps(reasons)
    failed_case(tmp_path, line)
    with pytest.raises(RuntimeError, match='Not an approved'):
        supervisor.retry_decision(tmp_path)
