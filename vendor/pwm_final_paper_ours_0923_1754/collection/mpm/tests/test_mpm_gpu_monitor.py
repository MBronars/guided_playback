import importlib.util
import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

spec=importlib.util.spec_from_file_location('mpm_gpu_monitor',Path(__file__).parents[1]/'launch/mpm_gpu_monitor.py')
monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
CMD=['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']


def test_transient_timeout_rechecks_and_preserves_arguments():
    check=Mock(side_effect=[subprocess.TimeoutExpired(CMD,10),{'checked':True}])
    assert monitor.retry_gpu_status(check)('gpu',allow_own=True)=={'checked':True}
    assert check.call_count==2
    check.assert_called_with('gpu',allow_own=True)


def test_persistent_timeout_stops_after_three():
    check=Mock(side_effect=subprocess.TimeoutExpired(CMD,10))
    with pytest.raises(subprocess.TimeoutExpired):monitor.retry_gpu_status(check)('gpu')
    assert check.call_count==3


@pytest.mark.parametrize('error',[RuntimeError('Foreign process'),ValueError('Malformed telemetry'),
    subprocess.CalledProcessError(1,CMD),subprocess.TimeoutExpired(['simulation'],10)])
def test_other_errors_never_retry(error):
    check=Mock(side_effect=error)
    with pytest.raises(type(error)):monitor.retry_gpu_status(check)('gpu')
    assert check.call_count==1


def test_guard_failure_after_timeout_is_not_masked():
    check=Mock(side_effect=[subprocess.TimeoutExpired(CMD,10),RuntimeError('Foreign process')])
    with pytest.raises(RuntimeError,match='Foreign'):monitor.retry_gpu_status(check)('gpu')
    assert check.call_count==2


def test_ownership_query_timeout_retries():
    command=[CMD[0],'--query-compute-apps=gpu_uuid,pid,used_memory',CMD[2]]
    check=Mock(side_effect=[subprocess.TimeoutExpired(command,10),True])
    assert monitor.retry_gpu_status(check)('gpu') is True


def real_guard(outputs):
    # Exercise the actual guard without importing GPU/physics dependencies.
    source=Path(__file__).parents[1]/'collect.py'
    node=next(n for n in ast.parse(source.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='gpu_status')
    query=Mock(side_effect=outputs)
    namespace=dict(subprocess=SimpleNamespace(check_output=query),os=os,now=lambda:'now')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['gpu_status'],query


def test_real_guard_after_timeout_matches_successful_guard():
    inventory='GPU-1, 757, 77\n';owned=f'GPU-1, {os.getpid()}, 750\n'
    plain,_=real_guard([inventory,owned])
    retried,query=real_guard([subprocess.TimeoutExpired(CMD,10),inventory,owned])
    assert monitor.retry_gpu_status(retried)('GPU-1',allow_own=True)==plain('GPU-1',allow_own=True)
    assert query.call_count==3


@pytest.mark.parametrize('inventory,processes,message',[
    ('GPU-1, 757, 77\n','GPU-1, -123, 750\n','Foreign process'),
    ('GPU-1, 90000, 77\n','', 'memory guard'),
    ('GPU-2, 0, 0\n','', 'missing/ambiguous'),
])
def test_real_guard_remains_fail_closed_after_timeout(inventory,processes,message):
    check,_=real_guard([subprocess.TimeoutExpired(CMD,10),inventory,processes])
    with pytest.raises(RuntimeError,match=message):
        monitor.retry_gpu_status(check)('GPU-1',allow_own=True,memory_limit_mib=80000)
