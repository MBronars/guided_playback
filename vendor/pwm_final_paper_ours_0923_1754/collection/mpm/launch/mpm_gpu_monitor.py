"""Bounded retries for telemetry timeouts, never for simulation/guard failures."""
import functools
import subprocess
import sys


def retry_gpu_status(check, *, attempts=3):
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError('GPU monitor permits one to three attempts')

    @functools.wraps(check)
    def guarded(*args, **kwargs):
        for index in range(attempts):
            try:
                # Retry the entire inventory + ownership check, not stale output.
                return check(*args, **kwargs)
            except subprocess.TimeoutExpired as exc:
                command = exc.cmd
                telemetry = (isinstance(command, (tuple, list)) and len(command) == 3
                    and command[0] == 'nvidia-smi'
                    and command[1] in ('--query-gpu=uuid,memory.used,utilization.gpu',
                                      '--query-compute-apps=gpu_uuid,pid,used_memory')
                    and command[2] == '--format=csv,noheader,nounits')
                if not telemetry or index + 1 == attempts:
                    raise
                print(f'GPU_MONITOR_TIMEOUT retry={index + 1}/{attempts - 1} '
                      f'command={command[1]}', file=sys.stderr, flush=True)
        raise AssertionError('Unreachable')

    return guarded


def install():
    from collection.mpm import collect
    collect.gpu_status = retry_gpu_status(collect.gpu_status)
