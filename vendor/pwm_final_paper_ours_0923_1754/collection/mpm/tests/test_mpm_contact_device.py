import os
import pytest
from collection.mpm.drag import study_device


@pytest.mark.parametrize('damage',[None,'unknown_process','reused_pid','own_memory','free_memory'])
def test_sharing_is_pid_bound_and_memory_bounded(monkeypatch,damage):
    pid=os.getpid()+12345;start=['123']
    monkeypatch.setattr(study_device,'process_start',lambda p:start[0])
    guard=study_device.DeviceGuard('GPU-unit',pid)
    if damage=='reused_pid':start[0]='456'
    used=90000 if damage=='free_memory' else 55000
    own=17000 if damage=='own_memory' else 4000
    rows=f'GPU-unit, {pid}, 50000\nGPU-unit, {os.getpid()}, {own}\n'
    if damage=='unknown_process':rows+=f'GPU-unit, {pid+1}, 10\n'
    monkeypatch.setattr(study_device.subprocess,'check_output',lambda args,**kw:
                        f'GPU-unit, 95000, {used}\n' if '--query-gpu=uuid,memory.total,memory.used' in args else rows)
    if damage:
        with pytest.raises(RuntimeError):guard.check()
    else:
        result=guard.check()
        assert result['own_memory_mib']==4000 and result['performance_comparison_valid'] is False
        assert result['production_guard_unchanged'] is True
