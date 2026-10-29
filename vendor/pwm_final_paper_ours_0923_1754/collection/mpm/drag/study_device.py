"""Explicit sharing guard for contact diagnostics only; production unchanged."""
import os
from pathlib import Path
import subprocess


def process_start(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    if p.stat().st_uid!=os.getuid():raise RuntimeError('Only explicitly approved same-user co-tenant allowed')
    # /proc/stat's comm field can contain spaces; field 22 is start time.
    return (p/'stat').read_text().rsplit(')',1)[1].split()[19]


class DeviceGuard:
    def __init__(self,uuid,shared_pid=None):
        if shared_pid is not None and (type(shared_pid) is not int or shared_pid<=0 or shared_pid==os.getpid()):
            raise ValueError('Explicit distinct co-tenant PID required')
        self.uuid=uuid;self.shared_pid=shared_pid
        self.shared_start=None if shared_pid is None else process_start(shared_pid)

    def check(self,startup=False):
        if self.shared_pid is None:
            if startup:
                from .cli import wait_idle
                return wait_idle(self.uuid)
            from ..collect import gpu_status
            return gpu_status(self.uuid,allow_own=True,memory_limit_mib=16000)
        inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
        matches=[r.split(',') for r in inventory.splitlines() if r.split(',')[0].strip()==self.uuid]
        if len(matches)!=1:raise RuntimeError('Missing/ambiguous diagnostic GPU')
        total,used=map(int,matches[0][1:]);free=total-used
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
        own=0;active=[]
        for line in output.splitlines():
            gpu,pid,memory=[s.strip() for s in line.split(',')]
            if gpu!=self.uuid:continue
            pid=int(pid);active.append(pid)
            if pid==os.getpid():own+=int(memory)
            elif (pid!=self.shared_pid or self.shared_start is None or process_start(pid)!=self.shared_start):
                raise RuntimeError('Unapproved process on diagnostic GPU')
        if own>16000 or free<8192:raise RuntimeError('Shared diagnostic GPU memory reserve exceeded')
        return dict(uuid=self.uuid,approved_co_tenant_pid=self.shared_pid,co_tenant_start_ticks=self.shared_start,
                    own_memory_mib=own,free_memory_mib=free,active_pids=active,
                    performance_comparison_valid=False,production_guard_unchanged=True)
