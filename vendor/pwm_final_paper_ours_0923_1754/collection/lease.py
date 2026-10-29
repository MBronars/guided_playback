"""Same-user, same-host GPU worker exclusion; not a cluster resource allocator."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat


@contextmanager
def gpu_lease(uuid, directory=None):
    if not uuid.startswith('GPU-') or not uuid.replace('-', '').isalnum():
        raise ValueError('Full simple GPU UUID required')
    directory = Path(directory or f'/tmp/pwm-collection-leases-{os.getuid()}')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise ValueError('Unsafe lease directory')
    fd = os.open(directory / (uuid + '.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode) or os.fstat(fd).st_uid != os.getuid():
            raise ValueError('Unsafe lease file')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another collection worker holds this GPU lease') from error
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps(dict(pid=os.getpid(), uuid=uuid)).encode())
        os.fsync(fd)
        yield
    finally:
        os.close(fd)  # Never unlink: replacing the inode would permit two owners.
