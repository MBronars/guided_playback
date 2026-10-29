from pathlib import Path
import os
import tempfile
import h5py
from .contracts import canonical_json
from .validate import SCHEMA, _fsync_dir, _fsync_file

def atomic_raw(output, manifest, arrays):
    """No-clobber same-filesystem link is the raw file's completion boundary."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    fd, tmp = tempfile.mkstemp(prefix="." + output.name + ".partial-", dir=output.parent)
    os.close(fd)
    tmp = Path(tmp)
    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["schema"] = SCHEMA
            f.attrs["manifest_json"] = canonical_json(manifest)
            f.attrs["accepted_for_training"] = False
            f.attrs["complete"] = False
            for name, value in arrays.items():
                f.create_dataset(name, data=value)
            f.attrs.modify("complete", True)
            f.flush()
        _fsync_file(tmp)
        os.link(tmp, output)  # Fails atomically if another producer owns this ID.
        _fsync_dir(output.parent)
    finally:
        # Keep failures for inspection; unlink only our successful temporary link.
        if output.exists() and tmp.exists() and os.path.samefile(output, tmp):
            tmp.unlink()
            _fsync_dir(output.parent)
