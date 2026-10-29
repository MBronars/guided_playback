"""Read only the requested NPZ member; memory-map stored numeric arrays.

Dataset shards also contain cameras and many parallel environments. np.load()[key]
would repeatedly allocate the entire member before slicing. ZIP_STORED members can
instead be mapped directly, including Zip64 local headers. Compressed exports fall
back to NumPy's checked reader. Object arrays are never unpickled.
"""
from __future__ import annotations

import struct
import zipfile

import numpy as np


def npz_array(path, key):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            with archive.open(info) as stream:
                return np.lib.format.read_array(stream, allow_pickle=False)
    with open(path, "rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError(f"Invalid ZIP local header: {path}:{key}")
        name_size, extra_size = struct.unpack("<HH", header[26:30])
        stream.seek(name_size + extra_size, 1)
        member_start = stream.tell()
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if dtype.hasobject:
            raise ValueError(f"Object arrays are not supported: {path}:{key}")
        offset = stream.tell()
        size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if offset + size != member_start + info.file_size:
            raise ValueError(f"NPY size disagrees with ZIP member: {path}:{key}")
    if size == 0:
        return np.empty(shape, dtype=dtype, order="F" if fortran else "C")
    return np.memmap(path, dtype=dtype, mode="r", shape=shape,
                     order="F" if fortran else "C", offset=offset)
