"""Deterministic, byte-reproducible WebDataset tar writing.

tarfile.add() writes filesystem mtime/uid/gid/mode and prefixes './', so tars would
differ run-to-run and machine-to-machine. Build TarInfo manually instead.
"""
from __future__ import annotations

import io
import tarfile


def add_bytes(tf: tarfile.TarFile, name: str, data: bytes) -> None:
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = 0
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mode = 0o644
    ti.type = tarfile.REGTYPE
    tf.addfile(ti, io.BytesIO(data))


def open_deterministic(path, mode: str = "w") -> tarfile.TarFile:
    # GNU_FORMAT pinned -- GNU vs PAX changes bytes for otherwise-identical content.
    return tarfile.open(path, mode, format=tarfile.GNU_FORMAT)
