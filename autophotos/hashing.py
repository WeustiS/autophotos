"""Content-based identity hash.

Full-file hashing of tens of GB is too slow, so we hash size + head + tail. This
is stable across moves/renames and collision-safe for distinct photographs, which
is all we need for identity. Falls back to blake2b if xxhash is unavailable.
"""
from __future__ import annotations
import os

try:
    import xxhash
    def _new():
        return xxhash.xxh64()
except Exception:  # pragma: no cover
    import hashlib
    def _new():
        return hashlib.blake2b(digest_size=8)

HEAD = 1 << 20  # 1 MiB
TAIL = 1 << 20


def content_hash(path: str, head: int = HEAD, tail: int = TAIL) -> str:
    size = os.path.getsize(path)
    h = _new()
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(head))
        if size > head + tail:
            f.seek(-tail, os.SEEK_END)
            h.update(f.read(tail))
    return h.hexdigest()
