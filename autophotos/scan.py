"""Filesystem <-> DB reconciliation, keyed by content hash.

Manual moves/renames are no-ops (same hash, path updated); new files are indexed;
vanished files are removed from the cache. Idempotent.
"""
from __future__ import annotations
import os
from datetime import datetime, timezone

from . import config, db, exif
from .hashing import content_hash


def _iter_raws(root: str):
    for dirpath, dirs, files in os.walk(root):
        if ".autophotos" in dirpath:
            continue
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in config.RAW_EXTS:
                yield os.path.join(dirpath, fn), ext


def scan(lib: config.Library, con) -> dict:
    """Returns {new, moved, deleted, unchanged} counts."""
    disk = {}  # hash -> (path, ext)
    for path, ext in _iter_raws(lib.root):
        try:
            h = content_hash(path)
        except OSError:
            continue
        disk[h] = (path, ext)

    known = db.all_hashes_paths(con)
    new = moved = unchanged = 0

    for h, (path, ext) in disk.items():
        if h not in known:
            _index_one(lib, con, h, path, ext)
            new += 1
        elif known[h] != path:
            db.update_path(con, h, path, os.path.basename(path))
            moved += 1
        else:
            unchanged += 1

    deleted = 0
    for h in list(known):
        if h not in disk:
            db.delete_photo(con, h)
            deleted += 1

    con.commit()
    return {"new": new, "moved": moved, "deleted": deleted,
            "unchanged": unchanged, "total": len(disk)}


def _index_one(lib, con, h, path, ext):
    md = exif.read_exif(path)
    row = {
        "hash": h, "path": path, "filename": os.path.basename(path),
        "library": lib.root, "ext": ext, "size": os.path.getsize(path),
        "file_mtime": os.path.getmtime(path),
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        **md,
    }
    db.upsert_photo(con, row)
