"""SQLite cache access. This DB is 100% rebuildable from the filesystem."""
from __future__ import annotations
import sqlite3
from typing import Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
  hash TEXT PRIMARY KEY, path TEXT NOT NULL, filename TEXT NOT NULL,
  library TEXT NOT NULL, ext TEXT NOT NULL, size INTEGER, file_mtime REAL,
  captured_at TEXT, sub_sec INTEGER, camera_model TEXT, lens TEXT,
  focal_len REAL, aperture REAL, shutter REAL, iso INTEGER, ev_bias REAL,
  focus_dist REAL, drive_mode TEXT, orientation INTEGER, width INTEGER,
  height INTEGER, indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS ratings (
  hash TEXT PRIMARY KEY REFERENCES photos(hash), rating INTEGER, pick INTEGER,
  label TEXT, xmp_path TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS scores (
  hash TEXT PRIMARY KEY REFERENCES photos(hash), aesthetic REAL, sharpness REAL,
  exposure_ok REAL, personal REAL, scored_at TEXT
);
CREATE TABLE IF NOT EXISTS groups (
  group_id TEXT PRIMARY KEY, kind_hint TEXT, t_start TEXT, t_end TEXT,
  n INTEGER, confidence REAL
);
CREATE TABLE IF NOT EXISTS group_members (
  group_id TEXT REFERENCES groups(group_id), hash TEXT REFERENCES photos(hash),
  seq INTEGER, PRIMARY KEY (group_id, hash)
);
CREATE INDEX IF NOT EXISTS idx_photos_captured ON photos(captured_at);
CREATE INDEX IF NOT EXISTS idx_photos_library ON photos(library);
"""

PHOTO_COLS = [
    "hash", "path", "filename", "library", "ext", "size", "file_mtime",
    "captured_at", "sub_sec", "camera_model", "lens", "focal_len", "aperture",
    "shutter", "iso", "ev_bias", "focus_dist", "drive_mode", "orientation",
    "width", "height", "indexed_at",
]


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert_photo(con, row: dict):
    cols = ",".join(PHOTO_COLS)
    ph = ",".join("?" for _ in PHOTO_COLS)
    upd = ",".join(f"{c}=excluded.{c}" for c in PHOTO_COLS if c != "hash")
    con.execute(
        f"INSERT INTO photos ({cols}) VALUES ({ph}) "
        f"ON CONFLICT(hash) DO UPDATE SET {upd}",
        [row.get(c) for c in PHOTO_COLS],
    )


def update_path(con, hash_: str, path: str, filename: str):
    con.execute("UPDATE photos SET path=?, filename=? WHERE hash=?",
                (path, filename, hash_))


def delete_photo(con, hash_: str):
    for t in ("photos", "ratings", "scores", "group_members"):
        con.execute(f"DELETE FROM {t} WHERE hash=?", (hash_,))


def all_hashes_paths(con) -> dict:
    return {r["hash"]: r["path"] for r in con.execute("SELECT hash, path FROM photos")}


def photos(con) -> list:
    return list(con.execute("SELECT * FROM photos ORDER BY captured_at"))


def upsert_rating(con, hash_, rating, pick, label, xmp_path, updated_at):
    con.execute(
        "INSERT INTO ratings (hash,rating,pick,label,xmp_path,updated_at) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(hash) DO UPDATE SET "
        "rating=excluded.rating,pick=excluded.pick,label=excluded.label,"
        "xmp_path=excluded.xmp_path,updated_at=excluded.updated_at",
        (hash_, rating, pick, label, xmp_path, updated_at),
    )


def upsert_score(con, hash_, aesthetic, sharpness, exposure_ok, personal, scored_at):
    con.execute(
        "INSERT INTO scores (hash,aesthetic,sharpness,exposure_ok,personal,scored_at) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(hash) DO UPDATE SET "
        "aesthetic=excluded.aesthetic,sharpness=excluded.sharpness,"
        "exposure_ok=excluded.exposure_ok,personal=excluded.personal,"
        "scored_at=excluded.scored_at",
        (hash_, aesthetic, sharpness, exposure_ok, personal, scored_at),
    )


def replace_groups(con, groups: Iterable):
    con.execute("DELETE FROM group_members")
    con.execute("DELETE FROM groups")
    for g in groups:
        con.execute(
            "INSERT INTO groups (group_id,kind_hint,t_start,t_end,n,confidence) "
            "VALUES (?,?,?,?,?,?)",
            (g.group_id, g.kind_hint, g.t_start, g.t_end, len(g.members), g.confidence),
        )
        for seq, h in enumerate(g.members):
            con.execute(
                "INSERT INTO group_members (group_id,hash,seq) VALUES (?,?,?)",
                (g.group_id, h, seq),
            )
