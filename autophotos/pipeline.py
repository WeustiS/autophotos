"""Orchestration: scan -> thumbs -> embed -> group -> score, plus XMP sync."""
from __future__ import annotations
import json, os
from datetime import datetime, timezone
import numpy as np
from . import config, db, raw, group as grp, score, xmp
from .embed import get_embedder


def _now():
    return datetime.now(timezone.utc).isoformat()


def make_thumbnails(lib, con):
    n = 0
    for r in db.photos(con):
        out = lib.thumb_paths(r["hash"])
        if os.path.exists(out["256"]):
            continue
        try:
            raw.make_thumbs(r["path"], out); n += 1
        except Exception as e:
            print(f"[thumb] {r['filename']}: {e}")
    return n


def embed_library(lib, con, prefer_clip=True):
    rows = db.photos(con)
    emb = get_embedder(prefer_clip=prefer_clip)
    paths, hashes = [], []
    for r in rows:
        tp = lib.thumb_paths(r["hash"])["1024"]
        if os.path.exists(tp):
            paths.append(tp); hashes.append(r["hash"])
    if not paths:
        return {"n": 0, "model": emb.id, "semantic": emb.semantic}
    vecs = emb.embed_images(paths)
    np.save(lib.emb_path, vecs)
    json.dump(hashes, open(lib.ids_path, "w"))
    json.dump({"id": emb.id, "dim": int(vecs.shape[1]), "semantic": emb.semantic,
               "embedded_at": _now()}, open(lib.model_path, "w"), indent=2)
    return {"n": len(hashes), "model": emb.id, "semantic": emb.semantic,
            "dim": int(vecs.shape[1])}


def load_embeddings(lib):
    if not (os.path.exists(lib.emb_path) and os.path.exists(lib.ids_path)):
        return {}
    vecs = np.load(lib.emb_path); ids = json.load(open(lib.ids_path))
    return {h: vecs[i] for i, h in enumerate(ids)}


def group_library(lib, con):
    rows = [dict(r) for r in db.photos(con)]
    emb = load_embeddings(lib)
    groups = grp.detect_groups(rows, emb or None)
    db.replace_groups(con, groups); con.commit()
    return groups


def score_library(lib, con):
    emb = load_embeddings(lib)
    model = json.load(open(lib.model_path)) if os.path.exists(lib.model_path) else {}
    semantic = model.get("semantic", False)
    n = 0
    for r in db.photos(con):
        tp = lib.thumb_paths(r["hash"])
        if not os.path.exists(tp["1024"]):
            continue
        sh = score.sharpness(tp["1024"]); ex = score.exposure_ok(tp["1024"])
        ae = score.aesthetic(emb.get(r["hash"]), semantic, lib.cache_dir)
        db.upsert_score(con, r["hash"], ae, sh, ex, None, _now()); n += 1
    con.commit()
    return n


def sync_ratings_from_xmp(lib, con):
    n = 0
    for r in db.photos(con):
        rt = xmp.read_rating(r["path"])
        if rt["xmp_path"]:
            db.upsert_rating(con, r["hash"], rt["rating"], rt["pick"],
                             rt["label"], rt["xmp_path"], _now()); n += 1
    con.commit()
    return n
