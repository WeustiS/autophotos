"""FastAPI engine — backend for the v2 culling viewer.

  pip install -e ".[api]"
  export AUTOPHOTOS_LIBRARY=/path/to/library
  uvicorn autophotos.api:app --port 8731   # open http://localhost:8731
"""
from __future__ import annotations
import os

from . import (config, db, pipeline, xmp, pick, semantic, taste, crop,
               categories, decisions)

try:
    from fastapi import FastAPI, HTTPException, Query, Body
    from fastapi.responses import HTMLResponse, Response
except Exception as e:  # pragma: no cover
    raise SystemExit("FastAPI not installed. Run: pip install -e '.[api]'") from e


def _lib():
    root = os.environ.get("AUTOPHOTOS_LIBRARY")
    if not root:
        raise HTTPException(400, "set AUTOPHOTOS_LIBRARY to a library path")
    lib = config.Library(root); lib.ensure_dirs()
    return lib, db.connect(lib.db_path)


app = FastAPI(title="autophotos")

_CROP = {}
def _crop_models(lib):
    if "done" in _CROP:
        return _CROP.get("emb"), _CROP.get("head"), _CROP.get("tw")
    _CROP["done"] = True
    try:
        import json, numpy as np
        from .score import AestheticHead
        from .embed import get_embedder
        meta = json.load(open(lib.model_path)) if os.path.exists(lib.model_path) else {}
        if meta.get("semantic"):
            _CROP["emb"] = get_embedder(prefer_clip=True)
            _CROP["head"] = AestheticHead.load(lib.cache_dir)
            tp = os.path.join(lib.cache_dir, "taste_head.npz")
            _CROP["tw"] = np.load(tp)["w"] if os.path.exists(tp) else None
    except Exception as e:
        print("[crops] model load failed:", e)
    return _CROP.get("emb"), _CROP.get("head"), _CROP.get("tw")



@app.get("/api/photos")
def photos():
    lib, con = _lib()
    gm = {r["hash"]: r["group_id"]
          for r in con.execute("SELECT hash, group_id FROM group_members")}
    rows = con.execute(
        "SELECT p.hash,p.filename,p.captured_at,p.camera_model,r.rating,r.pick,"
        "r.label,s.aesthetic,s.personal,s.sharpness,s.exposure_ok "
        "FROM photos p LEFT JOIN ratings r USING(hash) LEFT JOIN scores s USING(hash) "
        "ORDER BY p.captured_at").fetchall()
    q = set(decisions.queue_list(lib.decisions_path))
    out = [dict(r) | {"group": gm.get(r["hash"]), "queued": r["hash"] in q} for r in rows]
    return {"library": lib.root, "count": len(out), "photos": out}


@app.get("/api/thumb/{h}")
def thumb(h: str, size: int = 256):
    lib, _ = _lib()
    p = lib.thumb_paths(h)["1024" if size > 256 else "256"]
    if not os.path.exists(p):
        raise HTTPException(404, "no thumb")
    return Response(open(p, "rb").read(), media_type="image/jpeg")


@app.get("/api/preview/{h}")
def preview(h: str):
    lib, _ = _lib()
    p = lib.thumb_paths(h)["preview"]
    if not os.path.exists(p):
        raise HTTPException(404, "no preview")
    return Response(open(p, "rb").read(), media_type="image/jpeg")


@app.post("/api/rate")
def rate(hash: str = Query(...), stars: int | None = None,
         reject: bool = False, label: str | None = None):
    lib, con = _lib()
    row = con.execute("SELECT path FROM photos WHERE hash=?", (hash,)).fetchone()
    if not row:
        raise HTTPException(404, "unknown hash")
    xmp.write_rating(row["path"], rating=stars, pick=-1 if reject else None, label=label)
    pipeline.sync_ratings_from_xmp(lib, con)
    return {"ok": True, "rating": xmp.read_rating(row["path"])}


@app.get("/api/groups")
def groups():
    lib, con = _lib()
    gs = []
    for g in con.execute("SELECT * FROM groups ORDER BY t_start"):
        mem = [r["hash"] for r in con.execute(
            "SELECT hash FROM group_members WHERE group_id=? ORDER BY seq", (g["group_id"],))]
        gs.append(dict(g) | {"members": mem})
    return {"count": len(gs), "groups": gs}


@app.get("/api/picks")
def picks(k: int = 50):
    lib, con = _lib()
    return {"summary": pick.summary(con), "picks": pick.top_picks(con, k=k)}


@app.get("/api/review")
def review(k: int = 100):
    lib, con = _lib()
    return {"order": taste.review_order(lib, con)[:k]}


@app.post("/api/train-taste")
def train_taste(l2: float = 1.0):
    lib, con = _lib()
    rep = taste.train(lib, con, l2=l2)
    if rep.get("trained"):
        rep["applied"] = taste.apply_scores(lib, con)
    return rep


@app.get("/api/crops/{h}")
def crops(h: str, k: int = 3):
    lib, con = _lib()
    row = con.execute("SELECT width,height FROM photos WHERE hash=?", (h,)).fetchone()
    ratio = (row["width"]/row["height"]) if row and row["width"] and row["height"] else None
    tp = lib.thumb_paths(h)["1024"]
    if not os.path.exists(tp):
        raise HTTPException(404, "no thumb")
    emb, head, tw = _crop_models(lib)
    return {"hash": h, "crops": crop.suggest_crops(tp, n=k, orig_ratio=ratio,
                                                   embedder=emb, head=head, taste_w=tw)}


@app.get("/api/search")
def search(q: str = Query(...), k: int = 30):
    lib, con = _lib()
    from .embed import get_embedder
    try:
        res = semantic.search_text(lib, q, get_embedder(True), k=k)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"q": q, "results": [{"hash": h, "score": s} for h, s in res]}


@app.get("/api/queue")
def queue_get():
    lib, _ = _lib()
    return {"queue": decisions.queue_list(lib.decisions_path)}


@app.post("/api/queue")
def queue_post(op: str = Query(...), hashes: list[str] = Body(default=[])):
    lib, _ = _lib()
    p = lib.decisions_path
    q = (decisions.queue_add(p, hashes) if op == "add"
         else decisions.queue_remove(p, hashes) if op == "rm"
         else decisions.queue_list(p))
    return {"queue": q}


@app.post("/api/confirm-group")
def confirm_group(members: list[str] = Body(...), kind: str = Body("burst"),
                  action: str = Body("keep-best")):
    lib, _ = _lib()
    g = decisions.confirm_group(lib.decisions_path, members, kind, action)
    return {"confirmed_groups": len(g)}



@app.post("/api/caption")
def caption():
    lib, con = _lib()
    res = categories.caption_library(lib)
    return {"captioned": len(res)}

@app.get("/", response_class=HTMLResponse)
def index():
    p = os.path.join(os.path.dirname(__file__), "web", "culler.html")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else "<h1>autophotos</h1>"
