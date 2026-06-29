"""FastAPI engine — the backend for the v2 culling viewer.

Run:
    pip install -e ".[api]"
    set AUTOPHOTOS_LIBRARY=C:/Users/willc/Pictures/ukgood   (or export on bash)
    uvicorn autophotos.api:app --port 8731
    # open http://localhost:8731

The browser UI (served at /) is the culling client: keyboard rating, loupe,
stack-review queue, picks, and semantic search. A Tauri shell can later wrap this
exact server as a sidecar for a native app; nothing here changes.
"""
from __future__ import annotations
import io
import os

from . import config, db, pipeline, xmp, pick, semantic

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse, Response, JSONResponse
except Exception as e:  # pragma: no cover
    raise SystemExit("FastAPI not installed. Run: pip install -e '.[api]'") from e


def _lib():
    root = os.environ.get("AUTOPHOTOS_LIBRARY")
    if not root:
        raise HTTPException(400, "set AUTOPHOTOS_LIBRARY to a library path")
    lib = config.Library(root)
    lib.ensure_dirs()
    return lib, db.connect(lib.db_path)


app = FastAPI(title="autophotos")


@app.get("/api/photos")
def photos():
    lib, con = _lib()
    gm = {r["hash"]: r["group_id"]
          for r in con.execute("SELECT hash, group_id FROM group_members")}
    rows = con.execute(
        "SELECT p.hash,p.filename,p.captured_at,p.camera_model,"
        "r.rating,r.pick,r.label,s.aesthetic,s.sharpness,s.exposure_ok "
        "FROM photos p LEFT JOIN ratings r USING(hash) LEFT JOIN scores s USING(hash) "
        "ORDER BY p.captured_at").fetchall()
    out = [dict(r) | {"group": gm.get(r["hash"])} for r in rows]
    return {"library": lib.root, "count": len(out), "photos": out}


@app.get("/api/thumb/{h}")
def thumb(h: str, size: int = 256):
    lib, _ = _lib()
    key = "1024" if size > 256 else "256"
    p = lib.thumb_paths(h)[key]
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


@app.get("/api/search")
def search(q: str = Query(...), k: int = 30):
    lib, con = _lib()
    from .embed import get_embedder
    emb = get_embedder(prefer_clip=True)
    try:
        res = semantic.search_text(lib, q, emb, k=k)
    except Exception as e:
        raise HTTPException(400, str(e))
    return {"q": q, "results": [{"hash": h, "score": s} for h, s in res]}


@app.get("/", response_class=HTMLResponse)
def index():
    here = os.path.dirname(__file__)
    p = os.path.join(here, "web", "culler.html")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read()
    return "<h1>autophotos</h1><p>culler.html missing</p>"
