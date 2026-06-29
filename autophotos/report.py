"""One-shot self-test + report. Run on a machine with CLIP installed:

    python -m autophotos.report "C:\\Users\\willc\\Pictures\\ukgood"

Runs the whole pipeline (index + aesthetic head + a batch of semantic searches
and axes) and writes a single report.json you can hand back for review. Each step
is wrapped so a failure still produces a partial report with the traceback.
"""
from __future__ import annotations
import json
import os
import sys
import traceback
from datetime import datetime, timezone

from . import config, db, scan, pipeline, semantic, assets

# Edit these to taste — they're trip-photo oriented.
QUERIES = [
    "golden hour landscape", "close-up of food", "people smiling",
    "architecture and buildings", "water or ocean", "a quiet empty street",
]
AXES = {
    "cold_to_warm": (["warm", "cozy", "sunset", "orange"],
                     ["cold", "icy", "blue", "overcast"]),
    "empty_to_busy": (["busy", "crowded", "cluttered"],
                      ["minimal", "empty", "simple"]),
    "nature_to_urban": (["city", "urban", "concrete", "buildings"],
                        ["nature", "forest", "landscape", "wilderness"]),
}


def _try(rep, key, fn):
    try:
        rep[key] = fn()
    except Exception as e:
        rep[key] = {"error": f"{type(e).__name__}: {e}",
                    "trace": traceback.format_exc().splitlines()[-4:]}
    return rep.get(key)


def run(lib_path: str) -> dict:
    lib = config.Library(lib_path)
    lib.ensure_dirs()
    con = db.connect(lib.db_path)
    rep = {"library": lib.root, "when": datetime.now(timezone.utc).isoformat()}

    _try(rep, "scan", lambda: scan.scan(lib, con))
    _try(rep, "thumbs", lambda: pipeline.make_thumbnails(lib, con))
    _try(rep, "embed", lambda: pipeline.embed_library(lib, con, prefer_clip=True))
    _try(rep, "groups", lambda: len(pipeline.group_library(lib, con)))
    _try(rep, "aesthetic_weights", lambda: assets.fetch_aesthetic(lib_path))
    _try(rep, "scored", lambda: pipeline.score_library(lib, con))

    names = {r["hash"]: r["filename"] for r in db.photos(con)}

    def searches():
        from .embed import get_embedder
        emb = get_embedder(prefer_clip=True)
        out = {}
        for q in QUERIES:
            try:
                out[q] = [(names.get(h, h), round(s, 3))
                          for h, s in semantic.search_text(lib, q, emb, k=8)]
            except Exception as e:
                out[q] = f"{type(e).__name__}: {e}"
        return out
    _try(rep, "searches", searches)

    def axes():
        from .embed import get_embedder
        emb = get_embedder(prefer_clip=True)
        out = {}
        for name, (pos, neg) in AXES.items():
            try:
                vec = semantic.axis_from_text(lib, pos, neg, emb)
                proj = semantic.project_axis(lib, vec, normalize="rank")
                ordered = sorted(proj.items(), key=lambda kv: kv[1])
                out[name] = {
                    f"low ({'+'.join(neg)})": [names.get(h, h) for h, _ in ordered[:5]],
                    f"high ({'+'.join(pos)})": [names.get(h, h) for h, _ in ordered[-5:]],
                }
            except Exception as e:
                out[name] = f"{type(e).__name__}: {e}"
        return out
    _try(rep, "axes", axes)

    def top_aes():
        rows = con.execute(
            "SELECT p.filename, s.aesthetic FROM photos p JOIN scores s USING(hash) "
            "WHERE s.aesthetic IS NOT NULL ORDER BY s.aesthetic DESC LIMIT 10").fetchall()
        return [(r[0], round(r[1], 3)) for r in rows]
    _try(rep, "top_aesthetic", top_aes)

    def stats():
        return {
            "photos": con.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
            "cameras": [tuple(r) for r in con.execute(
                "SELECT camera_model, COUNT(*) FROM photos GROUP BY camera_model")],
            "aesthetic_scored": con.execute(
                "SELECT COUNT(*) FROM scores WHERE aesthetic IS NOT NULL").fetchone()[0],
            "model": json.load(open(lib.model_path)) if os.path.exists(lib.model_path) else None,
        }
    _try(rep, "stats", stats)

    out_path = os.path.join(lib.cache_dir, "report.json")
    json.dump(rep, open(out_path, "w"), indent=2, default=str)
    return rep, out_path


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print(__doc__); sys.exit(1)
    rep, out_path = run(argv[0])
    print(json.dumps(rep, indent=2, default=str))
    print(f"\n--- wrote {out_path} ---")


if __name__ == "__main__":
    main()
