"""autophotos CLI.

Index / cull:
  scan <lib> · thumbs <lib> · embed <lib> · group <lib> · score <lib>
  index <lib>            full pipeline (scan+thumbs+embed+group+score+xmp sync)
  stats <lib> · picks <lib>
  rate <raw> --stars N [--reject] [--label L]

Semantic (Stage 3/4):
  search <lib> "text"  · similar <lib> <hash>
  axis <lib> --pos a,b --neg c,d   · layout <lib>
"""
from __future__ import annotations
import argparse
import json
import os

from . import config, db, pipeline, xmp, semantic, pick
from . import scan as scan_mod


def _lib(p):
    lib = config.Library(p)
    lib.ensure_dirs()
    return lib, db.connect(lib.db_path)


def cmd_scan(a):
    lib, con = _lib(a.library)
    print(json.dumps(scan_mod.scan(lib, con), indent=2))


def cmd_thumbs(a):
    lib, con = _lib(a.library)
    print(f"thumbnails generated: {pipeline.make_thumbnails(lib, con)}")


def cmd_embed(a):
    lib, con = _lib(a.library)
    print(json.dumps(pipeline.embed_library(lib, con, prefer_clip=not a.no_clip), indent=2))


def cmd_group(a):
    lib, con = _lib(a.library)
    groups = pipeline.group_library(lib, con)
    print(f"{len(groups)} candidate groups:")
    for g in groups:
        print(f"  [{g.kind_hint:7}] n={len(g.members):3} {g.t_start}..{g.t_end} "
              f"conf={g.confidence} {g.evidence}")


def cmd_score(a):
    lib, con = _lib(a.library)
    print(f"scored: {pipeline.score_library(lib, con)}")


def cmd_rate(a):
    p = xmp.write_rating(a.raw, rating=a.stars,
                         pick=-1 if a.reject else None, label=a.label)
    print(f"wrote {p}: {xmp.read_rating(a.raw)}")


def cmd_index(a):
    lib, con = _lib(a.library)
    print("scan:", json.dumps(scan_mod.scan(lib, con)))
    print("thumbs:", pipeline.make_thumbnails(lib, con))
    print("embed:", json.dumps(pipeline.embed_library(lib, con, prefer_clip=not a.no_clip)))
    print("groups:", len(pipeline.group_library(lib, con)))
    print("score:", pipeline.score_library(lib, con))
    print("xmp sync:", pipeline.sync_ratings_from_xmp(lib, con))


def cmd_stats(a):
    lib, con = _lib(a.library)
    n = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    cams = con.execute("SELECT camera_model, COUNT(*) c FROM photos GROUP BY camera_model").fetchall()
    ng = con.execute("SELECT COUNT(*) FROM groups").fetchone()[0]
    rated = con.execute("SELECT COUNT(*) FROM ratings WHERE rating>0").fetchone()[0]
    aes = con.execute("SELECT COUNT(*) FROM scores WHERE aesthetic IS NOT NULL").fetchone()[0]
    print(f"library: {lib.root}")
    print(f"photos:  {n}")
    print(f"cameras: {[(c['camera_model'], c['c']) for c in cams]}")
    print(f"groups:  {ng}")
    print(f"rated:   {rated}")
    print(f"aesthetic scored: {aes}")
    if os.path.exists(lib.model_path):
        print("model:  ", json.load(open(lib.model_path)))


def cmd_picks(a):
    lib, con = _lib(a.library)
    print(json.dumps(pick.summary(con), indent=2))
    print("\ntop picks (best-of-group):")
    for i, p in enumerate(pick.top_picks(con, k=a.k), 1):
        tag = f"[group of {p['group_size']}]" if p["from_group"] else ""
        print(f"  {i:2}. {p['filename']}  aes={p['aesthetic']} sharp={p['sharpness']} {tag}")


def _names(lib, con, results):
    by = {r["hash"]: r["filename"] for r in db.photos(con)}
    for h, s in results:
        print(f"  {s:+.3f}  {by.get(h, h)}")


def cmd_search(a):
    lib, con = _lib(a.library)
    from .embed import get_embedder
    _names(lib, con, semantic.search_text(lib, a.query, get_embedder(prefer_clip=True), k=a.k))


def cmd_similar(a):
    lib, con = _lib(a.library)
    _names(lib, con, semantic.search_image(lib, a.hash, k=a.k))


def cmd_axis(a):
    lib, con = _lib(a.library)
    from .embed import get_embedder
    pos = a.pos.split(","); neg = a.neg.split(",")
    vec = semantic.axis_from_text(lib, pos, neg, get_embedder(prefer_clip=True))
    proj = semantic.project_axis(lib, vec, normalize="rank")
    by = {r["hash"]: r["filename"] for r in db.photos(con)}
    ordered = sorted(proj.items(), key=lambda kv: kv[1])
    print(f"axis  {neg} (low) <----> {pos} (high)")
    print("low:");  [print(f"  {v:.2f} {by.get(h,h)}") for h, v in ordered[:a.k]]
    print("high:"); [print(f"  {v:.2f} {by.get(h,h)}") for h, v in ordered[-a.k:]]


def cmd_layout(a):
    lib, con = _lib(a.library)
    coords = semantic.layout_2d(lib)
    out = os.path.join(lib.cache_dir, "layout2d.json")
    json.dump(coords, open(out, "w"))
    print(f"wrote {out} ({len(coords)} points)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="autophotos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn in [("scan", cmd_scan), ("thumbs", cmd_thumbs),
                     ("group", cmd_group), ("score", cmd_score),
                     ("stats", cmd_stats), ("layout", cmd_layout)]:
        s = sub.add_parser(name); s.add_argument("library"); s.set_defaults(fn=fn)

    s = sub.add_parser("picks")
    s.add_argument("library"); s.add_argument("-k", type=int, default=25)
    s.set_defaults(fn=cmd_picks)

    for name, fn in [("embed", cmd_embed), ("index", cmd_index)]:
        s = sub.add_parser(name); s.add_argument("library")
        s.add_argument("--no-clip", action="store_true"); s.set_defaults(fn=fn)

    r = sub.add_parser("rate")
    r.add_argument("raw"); r.add_argument("--stars", type=int, default=None)
    r.add_argument("--reject", action="store_true"); r.add_argument("--label", default=None)
    r.set_defaults(fn=cmd_rate)

    s = sub.add_parser("search")
    s.add_argument("library"); s.add_argument("query"); s.add_argument("-k", type=int, default=15)
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("similar")
    s.add_argument("library"); s.add_argument("hash"); s.add_argument("-k", type=int, default=15)
    s.set_defaults(fn=cmd_similar)

    s = sub.add_parser("axis")
    s.add_argument("library"); s.add_argument("--pos", required=True)
    s.add_argument("--neg", required=True); s.add_argument("-k", type=int, default=8)
    s.set_defaults(fn=cmd_axis)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
