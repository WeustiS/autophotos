"""autophotos CLI.

Index/cull:  scan thumbs embed group score index stats picks  + rate
Taste:       train-taste <lib>   review <lib>
Compose:     crops <lib> <hash>
Semantic:    search similar axis layout tag cluster
Workflow:    queue <lib> {add|rm|list} [hash...]   confirm-group <lib> ...
"""
from __future__ import annotations
import argparse, json, os

from . import config, db, pipeline, xmp, semantic, pick, taste, crop, categories, decisions
from . import scan as scan_mod


def _lib(p):
    lib = config.Library(p); lib.ensure_dirs()
    return lib, db.connect(lib.db_path)


def cmd_scan(a):
    lib, con = _lib(a.library); print(json.dumps(scan_mod.scan(lib, con), indent=2))

def cmd_thumbs(a):
    lib, con = _lib(a.library); print(f"thumbnails: {pipeline.make_thumbnails(lib, con)}")

def cmd_embed(a):
    lib, con = _lib(a.library)
    print(json.dumps(pipeline.embed_library(lib, con, prefer_clip=not a.no_clip), indent=2))

def cmd_group(a):
    lib, con = _lib(a.library)
    gs = pipeline.group_library(lib, con); print(f"{len(gs)} candidate groups:")
    for g in gs:
        print(f"  [{g.kind_hint:7}] n={len(g.members):3} {g.t_start}..{g.t_end} {g.evidence}")

def cmd_score(a):
    lib, con = _lib(a.library); print(f"scored: {pipeline.score_library(lib, con)}")

def cmd_rate(a):
    p = xmp.write_rating(a.raw, rating=a.stars, pick=-1 if a.reject else None, label=a.label)
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
    print(f"library: {lib.root}\nphotos:  {n}")
    print(f"cameras: {[(c['camera_model'], c['c']) for c in cams]}")
    print(f"groups:  {con.execute('SELECT COUNT(*) FROM groups').fetchone()[0]}")
    print(f"rated:   {con.execute('SELECT COUNT(*) FROM ratings WHERE rating>0').fetchone()[0]}")
    print(f"aesthetic: {con.execute('SELECT COUNT(*) FROM scores WHERE aesthetic IS NOT NULL').fetchone()[0]}")
    print(f"personal:  {con.execute('SELECT COUNT(*) FROM scores WHERE personal IS NOT NULL').fetchone()[0]}")
    if os.path.exists(lib.model_path): print("model:  ", json.load(open(lib.model_path)))

def cmd_picks(a):
    lib, con = _lib(a.library)
    print(json.dumps(pick.summary(con), indent=2)); print("\ntop picks:")
    for i, p in enumerate(pick.top_picks(con, k=a.k), 1):
        tag = f"[group of {p['group_size']}]" if p["from_group"] else ""
        print(f"  {i:2}. {p['filename']}  personal={p['personal']} aes={p['aesthetic']} sharp={p['sharpness']} {tag}")

def cmd_train_taste(a):
    lib, con = _lib(a.library)
    rep = taste.train(lib, con, l2=a.l2)
    print(json.dumps(rep, indent=2))
    if rep.get("trained"):
        print("applied personal scores:", taste.apply_scores(lib, con))

def cmd_review(a):
    lib, con = _lib(a.library)
    order = taste.review_order(lib, con)
    print(f"{len(order)} unrated, ranked by predicted taste:")
    for fn, v in order[:a.k]:
        print(f"  {v:+.3f}  {fn}")

def cmd_crops(a):
    lib, con = _lib(a.library)
    row = con.execute("SELECT width,height FROM photos WHERE hash=?", (a.hash,)).fetchone()
    ratio = (row["width"]/row["height"]) if row and row["width"] and row["height"] else None
    tp = lib.thumb_paths(a.hash)["1024"]
    for c in crop.crop_suggestions(tp, n=a.k, orig_ratio=ratio):
        print(f"  {c['aspect']:5} score={c['score']:.4f} box={c['box']}")

def _names(lib, con, results):
    by = {r["hash"]: r["filename"] for r in db.photos(con)}
    for h, s in results: print(f"  {s:+.3f}  {by.get(h, h)}")

def cmd_search(a):
    lib, con = _lib(a.library)
    from .embed import get_embedder
    _names(lib, con, semantic.search_text(lib, a.query, get_embedder(True), k=a.k))

def cmd_similar(a):
    lib, con = _lib(a.library); _names(lib, con, semantic.search_image(lib, a.hash, k=a.k))

def cmd_axis(a):
    lib, con = _lib(a.library)
    from .embed import get_embedder
    pos, neg = a.pos.split(","), a.neg.split(",")
    vec = semantic.axis_from_text(lib, pos, neg, get_embedder(True))
    proj = semantic.project_axis(lib, vec, normalize="rank")
    by = {r["hash"]: r["filename"] for r in db.photos(con)}
    o = sorted(proj.items(), key=lambda kv: kv[1])
    print(f"{neg} (low) <--> {pos} (high)")
    print("low: ", [by.get(h, h) for h, _ in o[:a.k]])
    print("high:", [by.get(h, h) for h, _ in o[-a.k:]])

def cmd_layout(a):
    lib, con = _lib(a.library)
    out = os.path.join(lib.cache_dir, "layout2d.json")
    json.dump(semantic.layout_2d(lib), open(out, "w")); print("wrote", out)

def cmd_tag(a):
    lib, con = _lib(a.library)
    from .embed import get_embedder
    res = categories.tag_library(lib, get_embedder(True))
    print(f"tagged {len(res)} photos -> {lib.cache_dir}/categories.json")

def cmd_cluster(a):
    lib, con = _lib(a.library)
    res = categories.discover(lib, k=a.k)
    from collections import Counter
    print("cluster sizes:", dict(Counter(res.values())))

def cmd_queue(a):
    lib, con = _lib(a.library); p = lib.decisions_path
    if a.op == "add": print("queue:", decisions.queue_add(p, a.hashes))
    elif a.op == "rm": print("queue:", decisions.queue_remove(p, a.hashes))
    else: print("queue:", decisions.queue_list(p))

def cmd_confirm_group(a):
    lib, con = _lib(a.library)
    g = decisions.confirm_group(lib.decisions_path, a.members.split(","), a.kind, a.action)
    print(f"confirmed groups: {len(g)}")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="autophotos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    def P(name, fn, lib=True):
        s = sub.add_parser(name)
        if lib: s.add_argument("library")
        s.set_defaults(fn=fn); return s
    for n, f in [("scan", cmd_scan), ("thumbs", cmd_thumbs), ("group", cmd_group),
                 ("score", cmd_score), ("stats", cmd_stats), ("layout", cmd_layout)]:
        P(n, f)
    P("picks", cmd_picks).add_argument("-k", type=int, default=25)
    P("tag", cmd_tag)
    P("cluster", cmd_cluster).add_argument("-k", type=int, default=8)
    P("review", cmd_review).add_argument("-k", type=int, default=30)
    s = P("train-taste", cmd_train_taste); s.add_argument("--l2", type=float, default=1.0)
    s = P("crops", cmd_crops); s.add_argument("hash"); s.add_argument("-k", type=int, default=3)
    for n, f in [("embed", cmd_embed), ("index", cmd_index)]:
        P(n, f).add_argument("--no-clip", action="store_true")
    r = P("rate", cmd_rate, lib=False); r.add_argument("raw")
    r.add_argument("--stars", type=int); r.add_argument("--reject", action="store_true"); r.add_argument("--label")
    s = P("search", cmd_search); s.add_argument("query"); s.add_argument("-k", type=int, default=15)
    s = P("similar", cmd_similar); s.add_argument("hash"); s.add_argument("-k", type=int, default=15)
    s = P("axis", cmd_axis); s.add_argument("--pos", required=True); s.add_argument("--neg", required=True); s.add_argument("-k", type=int, default=8)
    s = P("queue", cmd_queue); s.add_argument("op", choices=["add", "rm", "list"]); s.add_argument("hashes", nargs="*")
    s = P("confirm-group", cmd_confirm_group); s.add_argument("--members", required=True); s.add_argument("--kind", default="burst"); s.add_argument("--action", default="keep-best")
    a = ap.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    main()
