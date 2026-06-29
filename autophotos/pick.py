"""Best-of-group ranking.

A burst clogs a naive top-N with near-duplicates. Collapse every candidate group
to its single best frame, then rank representatives + ungrouped singletons.

Quality key per photo: personal taste (PIAA) if trained, else generic aesthetic,
tie-broken by sharpness. So once you've rated enough, picks reflect *your* eye.
"""
from __future__ import annotations
from collections import defaultdict


def _rank_key(scores: dict):
    def key(h):
        s = scores.get(h, {})
        p = s.get("personal")
        a = s.get("aesthetic")
        sh = s.get("sharpness") or 0.0
        primary = p if p is not None else (a if a is not None else float("-inf"))
        return (primary, sh)
    return key


def representatives(con):
    gm = {r["hash"]: r["group_id"]
          for r in con.execute("SELECT hash, group_id FROM group_members")}
    scores = {r["hash"]: dict(r) for r in con.execute("SELECT * FROM scores")}
    photos = [r["hash"] for r in con.execute("SELECT hash FROM photos")]
    groups = defaultdict(list)
    singles = []
    for h in photos:
        g = gm.get(h)
        (groups[g].append(h) if g else singles.append(h))
    key = _rank_key(scores)
    reps = [(max(members, key=key), g, len(members)) for g, members in groups.items()]
    reps += [(h, None, 1) for h in singles]
    return reps, key, scores


def top_picks(con, k: int = 25):
    reps, key, scores = representatives(con)
    reps.sort(key=lambda r: key(r[0]), reverse=True)
    names = {r["hash"]: r["filename"] for r in con.execute("SELECT hash, filename FROM photos")}
    out = []
    for h, g, n in reps[:k]:
        s = scores.get(h, {})
        out.append({
            "filename": names.get(h, h),
            "from_group": g is not None,
            "group_size": n,
            "personal": round(s["personal"], 3) if s.get("personal") is not None else None,
            "aesthetic": round(s["aesthetic"], 3) if s.get("aesthetic") is not None else None,
            "sharpness": round(s["sharpness"], 1) if s.get("sharpness") is not None else None,
        })
    return out


def summary(con):
    reps, _, _ = representatives(con)
    n_groups = sum(1 for _, g, _ in reps if g is not None)
    total = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    return {"photos": total, "representatives": len(reps),
            "from_groups": n_groups, "singletons": len(reps) - n_groups,
            "collapsed_away": total - len(reps)}
