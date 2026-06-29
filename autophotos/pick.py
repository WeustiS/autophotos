"""Best-of-group ranking.

A burst clogs a naive top-N with near-duplicates (observed: the cedar-waxwing
burst filled all 8 top aesthetic slots). The fix: collapse every candidate group
to its single best frame, then rank those representatives alongside ungrouped
singletons. This connects grouping (group.py) with scoring (score.py).

Quality key per photo: prefer aesthetic score, tie-break on sharpness. Photos
with neither sort last but still appear.
"""
from __future__ import annotations
from collections import defaultdict


def _rank_key(scores: dict):
    def key(h):
        s = scores.get(h, {})
        a = s.get("aesthetic")
        sh = s.get("sharpness") or 0.0
        return (a if a is not None else float("-inf"), sh)
    return key


def representatives(con):
    """Return list of (hash, group_id_or_None, group_size) — one per group + singletons."""
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
    """Best-of-group representatives ranked by quality."""
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
            "aesthetic": round(s["aesthetic"], 3) if s.get("aesthetic") is not None else None,
            "sharpness": round(s["sharpness"], 1) if s.get("sharpness") is not None else None,
        })
    return out


def summary(con):
    reps, _, _ = representatives(con)
    n_groups = sum(1 for _, g, _ in reps if g is not None)
    n_singles = sum(1 for _, g, _ in reps if g is None)
    total = con.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    return {"photos": total, "representatives": len(reps),
            "from_groups": n_groups, "singletons": n_singles,
            "collapsed_away": total - len(reps)}
