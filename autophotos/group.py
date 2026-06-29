"""Stack / burst candidate detection.

Pipeline (validated on real data): sort by capture time -> cut into runs on
time gaps -> split runs on scene change (embedding similarity) -> classify a
kind_hint from EXIF. Output is *candidate* groups; the user adjudicates. We
deliberately over-group: a missed group is worse than a false one.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import config


@dataclass
class Group:
    group_id: str
    members: list          # list[hash], in capture order
    kind_hint: str
    t_start: str
    t_end: str
    confidence: float = 0.5
    evidence: dict = field(default_factory=dict)


def _parse(ts):
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def detect_groups(rows, emb: dict | None = None,
                  max_gap_s=config.MAX_GAP_S, min_n=config.MIN_GROUP_N,
                  sim_split=config.SIM_SPLIT) -> list[Group]:
    """rows: list of dict-like with hash, captured_at, ev_bias, aperture,
    shutter, focus_dist, orientation. emb: {hash: vec} or None."""
    items = [r for r in rows if _parse(r["captured_at"])]
    items.sort(key=lambda r: _parse(r["captured_at"]))

    # 1) time-gap runs
    runs, cur = [], []
    prev = None
    for r in items:
        ts = _parse(r["captured_at"])
        if prev is not None and (ts - prev).total_seconds() > max_gap_s:
            runs.append(cur); cur = []
        cur.append(r); prev = ts
    if cur:
        runs.append(cur)

    # 2) split on scene change via embedding similarity
    groups = []
    for run in runs:
        for sub in _split_on_similarity(run, emb, sim_split):
            if len(sub) >= min_n:
                groups.append(_make_group(sub))
    return groups


def _split_on_similarity(run, emb, sim_split):
    if not emb or len(run) < 2:
        return [run]
    out, cur = [], [run[0]]
    for prev, nxt in zip(run, run[1:]):
        a, b = emb.get(prev["hash"]), emb.get(nxt["hash"])
        sim = float(np.dot(a, b)) if a is not None and b is not None else 1.0
        if sim < sim_split:
            out.append(cur); cur = []
        cur.append(nxt)
    out.append(cur)
    return out


def _make_group(sub) -> Group:
    hashes = [r["hash"] for r in sub]
    t0, t1 = sub[0]["captured_at"], sub[-1]["captured_at"]
    gid = f"g_{hashes[0][:10]}_{len(hashes)}"
    kind, conf, ev = _classify(sub)
    return Group(gid, hashes, kind, t0, t1, conf, ev)


def _classify(sub):
    evs = [r.get("ev_bias") for r in sub if r.get("ev_bias") is not None]
    distinct_ev = sorted(set(round(e, 2) for e in evs))
    fds = [r.get("focus_dist") for r in sub if r.get("focus_dist") is not None]
    orients = set(r.get("orientation") for r in sub)

    # exposure bracket: >=3 distinct EV steps, exposure deliberately varied
    if len(distinct_ev) >= config.BRACKET_MIN_STEPS:
        return "bracket", 0.8, {"ev_steps": distinct_ev}

    # focus stack: monotonic focus distance (needs exiftool); stable exposure
    if len(fds) >= 3 and _monotonic(fds) and len(distinct_ev) <= 1:
        return "focus", 0.75, {"focus_dist": fds}

    # panorama: many frames, single orientation, no EV variation, longer span
    # (true directional-drift test belongs in the UI overlay; hint here)
    if len(sub) >= 4 and len(distinct_ev) <= 1 and len(orients) == 1:
        return "burst", 0.5, {"note": "could be pano; confirm in UI"}

    return "burst", 0.5, {}


def _monotonic(xs):
    inc = all(b >= a for a, b in zip(xs, xs[1:]))
    dec = all(b <= a for a, b in zip(xs, xs[1:]))
    return inc or dec
