"""decisions.json — source of truth for user choices not expressible in XMP.

Confirmed stack/pano groups (+ chosen action) and the edit queue. Plain JSON:
greppable, diffable, survives a full cache rebuild. Atomic writes.
"""
from __future__ import annotations
import copy
import json
import os

DEFAULT = {"version": 1, "confirmed_groups": [], "edit_queue": []}


def load(path: str) -> dict:
    if not os.path.exists(path):
        return copy.deepcopy(DEFAULT)
    try:
        d = json.load(open(path, encoding="utf-8"))
        for k, v in DEFAULT.items():
            d.setdefault(k, v if not isinstance(v, list) else [])
        return d
    except Exception:
        return copy.deepcopy(DEFAULT)


def save(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w", encoding="utf-8"), indent=2)
    os.replace(tmp, path)


# --- edit queue -----------------------------------------------------------

def queue_add(path, hashes):
    d = load(path)
    q = d["edit_queue"]
    for h in hashes:
        if h not in q:
            q.append(h)
    save(path, d)
    return q


def queue_remove(path, hashes):
    d = load(path)
    s = set(hashes)
    d["edit_queue"] = [h for h in d["edit_queue"] if h not in s]
    save(path, d)
    return d["edit_queue"]


def queue_list(path):
    return load(path)["edit_queue"]


# --- confirmed groups -----------------------------------------------------

def confirm_group(path, members, kind, action):
    """action in {assemble, keep-best, discard}; records a user decision."""
    d = load(path)
    d["confirmed_groups"].append(
        {"members": list(members), "kind": kind, "action": action})
    save(path, d)
    return d["confirmed_groups"]


def groups_list(path):
    return load(path)["confirmed_groups"]
