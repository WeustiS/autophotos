"""decisions.json — source of truth for user choices not expressible in XMP.

Holds confirmed stack/pano groups and their chosen action, plus edit-queue
membership. Plain JSON: greppable, diffable, survives a full cache rebuild.
"""
from __future__ import annotations
import json
import os

DEFAULT = {"version": 1, "confirmed_groups": [], "edit_queue": []}


def load(path: str) -> dict:
    if not os.path.exists(path):
        return dict(DEFAULT)
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return dict(DEFAULT)


def save(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def confirm_group(path, members, kind, action):
    d = load(path)
    d["confirmed_groups"].append(
        {"members": list(members), "kind": kind, "action": action})
    save(path, d)
    return d
