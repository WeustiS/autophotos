"""decisions.json edit-queue + confirmed-group round-trips."""
import os, tempfile
from autophotos import decisions


def _p(): return os.path.join(tempfile.mkdtemp(), "decisions.json")


def test_queue_add_dedup_remove():
    p = _p()
    decisions.queue_add(p, ["a", "b", "a"])
    assert decisions.queue_list(p) == ["a", "b"]
    decisions.queue_add(p, ["b", "c"])
    assert decisions.queue_list(p) == ["a", "b", "c"]
    decisions.queue_remove(p, ["b"])
    assert decisions.queue_list(p) == ["a", "c"]


def test_confirm_group_persists():
    p = _p()
    decisions.confirm_group(p, ["x", "y", "z"], "focus", "assemble")
    g = decisions.groups_list(p)
    assert g[0]["members"] == ["x", "y", "z"]
    assert g[0]["action"] == "assemble" and g[0]["kind"] == "focus"


def test_survives_reload():
    p = _p()
    decisions.queue_add(p, ["a"]); decisions.confirm_group(p, ["a"], "burst", "keep-best")
    d = decisions.load(p)
    assert d["edit_queue"] == ["a"] and len(d["confirmed_groups"]) == 1
