"""Engine tests for the highest-risk behaviors.

Run: pytest -q   (from the repo root, after `pip install -e .`)

These cover the two things most likely to cause silent data loss or wrong
results: non-destructive XMP writes, and stack/burst classification.
"""
import os
import tempfile

from autophotos import group, xmp

DARKTABLE_XMP = '''<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:darktable="http://darktable.sf.net/"
   xmp:Rating="0"
   darktable:history_end="8">
   <darktable:history>
    <rdf:Seq>
     <rdf:li darktable:num="0" darktable:operation="exposure"
       darktable:params="DEADBEEFCAFE"/>
    </rdf:Seq>
   </darktable:history>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
'''


def test_xmp_write_is_non_destructive():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "X.ARW")
        open(raw, "w").close()
        side = raw + ".xmp"
        open(side, "w").write(DARKTABLE_XMP)

        xmp.write_rating(raw, rating=5)
        out = open(side).read()

        assert xmp.read_rating(raw)["rating"] == 5
        assert 'xmp:Rating="5"' in out
        # darktable history + its binary params must survive untouched
        assert "DEADBEEFCAFE" in out
        assert 'darktable:operation="exposure"' in out
        assert 'darktable:history_end="8"' in out


def test_xmp_reject_uses_negative_rating():
    with tempfile.TemporaryDirectory() as d:
        raw = os.path.join(d, "Y.ARW")
        open(raw, "w").close()
        xmp.write_rating(raw, pick=-1)
        r = xmp.read_rating(raw)
        assert r["pick"] == -1


def _row(h, t, ev, orient=1, fd=None):
    return {"hash": h, "captured_at": t, "ev_bias": ev, "aperture": 8,
            "shutter": 0.01, "focus_dist": fd, "orientation": orient}


def test_bracket_vs_burst():
    rows = [_row("b0", "2025-01-01T10:00:00", -1.0),
            _row("b1", "2025-01-01T10:00:01", 0.0),
            _row("b2", "2025-01-01T10:00:02", 1.0)]
    rows += [_row(f"k{i}", f"2025-01-01T10:01:0{i}", 0.0) for i in range(5)]
    gs = {g.kind_hint for g in group.detect_groups(rows, emb=None)}
    assert "bracket" in gs
    assert "burst" in gs


def test_time_gap_splits_groups():
    # two clusters separated by a >3s gap
    rows = [_row(f"a{i}", f"2025-01-01T10:00:0{i}", 0.0) for i in range(3)]
    rows += [_row(f"c{i}", f"2025-01-01T10:05:0{i}", 0.0) for i in range(3)]
    gs = group.detect_groups(rows, emb=None)
    assert len(gs) == 2
