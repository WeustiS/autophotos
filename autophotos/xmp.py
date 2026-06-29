"""Non-destructive XMP sidecar read/write.

The test library uses darktable sidecars (`<raw>.ARW.xmp`) that carry a full edit
history alongside `xmp:Rating`. We MUST preserve that history, so writes are done
as targeted attribute edits on the raw text (not a full XML re-serialize, which
would risk reformatting or dropping darktable nodes).

Conventions (standard, darktable/Lightroom-compatible):
  - stars      -> xmp:Rating="0".."5"
  - reject     -> xmp:Rating="-1"   (widely understood "rejected" convention)
  - color label-> xmp:Label="Red" (etc.)
We expose pick as: pick=1 -> keep stars as-is; pick=-1 -> rating -1 (reject).
"""
from __future__ import annotations
import os
import re
from datetime import datetime

_MINIMAL = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="autophotos">\n'
    ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
    '  <rdf:Description rdf:about=""\n'
    '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"\n'
    '    xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"\n'
    '   xmp:Rating="{rating}"\n'
    '   xmpMM:DerivedFrom="{src}"/>\n'
    ' </rdf:RDF>\n'
    '</x:xmpmeta>\n'
)


def sidecar_path(raw_path: str, create: bool = False) -> str | None:
    """Find an existing sidecar; if create=True return the darktable-style path."""
    cands = [raw_path + ".xmp", os.path.splitext(raw_path)[0] + ".xmp"]
    for c in cands:
        if os.path.exists(c):
            return c
    return (raw_path + ".xmp") if create else None


def read_rating(raw_path: str) -> dict:
    p = sidecar_path(raw_path)
    if not p:
        return {"rating": None, "pick": None, "label": None, "xmp_path": None}
    txt = open(p, "r", encoding="utf-8", errors="replace").read()
    rating = _attr(txt, "xmp:Rating")
    label = _attr_str(txt, "xmp:Label")
    r = int(rating) if rating is not None else None
    pick = -1 if r == -1 else None
    return {"rating": max(r, 0) if r is not None else None,
            "pick": pick, "label": label, "xmp_path": p}


def write_rating(raw_path: str, rating: int | None = None,
                 pick: int | None = None, label: str | None = None) -> str:
    """Set only rating/label, preserving every other byte of an existing sidecar."""
    eff_rating = rating
    if pick == -1:
        eff_rating = -1
    p = sidecar_path(raw_path, create=True)

    if not os.path.exists(p):
        with open(p, "w", encoding="utf-8") as f:
            f.write(_MINIMAL.format(rating=eff_rating if eff_rating is not None else 0,
                                    src=os.path.basename(raw_path)))
        if label is not None:
            _set_in_description(p, "xmp:Label", label, quoted=True)
        return p

    if eff_rating is not None:
        _set_in_description(p, "xmp:Rating", str(eff_rating), quoted=False)
    if label is not None:
        _set_in_description(p, "xmp:Label", label, quoted=True)
    return p


# --- internals: operate only on the rdf:Description start tag ---

def _attr(txt, name):
    m = re.search(rf'{re.escape(name)}="(-?\d+)"', txt)
    return m.group(1) if m else None


def _attr_str(txt, name):
    m = re.search(rf'{re.escape(name)}="([^"]*)"', txt)
    return m.group(1) if m else None


def _set_in_description(path, name, value, quoted):
    txt = open(path, "r", encoding="utf-8").read()
    val = value  # both numeric and string are written as "value"
    if re.search(rf'{re.escape(name)}="[^"]*"', txt):
        new = re.sub(rf'{re.escape(name)}="[^"]*"', f'{name}="{val}"', txt, count=1)
    else:
        # insert into the <rdf:Description ...> open tag, before its closing '>'
        m = re.search(r'<rdf:Description\b', txt)
        if not m:
            raise ValueError("no rdf:Description in sidecar")
        close = txt.index(">", m.end())
        new = txt[:close] + f'\n   {name}="{val}"' + txt[close:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(new)
