"""RAW preview + thumbnail handling.

Strategy (validated: embedded preview extracts in ~6 ms): use the embedded JPEG
for grid/fit views instantly; only decode the full RAW on demand (loupe/export).
"""
from __future__ import annotations
import io
import os

import numpy as np
import rawpy
from PIL import Image, ImageOps


def extract_preview(path: str) -> bytes | None:
    """Return the embedded JPEG preview bytes, or None if it's a bitmap thumb."""
    with rawpy.imread(path) as raw:
        try:
            thumb = raw.extract_thumb()
        except rawpy.LibRawNoThumbnailError:
            return None
    if thumb.format == rawpy.ThumbFormat.JPEG:
        return thumb.data
    # bitmap thumbnail -> encode to jpeg
    img = Image.fromarray(thumb.data)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def make_thumbs(path: str, out: dict) -> dict:
    """Write preview.jpg + 1024.jpg + 256.jpg. Returns produced paths.

    `out` is Library.thumb_paths(hash). Auto-orients via EXIF.
    """
    os.makedirs(out["dir"], exist_ok=True)
    data = extract_preview(path)
    if data is None:
        # fall back to a half-size RAW decode (slower, rare)
        arr = decode_full(path, half=True)
        base = Image.fromarray(arr)
    else:
        base = Image.open(io.BytesIO(data))
    base = ImageOps.exif_transpose(base).convert("RGB")

    with open(out["preview"], "wb") as f:
        base.save(f, "JPEG", quality=90)
    for size, key in ((1024, "1024"), (256, "256")):
        im = base.copy()
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(out[key], "JPEG", quality=88)
    return {k: out[k] for k in ("preview", "1024", "256")}


def decode_full(path: str, half: bool = False) -> np.ndarray:
    """On-demand full RAW develop (for 100% loupe / final export)."""
    with rawpy.imread(path) as raw:
        return raw.postprocess(
            half_size=half, use_camera_wb=True, no_auto_bright=False, output_bps=8,
        )
