"""EXIF extraction.

Pure-Python via exifread covers everything v1 needs (validated on the real
files). If the `exiftool` binary is present we enrich with Sony MakerNotes
(focus distance, drive mode) which materially help focus-stack detection.
"""
from __future__ import annotations
import json
import shutil
import subprocess
from datetime import datetime

import exifread


def _ratio(v):
    if v is None:
        return None
    s = str(v)
    try:
        if "/" in s:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except Exception:
        return None


def _int(v):
    try:
        return int(str(v))
    except Exception:
        return None


def read_exif(path: str) -> dict:
    with open(path, "rb") as fh:
        t = exifread.process_file(fh, details=False)

    def g(k):
        v = t.get(k)
        return str(v) if v is not None else None

    captured_at = None
    dt = g("EXIF DateTimeOriginal")
    if dt:
        try:
            captured_at = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S").isoformat()
        except Exception:
            captured_at = dt

    orient_map = {
        "Horizontal (normal)": 1, "Rotated 180": 3,
        "Rotated 90 CW": 6, "Rotated 90 CCW": 8,
    }
    out = {
        "captured_at": captured_at,
        "sub_sec": _int(g("EXIF SubSecTimeOriginal")),
        "camera_model": g("Image Model"),
        "lens": g("EXIF LensModel"),
        "focal_len": _ratio(g("EXIF FocalLength")),
        "aperture": _ratio(g("EXIF FNumber")),
        "shutter": _ratio(g("EXIF ExposureTime")),
        "iso": _int(g("EXIF ISOSpeedRatings")),
        "ev_bias": _ratio(g("EXIF ExposureBiasValue")),
        "focus_dist": None,
        "drive_mode": None,
        "orientation": orient_map.get(g("Image Orientation")),
        "width": _int(g("EXIF ExifImageWidth")),
        "height": _int(g("EXIF ExifImageLength")),
    }
    if _HAS_EXIFTOOL:
        out.update(_exiftool_enrich(path))
    return out


_HAS_EXIFTOOL = shutil.which("exiftool") is not None


def _exiftool_enrich(path: str) -> dict:
    """Sony MakerNotes that exifread can't read. Best-effort."""
    try:
        r = subprocess.run(
            ["exiftool", "-j", "-FocusDistance2", "-FocusDistance",
             "-DriveMode", "-SequenceNumber", path],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(r.stdout)[0]
        fd = data.get("FocusDistance2") or data.get("FocusDistance")
        return {
            "focus_dist": _ratio(str(fd).split()[0]) if fd else None,
            "drive_mode": data.get("DriveMode"),
        }
    except Exception:
        return {}
