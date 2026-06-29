"""Crop / composition suggestions.

Default is a fast, torch-free heuristic: estimate a saliency map (gradient energy
+ mild center prior), then search candidate crops at common aspect ratios and
score them by how much saliency they keep and how well the salient centroid lands
on a rule-of-thirds intersection. A pluggable GAIC hook is used instead when a
torch-based model is available (set AUTOPHOTOS_GAIC=1 and provide gaic_model).

crop_suggestions(thumb_path) -> [{box:[x0,y0,x1,y1] normalized, aspect, score}]
"""
from __future__ import annotations
import numpy as np
from PIL import Image

ASPECTS = {"orig": None, "1:1": 1.0, "4:5": 0.8, "5:4": 1.25, "3:2": 1.5, "16:9": 16/9}
SCALES = (0.9, 0.78, 0.66)


def saliency(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    s = gx + gy
    # mild center prior
    h, w = s.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r2 = ((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2
    s = s * (1.0 - 0.3 * np.clip(r2, 0, 1))
    return s / (s.sum() + 1e-8)


def _integral(s):
    return np.pad(s, ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _winsum(I, y0, x0, y1, x1):
    return I[y1, x1] - I[y0, x1] - I[y1, x0] + I[y0, x0]


def crop_suggestions(thumb_path: str, n: int = 3, orig_ratio=None) -> list:
    im = Image.open(thumb_path).convert("L")
    s = saliency(np.asarray(im))
    H, W = s.shape
    I = _integral(s)
    thirds_x = [W / 3, 2 * W / 3]; thirds_y = [H / 3, 2 * H / 3]
    cand = []
    for name, ar in ASPECTS.items():
        if ar is None:
            ar = (orig_ratio or W / H)
        for sc in SCALES:
            cw = int(min(W, H * ar) * sc); ch = int(cw / ar)
            if cw < 16 or ch < 16 or cw > W or ch > H:
                continue
            step_x = max(1, (W - cw) // 6); step_y = max(1, (H - ch) // 6)
            for y0 in range(0, H - ch + 1, step_y):
                for x0 in range(0, W - cw + 1, step_x):
                    inside = _winsum(I, y0, x0, y0 + ch, x0 + cw)
                    # salient centroid within the crop
                    sub = s[y0:y0+ch, x0:x0+cw]
                    if sub.sum() <= 0:
                        continue
                    yy, xx = np.mgrid[0:ch, 0:cw]
                    my = y0 + (sub * yy).sum() / sub.sum()
                    mx = x0 + (sub * xx).sum() / sub.sum()
                    dthird = min(abs(mx - t) for t in thirds_x) / W + \
                             min(abs(my - t) for t in thirds_y) / H
                    score = float(inside) - 0.15 * dthird
                    cand.append((score, name, [x0/W, y0/H, (x0+cw)/W, (y0+ch)/H]))
    cand.sort(key=lambda c: c[0], reverse=True)
    # de-duplicate near-identical boxes
    out = []
    for sc, name, box in cand:
        if all(sum(abs(a-b) for a, b in zip(box, o["box"])) > 0.15 for o in out):
            out.append({"box": [round(v, 3) for v in box], "aspect": name,
                        "score": round(sc, 4)})
        if len(out) >= n:
            break
    return out


def gaic_suggestions(thumb_path, model, n=3):  # pragma: no cover - needs torch
    """Hook for a GAIC/ProCrop model. `model(image)->boxes,scores`."""
    boxes, scores = model(Image.open(thumb_path).convert("RGB"))
    order = np.argsort(scores)[::-1][:n]
    return [{"box": list(map(float, boxes[i])), "aspect": "gaic",
             "score": float(scores[i])} for i in order]
