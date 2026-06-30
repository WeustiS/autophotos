"""Crop / composition suggestions.

Three tiers, best-available wins:
  1. GAIC/ProCrop hook  - a real learned cropper, if you wire one (gaic_suggestions)
  2. aesthetic-guided    - generate candidate boxes, then *score each crop with the
                           CLIP + LAION aesthetic head you already have* and keep the
                           best. This is a real learned selector, no extra download.
  3. heuristic           - saliency + rule-of-thirds, torch-free fallback.

suggest_crops(...) picks the highest available tier automatically.
Boxes are normalized [x0,y0,x1,y1].
"""
from __future__ import annotations
import os
import tempfile

import numpy as np
from PIL import Image

ASPECTS = {"orig": None, "1:1": 1.0, "4:5": 0.8, "5:4": 1.25, "3:2": 1.5, "16:9": 16/9}
SCALES = (0.92, 0.82, 0.7, 0.6)


def saliency(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    gx = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
    gy = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
    s = gx + gy
    h, w = s.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r2 = ((yy-(h-1)/2)/(h/2))**2 + ((xx-(w-1)/2)/(w/2))**2
    return (s * (1.0 - 0.3*np.clip(r2, 0, 1))) / (s.sum() + 1e-8)


def _integral(s): return np.pad(s, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
def _winsum(I, y0, x0, y1, x1): return I[y1, x1]-I[y0, x1]-I[y1, x0]+I[y0, x0]


def _candidates(thumb_path, orig_ratio=None, limit=16):
    """Heuristic candidate boxes scored by retained saliency + thirds bonus."""
    im = Image.open(thumb_path).convert("L")
    s = saliency(np.asarray(im)); H, W = s.shape; I = _integral(s)
    tx = [W/3, 2*W/3]; ty = [H/3, 2*H/3]
    cand = []
    for name, ar in ASPECTS.items():
        a = (orig_ratio or W/H) if ar is None else ar
        for sc in SCALES:
            cw = int(min(W, H*a)*sc); ch = int(cw/a)
            if cw < 16 or ch < 16 or cw > W or ch > H:
                continue
            sx = max(1, (W-cw)//6); sy = max(1, (H-ch)//6)
            for y0 in range(0, H-ch+1, sy):
                for x0 in range(0, W-cw+1, sx):
                    sub = s[y0:y0+ch, x0:x0+cw]
                    tot = sub.sum()
                    if tot <= 0:
                        continue
                    yy, xx = np.mgrid[0:ch, 0:cw]
                    my = y0+(sub*yy).sum()/tot; mx = x0+(sub*xx).sum()/tot
                    d3 = min(abs(mx-t) for t in tx)/W + min(abs(my-t) for t in ty)/H
                    cand.append((float(_winsum(I, y0, x0, y0+ch, x0+cw)-0.15*d3),
                                 name, [x0/W, y0/H, (x0+cw)/W, (y0+ch)/H]))
    cand.sort(key=lambda c: c[0], reverse=True)
    out = []
    for sc, name, box in cand:
        if all(sum(abs(a-b) for a, b in zip(box, o[2])) > 0.12 for o in out):
            out.append((sc, name, box))
        if len(out) >= limit:
            break
    return out


def crop_suggestions(thumb_path, n=3, orig_ratio=None):
    """Heuristic-only suggestions (no model)."""
    return [{"box": [round(v, 3) for v in box], "aspect": name,
             "score": round(sc, 4), "method": "heuristic"}
            for sc, name, box in _candidates(thumb_path, orig_ratio)[:n]]


def aesthetic_rerank(thumb_path, candidates, embedder, head, taste_w=None):
    """Crop each candidate, embed with CLIP, score with the aesthetic head
    (+ optional personal taste vector). Returns candidates sorted by model score."""
    im = Image.open(thumb_path).convert("RGB"); W, H = im.size
    tmp = tempfile.mkdtemp(); paths = []
    for i, (_, _, box) in enumerate(candidates):
        x0, y0, x1, y1 = box
        c = im.crop((int(x0*W), int(y0*H), int(x1*W), int(y1*H)))
        p = os.path.join(tmp, f"{i}.jpg"); c.save(p, "JPEG", quality=90); paths.append(p)
    embs = embedder.embed_images(paths)
    scored = []
    for i, (_, name, box) in enumerate(candidates):
        e = embs[i]
        sc = head(e) if (head is not None and head.dim == e.shape[-1]) else 0.0
        if taste_w is not None and taste_w.shape[0] == e.shape[-1] + 1:
            sc += float(np.append(e, 1.0) @ taste_w)
        scored.append({"box": [round(v, 3) for v in box], "aspect": name,
                       "score": round(float(sc), 4), "method": "aesthetic"})
    scored.sort(key=lambda c: c["score"], reverse=True)
    for p in paths:
        try: os.remove(p)
        except OSError: pass
    return scored


def suggest_crops(thumb_path, n=3, orig_ratio=None, embedder=None, head=None,
                  taste_w=None, gaic_model=None):
    if gaic_model is not None:
        return gaic_suggestions(thumb_path, gaic_model, n)  # pragma: no cover
    if embedder is not None and getattr(embedder, "semantic", False) and head is not None:
        cands = _candidates(thumb_path, orig_ratio, limit=16)
        return aesthetic_rerank(thumb_path, cands, embedder, head, taste_w)[:n]
    return crop_suggestions(thumb_path, n, orig_ratio)


def gaic_suggestions(thumb_path, model, n=3):  # pragma: no cover - needs torch model
    boxes, scores = model(Image.open(thumb_path).convert("RGB"))
    order = np.argsort(scores)[::-1][:n]
    return [{"box": list(map(float, boxes[i])), "aspect": "gaic",
             "score": float(scores[i]), "method": "gaic"} for i in order]
