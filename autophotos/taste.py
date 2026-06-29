"""Personalized aesthetics (PIAA) — learn *your* taste from your star ratings.

A ridge-regression head maps CLIP embeddings -> your rating (1..5). Pure numpy,
trained on whatever you've rated so far; re-fit anytime as more ratings arrive.
This is the personalized counterpart to the generic LAION aesthetic head, and it
directly addresses the LAION "landscape bias" by learning what *you* keep.

    w = (XᵀX + λI)⁻¹ Xᵀy     on L2-normalized embeddings (bias via augmented col)

Stored at <cache>/taste_head.npz. Predictions land in scores.personal.
"""
from __future__ import annotations
import json
import os

import numpy as np

from . import config, db


def _design(X):
    # augment with a bias column of ones
    return np.hstack([X, np.ones((X.shape[0], 1), np.float32)])


def train(lib: config.Library, con, l2: float = 1.0, min_labels: int = 8) -> dict:
    """Fit a ridge head from rated photos. Returns a small report."""
    emb_path, ids_path = lib.emb_path, lib.ids_path
    if not (os.path.exists(emb_path) and os.path.exists(ids_path)):
        return {"trained": False, "reason": "no embeddings; run embed first"}
    vecs = np.load(emb_path).astype(np.float32)
    ids = json.load(open(ids_path))
    idx = {h: i for i, h in enumerate(ids)}

    rows = con.execute("SELECT hash, rating FROM ratings WHERE rating IS NOT NULL AND rating>0")
    X, y = [], []
    for r in rows:
        if r["hash"] in idx:
            X.append(vecs[idx[r["hash"]]]); y.append(float(r["rating"]))
    n = len(y)
    if n < min_labels:
        return {"trained": False, "reason": f"need >= {min_labels} rated photos, have {n}"}

    X = _design(np.stack(X)); y = np.asarray(y, np.float32)
    d = X.shape[1]
    reg = l2 * np.eye(d, dtype=np.float32); reg[-1, -1] = 0.0  # don't regularize bias
    w = np.linalg.solve(X.T @ X + reg, X.T @ y).astype(np.float32)
    pred = X @ w
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    np.savez(os.path.join(lib.cache_dir, "taste_head.npz"), w=w)
    return {"trained": True, "n_labels": n, "rmse": round(rmse, 3),
            "rating_mean": round(float(y.mean()), 2)}


def _load(lib):
    p = os.path.join(lib.cache_dir, "taste_head.npz")
    return np.load(p)["w"] if os.path.exists(p) else None


def predict_all(lib: config.Library) -> dict:
    """Return {hash: personal_score} for the whole library, or {} if untrained."""
    w = _load(lib)
    if w is None or not os.path.exists(lib.emb_path):
        return {}
    vecs = np.load(lib.emb_path).astype(np.float32)
    ids = json.load(open(lib.ids_path))
    s = _design(vecs) @ w
    return {ids[i]: float(s[i]) for i in range(len(ids))}


def apply_scores(lib: config.Library, con) -> int:
    """Write personal scores into the scores table. Returns count."""
    preds = predict_all(lib)
    n = 0
    for h, v in preds.items():
        cur = con.execute("SELECT aesthetic, sharpness, exposure_ok FROM scores WHERE hash=?",
                          (h,)).fetchone()
        a = cur["aesthetic"] if cur else None
        sh = cur["sharpness"] if cur else None
        ex = cur["exposure_ok"] if cur else None
        from datetime import datetime, timezone
        db.upsert_score(con, h, a, sh, ex, v, datetime.now(timezone.utc).isoformat())
        n += 1
    con.commit()
    return n


def review_order(lib: config.Library, con):
    """Active-learning: unrated photos ranked by predicted taste (highest first),
    so your next culling pass surfaces likely-keepers early. Falls back to
    aesthetic when taste is untrained."""
    preds = predict_all(lib)
    rated = {r["hash"] for r in con.execute("SELECT hash FROM ratings WHERE rating>0")}
    names = {r["hash"]: r["filename"] for r in con.execute("SELECT hash, filename FROM photos")}
    if not preds:  # fall back to aesthetic
        preds = {r["hash"]: (r["aesthetic"] or 0.0)
                 for r in con.execute("SELECT hash, aesthetic FROM scores")}
    unrated = [(h, v) for h, v in preds.items() if h not in rated]
    unrated.sort(key=lambda kv: kv[1], reverse=True)
    return [(names.get(h, h), round(v, 3)) for h, v in unrated]
