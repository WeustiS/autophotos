"""Semantic search, custom axes, and 2D layout over cached embeddings.

This is the engine side of Stage 3/4. Everything operates on the cached
embeddings.npy + ids.json, so it's instant and needs no model at query time —
except text queries/axes, which need the embedder's text tower (CLIP). Exemplar
queries/axes (using image hashes as poles) work with ANY embedder, including the
non-semantic fallback, which is what lets us validate the plumbing without torch.
"""
from __future__ import annotations
import json
import os

import numpy as np

from . import config


def load_index(lib: config.Library):
    """Return (ids: list[str], mat: np.ndarray [N,D], meta: dict)."""
    if not (os.path.exists(lib.emb_path) and os.path.exists(lib.ids_path)):
        raise FileNotFoundError("no embeddings; run `autophotos embed` first")
    mat = np.load(lib.emb_path).astype(np.float32)
    ids = json.load(open(lib.ids_path))
    meta = json.load(open(lib.model_path)) if os.path.exists(lib.model_path) else {}
    return ids, mat, meta


def _rank(mat, query_vec, ids, k):
    sims = mat @ query_vec  # rows are L2-normalized
    order = np.argsort(-sims)[:k]
    return [(ids[i], float(sims[i])) for i in order]


def search_text(lib, query: str, embedder, k=20):
    """Rank photos by similarity to a text query (requires a semantic embedder)."""
    if not getattr(embedder, "semantic", False):
        raise RuntimeError("text search needs a semantic (CLIP) embedder")
    ids, mat, _ = load_index(lib)
    q = embedder.embed_texts([query])[0]
    return _rank(mat, q, ids, k)


def search_image(lib, hash_: str, k=20):
    """Rank photos by similarity to one photo (works with any embedder)."""
    ids, mat, _ = load_index(lib)
    if hash_ not in ids:
        raise KeyError(hash_)
    q = mat[ids.index(hash_)]
    return [r for r in _rank(mat, q, ids, k + 1) if r[0] != hash_][:k]


def axis_from_text(lib, pos: list[str], neg: list[str], embedder):
    """Direction vector for a semantic axis defined by text poles (needs CLIP)."""
    if not getattr(embedder, "semantic", False):
        raise RuntimeError("text axes need a semantic (CLIP) embedder")
    p = embedder.embed_texts(pos).mean(0)
    n = embedder.embed_texts(neg).mean(0)
    return _unit(p - n)


def axis_from_exemplars(lib, pos_hashes: list[str], neg_hashes: list[str]):
    """Direction vector from example photos (works with any embedder)."""
    ids, mat, _ = load_index(lib)
    idx = {h: i for i, h in enumerate(ids)}
    p = mat[[idx[h] for h in pos_hashes]].mean(0)
    n = mat[[idx[h] for h in neg_hashes]].mean(0)
    return _unit(p - n)


def project_axis(lib, axis_vec: np.ndarray, normalize="rank"):
    """Project every photo onto an axis. Returns {id: scalar}.

    normalize: 'rank' -> [0,1] by rank (even spread, great for layout);
               'z'    -> z-score; 'raw' -> raw dot product.
    """
    ids, mat, _ = load_index(lib)
    s = mat @ _unit(axis_vec)
    if normalize == "z":
        s = (s - s.mean()) / (s.std() + 1e-8)
    elif normalize == "rank":
        order = np.argsort(np.argsort(s))
        s = order / max(len(order) - 1, 1)
    return {ids[i]: float(s[i]) for i in range(len(ids))}


def layout_2d(lib):
    """PCA to 2D (numpy SVD, no sklearn). Returns {id: [x, y]} in [0,1]^2."""
    ids, mat, _ = load_index(lib)
    x = mat - mat.mean(0, keepdims=True)
    # top-2 right singular vectors
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    proj = x @ vt[:2].T
    # min-max each axis to [0,1] for direct thumbnail placement
    lo, hi = proj.min(0), proj.max(0)
    proj = (proj - lo) / (hi - lo + 1e-8)
    return {ids[i]: [float(proj[i, 0]), float(proj[i, 1])] for i in range(len(ids))}


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-8)
