"""Semantic categories for the library (Stage 3).

Two complementary tools:
  - tag_library: zero-shot CLIP tags from a candidate vocabulary (needs CLIP)
  - discover: unsupervised k-means clusters over embeddings (works with any
    embedder); name clusters by their dominant zero-shot tag or a VLM caption.

The pure functions (assign_tags, kmeans) are model-free and unit-tested; the
library wrappers just supply embeddings/text vectors.
"""
from __future__ import annotations
import json
import os

import numpy as np

from . import config

DEFAULT_VOCAB = [
    "landscape", "portrait", "wildlife", "bird", "architecture", "street",
    "food", "macro", "water", "mountains", "forest", "sunset", "night",
    "snow", "flowers", "people", "animal", "vehicle", "beach", "desert",
    "indoor", "candid", "group photo", "close-up", "panorama",
]


def assign_tags(img: np.ndarray, tags: np.ndarray, names, topk=3, thresh=0.18):
    """img [N,D], tags [T,D] (all L2-normalized). -> list per image of (name,score)."""
    sims = img @ tags.T
    out = []
    for row in sims:
        order = np.argsort(-row)[:topk]
        out.append([(names[i], float(row[i])) for i in order if row[i] >= thresh])
    return out


def kmeans(X: np.ndarray, k: int, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    labels = np.zeros(len(X), int)
    for _ in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for j in range(len(c)):
            m = labels == j
            if m.any():
                c[j] = X[m].mean(0)
    return labels, c


def _load(lib):
    vecs = np.load(lib.emb_path).astype(np.float32)
    ids = json.load(open(lib.ids_path))
    meta = json.load(open(lib.model_path)) if os.path.exists(lib.model_path) else {}
    return ids, vecs, meta


def tag_library(lib: config.Library, embedder, vocab=None, topk=3, thresh=0.18):
    if not getattr(embedder, "semantic", False):
        raise RuntimeError("zero-shot tagging needs a semantic (CLIP) embedder")
    vocab = vocab or DEFAULT_VOCAB
    ids, vecs, _ = _load(lib)
    tvecs = embedder.embed_texts([f"a photo of {v}" for v in vocab])
    tags = assign_tags(vecs, tvecs, vocab, topk=topk, thresh=thresh)
    res = {ids[i]: tags[i] for i in range(len(ids))}
    json.dump(res, open(os.path.join(lib.cache_dir, "categories.json"), "w"), indent=2)
    return res


def discover(lib: config.Library, k=8):
    """k-means clusters over embeddings (no CLIP needed). -> {hash: cluster}."""
    ids, vecs, _ = _load(lib)
    labels, _ = kmeans(vecs, k)
    res = {ids[i]: int(labels[i]) for i in range(len(ids))}
    json.dump(res, open(os.path.join(lib.cache_dir, "clusters.json"), "w"), indent=2)
    return res


def caption_images(paths, model):  # pragma: no cover - needs a VLM
    """Hook for a VLM captioner (Qwen-VL/LLaVA/BLIP-2). model(path)->str."""
    return {p: model(p) for p in paths}
