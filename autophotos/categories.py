"""Semantic categories + captions for the library (Stage 3).

  - tag_library: zero-shot CLIP tags from a vocabulary (needs CLIP)
  - discover: unsupervised k-means clusters over embeddings (any embedder)
  - caption_library: real image captions via a BLIP model (transformers); writes
    captions.json. Captions double as alt-text, searchable metadata, and cluster
    labels. Graceful no-op if transformers/weights aren't installed.

Pure functions (assign_tags, kmeans) are model-free and unit-tested.
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


def assign_tags(img, tags, names, topk=3, thresh=0.18):
    sims = img @ tags.T
    out = []
    for row in sims:
        order = np.argsort(-row)[:topk]
        out.append([(names[i], float(row[i])) for i in order if row[i] >= thresh])
    return out


def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    c = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    labels = np.zeros(len(X), int)
    for it in range(iters):
        d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        new = d.argmin(1)
        if it > 0 and np.array_equal(new, labels):
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
    return ids, vecs


def tag_library(lib, embedder, vocab=None, topk=3, thresh=0.18):
    if not getattr(embedder, "semantic", False):
        raise RuntimeError("zero-shot tagging needs a semantic (CLIP) embedder")
    vocab = vocab or DEFAULT_VOCAB
    ids, vecs = _load(lib)
    tvecs = embedder.embed_texts([f"a photo of {v}" for v in vocab])
    res = {ids[i]: t for i, t in enumerate(assign_tags(vecs, tvecs, vocab, topk, thresh))}
    json.dump(res, open(os.path.join(lib.cache_dir, "categories.json"), "w"), indent=2)
    return res


def discover(lib, k=8):
    ids, vecs = _load(lib)
    labels, _ = kmeans(vecs, k)
    res = {ids[i]: int(labels[i]) for i in range(len(ids))}
    json.dump(res, open(os.path.join(lib.cache_dir, "clusters.json"), "w"), indent=2)
    return res


# --- captions -------------------------------------------------------------

class BlipCaptioner:
    """BLIP image captioner (transformers). CPU is fine for a few thousand thumbs.

    model id default: Salesforce/blip-image-captioning-base (downloaded once).
    """
    def __init__(self, model_id="Salesforce/blip-image-captioning-base"):
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.torch = torch
        self.proc = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id).eval()
        self.id = model_id

    def __call__(self, path: str) -> str:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        inp = self.proc(im, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=30)
        return self.proc.decode(out[0], skip_special_tokens=True).strip()


def load_captioner():
    """Return a BlipCaptioner, or None if transformers/torch/weights unavailable."""
    try:
        return BlipCaptioner()
    except Exception as e:
        print(f"[caption] captioner unavailable ({type(e).__name__}: {e})")
        return None


def caption_library(lib: config.Library, captioner=None) -> dict:
    """Caption every photo from its 1024 thumb; writes captions.json. Returns map."""
    captioner = captioner or load_captioner()
    if captioner is None:
        return {}
    ids = json.load(open(lib.ids_path))
    out = {}
    for h in ids:
        tp = lib.thumb_paths(h)["1024"]
        if os.path.exists(tp):
            try:
                out[h] = captioner(tp)
            except Exception as e:
                out[h] = ""
                print(f"[caption] {h[:8]}: {e}")
    json.dump(out, open(os.path.join(lib.cache_dir, "captions.json"), "w"), indent=2)
    return out
