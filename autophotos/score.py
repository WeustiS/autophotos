"""Technical heads (always) + LAION aesthetic head (CLIP only)."""
from __future__ import annotations
import os
import numpy as np
from PIL import Image


def sharpness(thumb_path: str) -> float:
    a = np.asarray(Image.open(thumb_path).convert("L"), dtype=np.float32)
    lap = (-4 * a + np.roll(a, 1, 0) + np.roll(a, -1, 0)
           + np.roll(a, 1, 1) + np.roll(a, -1, 1))
    return float(lap.var())


def exposure_ok(thumb_path: str) -> float:
    a = np.asarray(Image.open(thumb_path).convert("L"))
    return float(1.0 - np.mean((a < 4) | (a > 251)))


class AestheticHead:
    """MLP (or single linear) over normalized CLIP embeddings, pure numpy."""

    def __init__(self, layers):
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32))
                       for W, b in layers]
        self.dim = self.layers[0][0].shape[1]

    def __call__(self, emb: np.ndarray) -> float:
        x = emb.astype(np.float32)
        for i, (W, b) in enumerate(self.layers):
            x = W @ x + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
        return float(x.reshape(-1)[0])

    @classmethod
    def load(cls, cache_dir: str):
        p = os.path.join(cache_dir, "aesthetic_head.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p)
        n = sum(1 for k in d.files if k.startswith("W"))
        return cls([(d[f"W{i}"], d[f"b{i}"]) for i in range(n)])


_HEAD_CACHE = {}


def aesthetic(embedding, semantic: bool, cache_dir: str):
    if not semantic or embedding is None:
        return None
    if cache_dir not in _HEAD_CACHE:
        _HEAD_CACHE[cache_dir] = AestheticHead.load(cache_dir)
    head = _HEAD_CACHE[cache_dir]
    if head is None or head.dim != embedding.shape[-1]:
        return None
    return head(embedding)
