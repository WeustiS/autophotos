"""Pluggable image/text embedders. CLIP backbone + non-semantic fallback."""
from __future__ import annotations
from typing import Protocol
import numpy as np
from PIL import Image
from . import config


def _l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


class Embedder(Protocol):
    id: str
    dim: int
    semantic: bool
    def embed_images(self, paths): ...
    def embed_texts(self, texts): ...


class FallbackEmbedder:
    """Downscaled color grid + gradient energy. NOT semantic (no text tower)."""
    id = "fallback-colorgrad-v1"
    dim = 8 * 8 * 3 + 8 * 8
    semantic = False

    def embed_images(self, paths):
        out = []
        for p in paths:
            im = Image.open(p).convert("RGB").resize((64, 64))
            a = np.asarray(im, dtype=np.float32) / 255.0
            color = np.asarray(im.resize((8, 8)), np.float32).reshape(-1) / 255.0
            gray = a.mean(2)
            gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
            gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
            ge = (gx + gy).reshape(8, 8, 8, 8).mean(axis=(1, 3)).reshape(-1)
            out.append(np.concatenate([color, ge]))
        return _l2(np.stack(out)).astype(np.float32)

    def embed_texts(self, texts):
        raise NotImplementedError("FallbackEmbedder has no text tower")


class ClipEmbedder:
    """open_clip backbone. Forces QuickGELU for OpenAI weights (correctness)."""
    semantic = True

    def __init__(self, model=None, pretrained=None):
        import torch, open_clip
        model = model or config.EMBED_MODEL
        pretrained = pretrained or config.EMBED_PRETRAINED
        self.torch = torch
        force_qg = (pretrained or "").lower() == "openai"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model, pretrained=pretrained, force_quick_gelu=force_qg)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.id = f"open_clip/{model}/{pretrained}" + ("+qg" if force_qg else "")
        with torch.no_grad():
            d = self.model.encode_image(
                self.preprocess(Image.new("RGB", (64, 64))).unsqueeze(0))
        self.dim = d.shape[-1]

    def embed_images(self, paths, batch=16):
        t = self.torch
        vecs = []
        for i in range(0, len(paths), batch):
            ims = [self.preprocess(Image.open(p).convert("RGB"))
                   for p in paths[i:i + batch]]
            with t.no_grad():
                v = self.model.encode_image(t.stack(ims)).cpu().numpy()
            vecs.append(v)
        return _l2(np.concatenate(vecs)).astype(np.float32)

    def embed_texts(self, texts):
        t = self.torch
        with t.no_grad():
            v = self.model.encode_text(self.tokenizer(texts)).cpu().numpy()
        return _l2(v).astype(np.float32)


def get_embedder(prefer_clip: bool = True) -> Embedder:
    if prefer_clip:
        try:
            return ClipEmbedder()
        except Exception as e:
            print(f"[embed] CLIP unavailable ({type(e).__name__}: {e}); "
                  f"using FallbackEmbedder (non-semantic).")
    return FallbackEmbedder()
