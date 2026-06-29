"""Paths, model ids, and tunable thresholds.

Embedding model is env-overridable:
    AUTOPHOTOS_EMBED_MODEL=ViT-L-14 AUTOPHOTOS_EMBED_PRETRAINED=openai
Cache can be relocated off a network/FUSE mount:
    AUTOPHOTOS_CACHE_DIR=/some/local/disk
"""
from __future__ import annotations
import os
from dataclasses import dataclass

RAW_EXTS = {".arw", ".raf", ".cr2", ".cr3", ".nef", ".dng", ".rw2", ".orf"}
VIDEO_EXTS = {".mp4", ".mov"}

MAX_GAP_S = 3.0
MIN_GROUP_N = 3
SIM_SPLIT = 0.80
BRACKET_MIN_STEPS = 3

EMBED_MODEL = os.environ.get("AUTOPHOTOS_EMBED_MODEL", "ViT-B-32")
EMBED_PRETRAINED = os.environ.get("AUTOPHOTOS_EMBED_PRETRAINED", "laion2b_s34b_b79k")


@dataclass
class Library:
    root: str

    def __post_init__(self):
        self.root = os.path.abspath(self.root)

    @property
    def state_dir(self) -> str:
        override = os.environ.get("AUTOPHOTOS_CACHE_DIR")
        if override:
            key = self.root.replace(os.sep, "_").replace(":", "").strip("_")
            return os.path.join(os.path.abspath(override), key)
        return os.path.join(self.root, ".autophotos")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.state_dir, "cache")

    @property
    def db_path(self) -> str:
        return os.path.join(self.cache_dir, "index.sqlite")

    @property
    def emb_path(self) -> str:
        return os.path.join(self.cache_dir, "embeddings.npy")

    @property
    def ids_path(self) -> str:
        return os.path.join(self.cache_dir, "ids.json")

    @property
    def thumb_dir(self) -> str:
        return os.path.join(self.cache_dir, "thumb")

    @property
    def model_path(self) -> str:
        return os.path.join(self.cache_dir, "model.json")

    @property
    def decisions_path(self) -> str:
        return os.path.join(self.state_dir, "decisions.json")

    def ensure_dirs(self):
        for d in (self.state_dir, self.cache_dir, self.thumb_dir):
            os.makedirs(d, exist_ok=True)

    def thumb_paths(self, hash_: str) -> dict:
        d = os.path.join(self.thumb_dir, hash_)
        return {
            "dir": d,
            "256": os.path.join(d, "256.jpg"),
            "1024": os.path.join(d, "1024.jpg"),
            "preview": os.path.join(d, "preview.jpg"),
        }
