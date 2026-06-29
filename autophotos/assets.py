"""Fetch + convert the LAION aesthetic predictor into our numpy head format.

Usage (on a machine with internet + torch):
    python -m autophotos.assets fetch-aesthetic /path/to/library

The LAION "improved-aesthetic-predictor" MLP has NO activations between its
linear layers, so at eval (dropout off) it is exactly a single affine map. We
fold the chain into one (W, b) and save it as <cache>/aesthetic_head.npz, which
score.AestheticHead loads with zero torch dependency at score time.

Expects embeddings from CLIP ViT-L/14 (768-d). Set:
    AUTOPHOTOS_EMBED_MODEL=ViT-L-14 AUTOPHOTOS_EMBED_PRETRAINED=openai
"""
from __future__ import annotations
import os
import sys
import urllib.request

import numpy as np

# Official improved-aesthetic-predictor linear-MLP weights (CLIP ViT-L/14).
LAION_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/"
             "raw/main/sac+logos+ava1-l14-linearMSE.pth")


def fetch_aesthetic(library: str) -> str:
    from .config import Library
    import torch

    lib = Library(library)
    lib.ensure_dirs()
    pth = os.path.join(lib.cache_dir, "aesthetic_predictor.pth")
    if not os.path.exists(pth):
        print(f"downloading {LAION_URL}")
        urllib.request.urlretrieve(LAION_URL, pth)

    sd = torch.load(pth, map_location="cpu")
    # collect linear layers in order: keys like 'layers.0.weight','layers.0.bias'
    idxs = sorted({int(k.split(".")[1]) for k in sd if k.endswith(".weight")})
    W_eff, b_eff = None, None
    for i in idxs:
        W = sd[f"layers.{i}.weight"].numpy().astype(np.float64)  # [out,in]
        b = sd[f"layers.{i}.bias"].numpy().astype(np.float64)    # [out]
        if W_eff is None:
            W_eff, b_eff = W, b
        else:  # compose affine maps (no activation between)
            W_eff, b_eff = W @ W_eff, W @ b_eff + b
    out = os.path.join(lib.cache_dir, "aesthetic_head.npz")
    np.savez(out, W0=W_eff.astype(np.float32), b0=b_eff.astype(np.float32))
    print(f"wrote {out}  (folded {len(idxs)} linear layers -> 1; "
          f"in={W_eff.shape[1]}, out={W_eff.shape[0]})")
    return out


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) == 2 and argv[0] == "fetch-aesthetic":
        fetch_aesthetic(argv[1])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
