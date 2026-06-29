"""Galaxy export produces a self-contained gallery with PCA axes."""
import json, os, tempfile
import numpy as np
from PIL import Image
from autophotos import config, export


def test_build_gallery():
    d = tempfile.mkdtemp()
    os.environ["AUTOPHOTOS_CACHE_DIR"] = os.path.join(d, "cache")
    lib = config.Library(os.path.join(d, "lib")); lib.ensure_dirs()
    n, dim = 12, 16
    rng = np.random.default_rng(0)
    V = rng.standard_normal((n, dim)).astype(np.float32)
    ids = [f"h{i}" for i in range(n)]
    np.save(lib.emb_path, V); json.dump(ids, open(lib.ids_path, "w"))
    json.dump({"semantic": False, "dim": dim}, open(lib.model_path, "w"))
    for h in ids:  # a tiny thumbnail per hash
        tp = lib.thumb_paths(h); os.makedirs(tp["dir"], exist_ok=True)
        Image.new("RGB", (32, 32), (100, 120, 140)).save(tp["256"])
    out, count = export.build(lib.root)
    assert count == n and os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert "autophotos galaxy" in html and "const DATA=" in html
    # the data blob carries all items
    blob = html.split("const DATA=", 1)[1].split(";\n", 1)[0]
    data = json.loads(blob)
    assert data["count"] == n and data["axes"] == []  # no CLIP -> no named axes
    os.environ.pop("AUTOPHOTOS_CACHE_DIR", None)
