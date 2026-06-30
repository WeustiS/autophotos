"""caption_library writes captions using an injected captioner (mock)."""
import json, os, tempfile
import numpy as np
from PIL import Image
from autophotos import config, categories


def test_caption_library_with_mock():
    d = tempfile.mkdtemp()
    os.environ["AUTOPHOTOS_CACHE_DIR"] = os.path.join(d, "cache")
    lib = config.Library(os.path.join(d, "lib")); lib.ensure_dirs()
    ids = ["h0", "h1"]
    np.save(lib.emb_path, np.zeros((2, 4), np.float32)); json.dump(ids, open(lib.ids_path, "w"))
    for h in ids:
        tp = lib.thumb_paths(h); os.makedirs(tp["dir"], exist_ok=True)
        Image.new("RGB", (16, 16)).save(tp["1024"])
    res = categories.caption_library(lib, captioner=lambda p: "a test caption")
    assert res == {"h0": "a test caption", "h1": "a test caption"}
    saved = json.load(open(os.path.join(lib.cache_dir, "captions.json")))
    assert saved == res
    os.environ.pop("AUTOPHOTOS_CACHE_DIR", None)


def test_caption_library_no_model_is_graceful():
    d = tempfile.mkdtemp()
    os.environ["AUTOPHOTOS_CACHE_DIR"] = os.path.join(d, "cache")
    lib = config.Library(os.path.join(d, "lib")); lib.ensure_dirs()
    json.dump(["h0"], open(lib.ids_path, "w"))
    np.save(lib.emb_path, np.zeros((1, 4), np.float32))
    # no captioner available -> empty dict, no crash
    import autophotos.categories as c
    res = c.caption_library(lib, captioner=None) if False else c.caption_library(lib, captioner=None)
    assert isinstance(res, dict)
    os.environ.pop("AUTOPHOTOS_CACHE_DIR", None)
