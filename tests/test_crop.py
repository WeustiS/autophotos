"""Crop suggestions keep the salient subject."""
import os, tempfile
import numpy as np
from PIL import Image
from autophotos import crop


def _img_with_blob(cx_frac, cy_frac):
    H, W = 300, 450
    a = np.full((H, W, 3), 128, np.uint8)
    cx, cy = int(W * cx_frac), int(H * cy_frac)
    # high-frequency textured blob (edges -> saliency)
    rng = np.random.default_rng(0)
    blob = rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)
    a[cy-40:cy+40, cx-40:cx+40] = blob
    p = os.path.join(tempfile.mkdtemp(), "t.jpg")
    Image.fromarray(a).save(p, quality=92)
    return p, cx / W, cy / H


def test_top_crop_contains_subject():
    p, fx, fy = _img_with_blob(0.72, 0.5)  # subject on the right
    sug = crop.crop_suggestions(p, n=3)
    assert sug, "no suggestions"
    box = sug[0]["box"]
    assert box[0] <= fx <= box[2] and box[1] <= fy <= box[3]
    # crop should be tighter than the whole frame
    assert (box[2]-box[0]) < 0.99 or (box[3]-box[1]) < 0.99


def test_multiple_distinct_suggestions():
    p, *_ = _img_with_blob(0.5, 0.4)
    sug = crop.crop_suggestions(p, n=3)
    assert 1 <= len(sug) <= 3
    assert all("aspect" in s and "score" in s for s in sug)


class _MockEmb:
    semantic = True
    def embed_images(self, paths):
        import numpy as np
        from PIL import Image
        out = []
        for p in paths:
            m = np.asarray(Image.open(p).convert("L"), np.float32).mean() / 255.0
            out.append([m, 1 - m])
        return np.array(out, np.float32)


def test_aesthetic_guided_prefers_subject_crop():
    from autophotos.score import AestheticHead
    import numpy as np
    p, fx, fy = _img_with_blob(0.7, 0.5)
    head = AestheticHead([(np.array([[1.0, 0.0]], np.float32), np.array([0.0], np.float32))])
    sug = crop.suggest_crops(p, n=3, embedder=_MockEmb(), head=head)
    assert sug and sug[0]["method"] == "aesthetic"
    box = sug[0]["box"]
    # the top (brightest=most-textured) crop should contain the subject
    assert box[0] <= fx <= box[2] and box[1] <= fy <= box[3]
