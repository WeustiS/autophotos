"""Zero-shot tagging + k-means cluster discovery (model-free math)."""
import numpy as np
from autophotos import categories


def _unit(v): return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_assign_tags_picks_nearest():
    # 3 tag vectors along 3 axes; images aligned to each
    tags = _unit(np.eye(3, dtype=np.float32))
    names = ["a", "b", "c"]
    img = _unit(np.array([[1, .1, 0], [0, 1, .1], [.1, 0, 1]], np.float32))
    res = categories.assign_tags(img, tags, names, topk=1, thresh=0.0)
    assert [r[0][0] for r in res] == ["a", "b", "c"]


def test_threshold_filters_weak():
    tags = _unit(np.eye(2, dtype=np.float32)); names = ["a", "b"]
    img = _unit(np.array([[1, 1]], np.float32))  # 0.707 to each
    assert categories.assign_tags(img, tags, names, topk=2, thresh=0.9) == [[]]


def test_kmeans_separates_two_blobs():
    rng = np.random.default_rng(0)
    A = rng.normal(0, .05, (30, 5)) + np.array([2, 0, 0, 0, 0])
    B = rng.normal(0, .05, (30, 5)) + np.array([-2, 0, 0, 0, 0])
    X = np.vstack([A, B]).astype(np.float32)
    labels, _ = categories.kmeans(X, 2)
    # each true blob should be internally consistent
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[30]
