"""Tests for semantic axes / layout / aesthetic head (synthetic geometry)."""
import json
import os
import tempfile

import numpy as np

from autophotos import config, semantic
from autophotos.score import AestheticHead


def _fake_library(vecs):
    d = tempfile.mkdtemp()
    os.environ["AUTOPHOTOS_CACHE_DIR"] = os.path.join(d, "cache")
    lib = config.Library(os.path.join(d, "lib"))
    lib.ensure_dirs()
    ids = [f"h{i}" for i in range(len(vecs))]
    np.save(lib.emb_path, vecs.astype(np.float32))
    json.dump(ids, open(lib.ids_path, "w"))
    json.dump({"id": "synthetic", "dim": vecs.shape[1], "semantic": True},
              open(lib.model_path, "w"))
    return lib, ids


def test_axis_projection_orders_by_direction():
    # 5 points spread along the x-axis; projecting onto +x must order them by x.
    vecs = np.zeros((5, 3), np.float32)
    vecs[:, 0] = [-2, -1, 0, 1, 2]
    vecs[:, 1] = 0.1  # tiny noise on another dim
    lib, ids = _fake_library(vecs)
    axis = np.array([1.0, 0.0, 0.0], np.float32)
    proj = semantic.project_axis(lib, axis, normalize="rank")
    ordered = [h for h, _ in sorted(proj.items(), key=lambda kv: kv[1])]
    assert ordered == ids  # already in increasing-x order


def test_exemplar_axis_sign():
    vecs = np.eye(4, dtype=np.float32)  # 4 orthonormal points
    lib, ids = _fake_library(vecs)
    # axis from point0 (pos) vs point1 (neg): point0 should score highest
    axis = semantic.axis_from_exemplars(lib, [ids[0]], [ids[1]])
    proj = semantic.project_axis(lib, axis, normalize="z")
    assert proj[ids[0]] == max(proj.values())
    assert proj[ids[1]] == min(proj.values())


def test_layout_2d_in_unit_square():
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((20, 8)).astype(np.float32)
    lib, _ = _fake_library(vecs)
    xy = semantic.layout_2d(lib)
    arr = np.array(list(xy.values()))
    assert arr.shape == (20, 2)
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_aesthetic_head_folds_linear_chain():
    # A no-activation 2-layer chain must equal one folded affine map.
    rng = np.random.default_rng(1)
    W1, b1 = rng.standard_normal((4, 3)), rng.standard_normal(4)
    W2, b2 = rng.standard_normal((1, 4)), rng.standard_normal(1)
    x = rng.standard_normal(3).astype(np.float32)
    chain = W2 @ (W1 @ x + b1) + b2
    folded = AestheticHead([(W2 @ W1, W2 @ b1 + b2)])
    assert np.isclose(folded(x), chain[0], atol=1e-4)
    assert folded.dim == 3
