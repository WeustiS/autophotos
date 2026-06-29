"""Personalized taste head: learns a direction from synthetic ratings."""
import json, os, sqlite3, tempfile
import numpy as np
from autophotos import config, db, taste


def _setup(n=40, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    d = tempfile.mkdtemp()
    os.environ["AUTOPHOTOS_CACHE_DIR"] = os.path.join(d, "cache")
    lib = config.Library(os.path.join(d, "lib")); lib.ensure_dirs()
    # a "taste direction": rating grows along axis 0
    V = rng.standard_normal((n, dim)).astype(np.float32)
    V[:, 0] = np.linspace(-2, 2, n)               # signal
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    ids = [f"h{i}" for i in range(n)]
    np.save(lib.emb_path, V); json.dump(ids, open(lib.ids_path, "w"))
    con = db.connect(lib.db_path)
    for i, h in enumerate(ids):
        con.execute("INSERT INTO photos (hash,path,filename,library,ext) VALUES (?,?,?,?,?)",
                    (h, f"/x/{h}", f"{h}.ARW", "/x", ".arw"))
    # rate the first 20 by their position along axis 0 -> ratings 1..5
    rank = np.argsort(V[:, 0])
    for r, i in enumerate(rank):
        if r % 2 == 0:  # rate half
            stars = 1 + min(4, int(5 * r / n))
            con.execute("INSERT INTO ratings (hash,rating) VALUES (?,?)", (ids[i], stars))
    con.commit()
    return lib, con, ids, V


def test_train_and_predict_recover_direction():
    lib, con, ids, V = _setup()
    rep = taste.train(lib, con, l2=0.5)
    assert rep["trained"] and rep["n_labels"] >= 8
    preds = taste.predict_all(lib)
    # predicted taste should correlate with the true signal (axis 0)
    p = np.array([preds[h] for h in ids])
    corr = np.corrcoef(p, V[:, 0])[0, 1]
    assert corr > 0.8, corr


def test_untrained_returns_empty():
    lib, con, ids, V = _setup()
    # no taste_head saved yet
    assert taste.predict_all(lib) == {}


def test_review_order_excludes_rated():
    lib, con, ids, V = _setup()
    taste.train(lib, con, l2=0.5)
    order = taste.review_order(lib, con)
    rated = {r["hash"] for r in con.execute("SELECT hash FROM ratings")}
    names_rated = {f"{h}.ARW" for h in rated}
    assert all(fn not in names_rated for fn, _ in order)
