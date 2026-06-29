"""Best-of-group collapsing."""
import sqlite3

from autophotos import db, pick


def _con():
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(db.SCHEMA)
    return con


def _photo(con, h):
    con.execute("INSERT INTO photos (hash,path,filename,library,ext) VALUES (?,?,?,?,?)",
                (h, f"/x/{h}.ARW", f"{h}.ARW", "/x", ".arw"))


def test_best_of_group_collapses_burst():
    con = _con()
    # a 5-frame burst g1 with varying aesthetic; b3 is best
    for i, aes in enumerate([4.0, 4.5, 4.2, 6.1, 5.0]):
        h = f"b{i}"
        _photo(con, h)
        con.execute("INSERT INTO scores (hash,aesthetic,sharpness) VALUES (?,?,?)", (h, aes, 100 + i))
        con.execute("INSERT INTO group_members (group_id,hash,seq) VALUES (?,?,?)", ("g1", h, i))
    con.execute("INSERT INTO groups (group_id,kind_hint,n) VALUES ('g1','burst',5)")
    # two ungrouped singletons
    for h, aes in [("s0", 5.5), ("s1", 3.0)]:
        _photo(con, h)
        con.execute("INSERT INTO scores (hash,aesthetic,sharpness) VALUES (?,?,?)", (h, aes, 50))
    con.commit()

    reps = pick.top_picks(con, k=10)
    # 7 photos -> 3 representatives (1 per group + 2 singletons)
    assert len(reps) == 3
    summ = pick.summary(con)
    assert summ["collapsed_away"] == 4  # 5-frame burst -> 1
    # ranking: b3 (6.1) > s0 (5.5) > s1 (3.0); only the best burst frame appears
    assert [r["filename"] for r in reps] == ["b3.ARW", "s0.ARW", "s1.ARW"]
    assert reps[0]["from_group"] and reps[0]["group_size"] == 5
