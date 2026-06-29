# autophotos — Architecture & Code Plan (v1)

Detailed, code-level companion to `PLAN.md`. This is the document to review
before and after the first implementation pass. It records concrete decisions,
on-disk formats, schemas, and module/function layout, grounded in findings from
the real test libraries.

---

## Findings from the real data (these shaped the design)

Probed `bday-palmsprings` (598 ARW + 2 MP4, ~31 GB) and `ukgood` (36 ARW + 1 XMP,
~1.2 GB):

- **Camera is a Sony A7 III (`ILCE-7M3`, 24 MP), not the A7R V.** → The pipeline
  must be **camera-agnostic**: read model/dimensions from EXIF, never hardcode.
- **Embedded JPEG preview extraction is ~6 ms/file** at **1616×1080**. → Use the
  embedded preview for grid + fit-to-screen instantly; it is *not* full-res, so
  **100% loupe requires an on-demand RAW decode** (rawpy, optionally half-size).
- **Standard EXIF is readable in pure Python** (exifread): DateTimeOriginal,
  ExposureTime, FNumber, ISO, FocalLength, ExposureBiasValue, Orientation, Model.
  → No exiftool dependency required for v1.
- **No sub-second timestamps** in these files (`SubSecTimeOriginal` empty). →
  Burst grouping uses 1 s windows + embedding similarity, not sub-second order.
- **70 timestamp-runs** (≥3 frames, ≤3 s gaps) found in palmsprings, up to 20
  frames in 4 s, all EV=0 → correctly continuous-shooting bursts, no brackets. The
  cheap timestamp+EV grouping is validated on real data.
- **The existing sidecar is a darktable XMP** (`WE_09609.ARW.xmp`) holding
  `xmp:Rating` plus a full edit history. → You use **darktable**. XMP writes must
  be **non-destructive** (touch only rating/pick/label; preserve all darktable
  nodes) and must handle the `<name>.ARW.xmp` naming convention.

---

## Top-level architecture: headless engine + thin UI

The ML/RAW ecosystem is Python; the perf-sensitive UI is Tauri (your call). The
clean seam is:

```
┌─────────────────────────────────────────────────────────────┐
│  autophotos ENGINE  (Python, headless, testable in isolation)│
│  scan · exif · raw/thumbs · embed · group · score · xmp · db │
│  exposes:  CLI  +  local HTTP API (FastAPI, added in v2)      │
└───────────────▲──────────────────────────┬──────────────────┘
                │ reads/writes              │ HTTP (localhost)
   ┌────────────┴───────────┐   ┌───────────▼───────────────┐
   │  Rebuildable cache      │   │  Tauri viewer (v2)         │
   │  .autophotos/cache/     │   │  Rust shell + web frontend │
   │  (sqlite, npy, thumbs)  │   │  spawns engine as sidecar  │
   └─────────────────────────┘   └────────────────────────────┘
                │
   ┌────────────▼───────────┐   ┌────────────────────────────┐
   │  Source of truth        │   │  Static website (v3)        │
   │  RAWs + XMP + decisions  │   │  exports JSON + thumbs      │
   └─────────────────────────┘   └────────────────────────────┘
```

Rationale:
- The engine produces the rebuildable cache and is the *only* component that
  understands RAW/EXIF/ML. It is fully testable headless (and is what we build and
  validate first, against your real photos).
- The Tauri viewer is a **thin client**: at runtime Tauri spawns the Python engine
  as a sidecar process and talks to it over localhost HTTP. Rust handles window/
  filesystem-watch/perf; Python handles decode + ML. (Tauri's sidecar mechanism is
  built for exactly this.)
- The website is a third client that calls the engine's `export` once and then
  serves static files. No model at runtime.

Why not pure-Rust core: the embedding models, rawpy, GAIC, and the aesthetic heads
are all Python; reimplementing in Rust is throwaway work. Why not pure-web/FastAPI
instead of Tauri: you chose Tauri for loupe latency, and the sidecar pattern keeps
that while reusing the Python engine.

**Build order maps to versions:** v1 = engine + CLI (this pass), v2 = HTTP API +
Tauri viewer, v3 = website export + frontend.

---

## On-disk layout (filesystem = source of truth)

Per library root (e.g. `…/Pictures/ukgood/`):

```
ukgood/
  WE_08592.ARW                 # source RAW (never modified or moved by us)
  WE_08592.ARW.xmp             # XMP sidecar — SOURCE OF TRUTH for rating/pick/label
  …
  .autophotos/                 # all autophotos state for this library
    decisions.json             # SOURCE OF TRUTH for user choices not in XMP
    cache/                     # 100% REBUILDABLE — safe to delete anytime
      index.sqlite             # metadata + EXIF + scores + candidate groups
      embeddings.npy           # float32 [N, D], row order == ids.json
      ids.json                 # ["<hash>", …] parallel to embeddings rows
      thumb/<hash>/256.jpg     # thumbnail pyramid
      thumb/<hash>/1024.jpg
      thumb/<hash>/preview.jpg # the embedded ~1616px JPEG, verbatim
      captions/<hash>.txt      # VLM caption (v2+)
      model.json               # which embed/caption models produced the cache
```

**Three data tiers, strictly separated:**

| Tier | Lives in | Rebuildable? | Examples |
|---|---|---|---|
| Source RAW | library root | no (it's your originals) | `*.ARW`, `*.MP4` |
| User decisions | XMP sidecars + `decisions.json` | no (human input) | rating, pick/reject, color label, confirmed stack groups + actions, edit-queue membership |
| Derived cache | `.autophotos/cache/` | **yes** — `rm -rf` and rebuild | sqlite, embeddings, thumbs, captions, *candidate* groups, scores |

`decisions.json` holds only things not expressible in XMP — confirmed stack/pano
groups and their chosen action, plus any queue state. Ratings/picks/labels go to
XMP for darktable/Lightroom interop. Both are plain text/JSON: greppable,
diffable, mergeable.

### Identity = content hash, not path

Every cache/decision record is keyed by a **content hash** so manual moves/renames
are no-ops and true new files are detected.

- Hash = `xxh64(size_bytes || first 1 MiB || last 1 MiB)` rendered hex. Full-file
  hashing 30 GB is too slow; size+head+tail is collision-safe for distinct photos
  and stable across moves. (xxhash via `pip install xxhash`; fall back to blake2b.)
- `scan` reconciles disk vs. `index.sqlite`:
  - hash present on disk, absent in DB → **new** (index it)
  - hash in DB, path changed on disk → **moved** (update path only)
  - hash in DB, absent on disk → **deleted** (mark/remove cache row)
  - Idempotent; run on launch and on filesystem-watch events.

---

## SQLite schema (`index.sqlite`, the rebuildable cache)

```sql
CREATE TABLE photos (
  hash         TEXT PRIMARY KEY,    -- content hash (identity)
  path         TEXT NOT NULL,       -- current absolute path
  filename     TEXT NOT NULL,
  library      TEXT NOT NULL,       -- library root path
  ext          TEXT NOT NULL,
  size         INTEGER,
  file_mtime   REAL,
  captured_at  TEXT,                -- ISO8601 from EXIF DateTimeOriginal
  sub_sec      INTEGER,             -- nullable
  camera_model TEXT,
  lens         TEXT,
  focal_len    REAL,
  aperture     REAL,                -- f-number
  shutter      REAL,                -- seconds
  iso          INTEGER,
  ev_bias      REAL,
  focus_dist   REAL,                -- nullable (needs exiftool/makernotes)
  drive_mode   TEXT,                -- nullable (needs exiftool/makernotes)
  orientation  INTEGER,
  width        INTEGER,
  height       INTEGER,
  indexed_at   TEXT
);

-- ratings mirrored FROM xmp on scan (xmp is source of truth)
CREATE TABLE ratings (
  hash       TEXT PRIMARY KEY REFERENCES photos(hash),
  rating     INTEGER,              -- 0..5 stars
  pick       INTEGER,              -- -1 reject / 0 none / 1 pick
  label      TEXT,                 -- color label
  xmp_path   TEXT,
  updated_at TEXT
);

CREATE TABLE scores (
  hash        TEXT PRIMARY KEY REFERENCES photos(hash),
  aesthetic   REAL,                -- general aesthetic (0..10-ish)
  sharpness   REAL,                -- laplacian variance, normalized
  exposure_ok REAL,                -- 0..1 (clipping penalty)
  personal    REAL,                -- PIAA head (null until trained)
  scored_at   TEXT
);

-- CANDIDATE groups (auto-detected, rebuildable). Confirmed groups -> decisions.json
CREATE TABLE groups (
  group_id   TEXT PRIMARY KEY,
  kind_hint  TEXT,                 -- burst|bracket|focus|pano|unknown
  t_start    TEXT,
  t_end      TEXT,
  n          INTEGER,
  confidence REAL
);
CREATE TABLE group_members (
  group_id TEXT REFERENCES groups(group_id),
  hash     TEXT REFERENCES photos(hash),
  seq      INTEGER,
  PRIMARY KEY (group_id, hash)
);

CREATE INDEX idx_photos_captured ON photos(captured_at);
CREATE INDEX idx_photos_library  ON photos(library);
```

Embeddings are *not* in SQLite — they live in `embeddings.npy` + `ids.json` for
fast vectorized cosine. `model.json` records the embed model id + dim so a model
change triggers a deliberate re-embed.

---

## Module layout (Python package `autophotos/`)

```
autophotos/
  __init__.py
  config.py        # paths, model ids, thresholds; cache/decisions locations
  hashing.py       # content_hash(path) -> str
  db.py            # connect(), schema, upserts, queries
  exif.py          # read_exif(path) -> dict   (exifread; optional exiftool enrich)
  raw.py           # extract_preview(), make_thumbs(), decode_full() (rawpy)
  scan.py          # scan(library) -> reconcile disk<->db, returns changes
  embed.py         # Embedder protocol; ClipEmbedder; FallbackEmbedder
  group.py         # detect_groups() timestamp-run + EXIF delta + similarity
  score.py         # aesthetic(), sharpness(), exposure_ok(); piaa hook
  xmp.py           # read_rating(), write_rating()  (non-destructive, darktable-safe)
  decisions.py     # load/save decisions.json (confirmed groups, queue)
  pipeline.py      # orchestration: scan -> index -> embed -> group -> score
  cli.py           # argparse entrypoints
  api.py           # FastAPI app (v2)
tests/
  test_hashing.py test_group.py test_xmp.py test_scan.py
pyproject.toml
```

### Key function contracts

```python
# hashing.py
def content_hash(path: str, head=1<<20, tail=1<<20) -> str: ...

# exif.py  — pure-python; exiftool used only if shutil.which('exiftool')
def read_exif(path: str) -> dict:   # normalized keys matching photos columns
    # {captured_at, sub_sec, camera_model, lens, focal_len, aperture,
    #  shutter, iso, ev_bias, focus_dist, drive_mode, orientation, width, height}

# raw.py
def extract_preview(path) -> bytes            # embedded JPEG, verbatim (~6ms)
def make_thumbs(path, out_dir) -> dict         # {256:..,1024:..,preview:..}
def decode_full(path, half=False) -> np.ndarray# on-demand, for loupe/export

# embed.py
class Embedder(Protocol):
    id: str; dim: int
    def embed_images(self, paths: list[str]) -> np.ndarray  # [n, dim] L2-normalized
    def embed_texts(self, texts: list[str]) -> np.ndarray
# ClipEmbedder: open_clip ViT-B-32 (CPU ok). FallbackEmbedder: tiny perceptual
# descriptor (downscaled-Lab histogram + gradients) so the pipeline runs even
# without torch; clearly flagged as non-semantic.

# group.py
def detect_groups(photos: list[Row], emb: dict[str,np.ndarray],
                  max_gap_s=3, min_n=3, sim_split=0.85) -> list[Group]:
    # 1. sort by captured_at; cut into runs where gap>max_gap_s
    # 2. within a run, split where adjacent cosine sim < sim_split (scene change)
    # 3. classify kind_hint:
    #    - ev_bias varies over >=3 distinct steps & framing stable -> 'bracket'
    #    - focus_dist monotonic & exposure stable (needs exiftool) -> 'focus'
    #    - framing drifts directionally, exposure stable          -> 'pano'
    #    - else                                                    -> 'burst'
    # liberal: prefer over-grouping; user adjudicates.

# xmp.py  — non-destructive
def read_rating(raw_path) -> dict      # finds <raw>.xmp or <raw>.ARW.xmp
def write_rating(raw_path, rating=None, pick=None, label=None):
    # if sidecar exists: parse XML, set only xmp:Rating / xmp:Label / pick attr,
    #   preserve every other node (darktable history etc.), write back.
    # if absent: create minimal sidecar using <raw>.ARW.xmp (darktable convention).
```

### Stack/burst detection detail

Timestamp-run cut (validated) → similarity split → EXIF-based `kind_hint`. Output
is *candidate* groups in the DB. The viewer shows each with its hint + triggering
EXIF; your confirmation writes to `decisions.json`. v1 implements
burst/bracket/pano from pure-python EXIF + embeddings; `focus` becomes reliable
once exiftool (Sony MakerNotes: focus distance, drive mode, sequence #) is present,
which is the one place exiftool materially helps and is therefore an optional
enrichment, not a hard dep.

---

## Aesthetic scoring (v1) and the path to personalization

- v1 technical: `sharpness` = variance of Laplacian on the 1024 thumb (normalized
  per-camera); `exposure_ok` = 1 − clipped-pixel fraction from the preview
  histogram. Cheap, deterministic, runs first for the cull.
- v1 aesthetic: LAION aesthetic linear head on CLIP embeddings when `ClipEmbedder`
  is active (small known weights). With `FallbackEmbedder`, aesthetic is skipped
  (flagged), technical scores still populate.
- Deferred (2c): personal `personal` column filled by a PIAA/LoRA head trained on
  accumulated `ratings`. Ratings are collected from day one, so turning this on
  later needs no new labeling.

---

## Testing & verification strategy

- Unit: `content_hash` stability across a simulated move; `group.detect_groups` on
  synthetic timestamp/EV sequences; `xmp.write_rating` round-trip that asserts
  darktable history bytes are preserved.
- Integration: run the full pipeline on `ukgood` (36 files, fast) every change;
  assert manifest counts, thumbnail existence, embeddings shape, ≥1 detected
  group, and an XMP rating round-trip on a copy.
- Scale check: `scan` + `make_thumbs` timing on `bday-palmsprings` (598 files) to
  confirm the ~6 ms preview path holds at trip scale.
- High-stakes correctness (XMP non-destructiveness) gets an explicit byte-diff
  test because clobbering darktable edits would be data loss.

---

## Open decisions (low-stakes; sensible defaults chosen, easy to revisit)

- Cache location: per-library `.autophotos/` (chosen) vs. central app-data dir.
  Per-library keeps it portable and deletable; revisit if you want one global DB.
- Embed model: open_clip ViT-B-32 default for speed; bump to SigLIP/ViT-L later
  (triggers a re-embed, which is fine at thousands).
- New-sidecar naming: `<name>.ARW.xmp` (darktable) chosen since you use darktable.
```
