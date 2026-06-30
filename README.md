# autophotos

A pipeline to ingest, cull, group, score, and (later) creatively display photos.
See `PLAN.md` for the high-level design and `ARCHITECTURE.md` for the code-level
plan.

**Status: v1 engine — working and validated on real libraries.**
Validated end-to-end on `ukgood` (36 ARW) and `bday-palmsprings` (598 ARW): scan
of 598 files in ~8.5 s, correct EXIF/scores, 70 candidate burst groups, and a
non-destructive XMP rating write that preserves darktable edit history
byte-for-byte.

## Quickstart (portable)

Clone and install on any machine (Windows/macOS/Linux, Python 3.10+):

```bash
git clone https://github.com/WeustiS/autophotos.git
cd autophotos
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[all]"                            # core + CLIP + API + captions
```

Point it at a folder of RAWs and go:

```bash
# CLIP backbone (recommended; matches the aesthetic head)
export AUTOPHOTOS_EMBED_MODEL=ViT-L-14 AUTOPHOTOS_EMBED_PRETRAINED=openai
autophotos index "/path/to/photos"          # scan + thumbs + embed + group + score
python -m autophotos.assets fetch-aesthetic "/path/to/photos"
autophotos score "/path/to/photos"

# cull in the browser
export AUTOPHOTOS_LIBRARY="/path/to/photos"
uvicorn autophotos.api:app --port 8731       # open http://localhost:8731
```

Nothing is hardcoded to a machine: libraries are passed by path, all state lives
under `<library>/.autophotos/` (rebuildable) and XMP sidecars (your ratings). Move
the repo or the photos freely. See FEATURES.md for every command, SETUP.md for
details, TAURI.md for the native window.

## What works now (the headless engine)

A Python package that treats the **filesystem as the source of truth** and builds
a fully rebuildable cache (`.autophotos/cache/`):

- `scan`   — walk a library, content-hash each RAW, reconcile new/moved/deleted, read EXIF
- `thumbs` — extract the embedded JPEG preview (~6 ms/file) into a 256/1024/preview pyramid
- `embed`  — image embeddings (CLIP when installed; a non-semantic fallback otherwise)
- `group`  — candidate stacks/bursts from timestamp runs + EXIF deltas + embedding similarity
- `score`  — technical scores (sharpness, exposure); aesthetic head when CLIP is active
- `rate`   — write star/reject/label to XMP sidecars, non-destructively (darktable/Lightroom-safe)

## Install

```bash
pip install -e .            # core engine
pip install -e .[clip]      # + torch + open_clip for real semantic embeddings
```

## Usage

```bash
autophotos index  /path/to/library      # scan + thumbs + embed + group + score + xmp sync
autophotos scan   /path/to/library
autophotos group  /path/to/library
autophotos stats  /path/to/library
autophotos rate   /path/to/photo.ARW --stars 4
autophotos rate   /path/to/photo.ARW --reject
```

By default the cache lives in `<library>/.autophotos/`. If the library is on a
network/FUSE mount that can't host a SQLite file, relocate the cache:

```bash
export AUTOPHOTOS_CACHE_DIR=/some/local/disk
```

## Data tiers (never confused)

| Tier | Location | Rebuildable? |
|---|---|---|
| Source RAW | library root | no (your originals) |
| User decisions | `*.xmp` sidecars + `.autophotos/decisions.json` | no (human input) |
| Derived cache | `.autophotos/cache/` (sqlite, embeddings.npy, thumbs) | yes — delete & rebuild |

Identity is a content hash, so moving/renaming files in Explorer is safe — a
rescan reconciles by hash, not path.

## Notes / known items

- Test data is a Sony A7 III (`ILCE-7M3`); the pipeline is camera-agnostic (reads
  model/dimensions from EXIF).
- Aesthetic scores are only populated with the CLIP backbone active. The fallback
  embedder is non-semantic (used to prove the pipeline runs without torch).
- `focus`-stack detection is reliable only with `exiftool` present (Sony
  MakerNotes: focus distance / drive mode); exposure-bracket, burst, and pano
  hints work from pure-Python EXIF.

## Implemented (v0.2)

Personalized taste (PIAA), crop suggestions, zero-shot tags + clusters, edit
queue + stack confirmation, named galaxy axes, and a full CLI/API. See
FEATURES.md and CHANGELOG.md.

## Roadmap (next)

- **v2** — FastAPI HTTP layer + Tauri viewer (Rust shell spawns the engine as a
  sidecar); keyboard-driven culling, loupe (on-demand RAW decode), stack-review queue.
- **v2.5** — GAIC crop proposals; personal-taste (PIAA/LoRA) head trained on
  accumulated ratings.
- **v3** — semantic axes + static WebGL website (semantic search, custom
  hot↔cold axes, galaxy map).
```
