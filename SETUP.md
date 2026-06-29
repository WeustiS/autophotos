# autophotos — local setup & test commands

The engine is validated end-to-end here **except CLIP**, because this sandbox is
PyPI-only (HuggingFace + the PyTorch CPU index are blocked, so model weights can't
be fetched). On your machine those work. Here's the full run.

Paths below use your real folders:
`C:\Users\willc\Pictures\ukgood` (36 RAW, quick) and
`C:\Users\willc\Pictures\bday-palmsprings` (598 RAW).

## 1. Create a venv and install

PowerShell, from the repo root (`C:\Users\willc\code\autophotos`):

```powershell
cd C:\Users\willc\code\autophotos
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip

# core engine + CLIP backbone (torch + open_clip)
pip install -e ".[clip]"
```

If you want a smaller CPU-only torch (no CUDA):

```powershell
pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install open_clip_torch
```

Optional, improves focus-stack detection (Sony MakerNotes): install exiftool and
put `exiftool.exe` on PATH. Not required.

## 2. Pick the embedding backbone

The LAION aesthetic head expects **CLIP ViT-L/14 (768-d, OpenAI weights)**. Set
this once per shell so embeddings and the aesthetic head match:

```powershell
$env:AUTOPHOTOS_EMBED_MODEL    = "ViT-L-14-quickgelu"   # quickgelu matches OpenAI CLIP
$env:AUTOPHOTOS_EMBED_PRETRAINED = "openai"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"              # silence a harmless Windows warning
```

Note: use the `-quickgelu` variant. Plain `ViT-L-14` + `openai` loads the right
weights but the wrong activation (open_clip warns about this), which degrades the
embeddings and the LAION aesthetic head.

(Skip this to use the faster default ViT-B-32/laion2b — semantic search/axes still
work, but the aesthetic head won't, since its dim won't match.)

The cache defaults to `<library>\.autophotos\`. That's fine on your local NTFS
disk. (Only set `$env:AUTOPHOTOS_CACHE_DIR` if a library lives on a network share
that can't host a SQLite file.)

## 3. Index a library

First run downloads the CLIP weights (~1 GB for ViT-L/14) once, then caches them.

```powershell
autophotos index "C:\Users\willc\Pictures\ukgood"
autophotos stats "C:\Users\willc\Pictures\ukgood"
```

`index` = scan + thumbnails + embed + group + score + XMP-sync. Re-running is
incremental (idempotent; only new/changed files do work).

## 4. Turn on the aesthetic head

```powershell
python -m autophotos.assets fetch-aesthetic "C:\Users\willc\Pictures\ukgood"
autophotos score "C:\Users\willc\Pictures\ukgood"   # re-score to fill aesthetic
```

## 5. Try the fun parts (semantic, Stage 3/4)

```powershell
# text search (needs CLIP)
autophotos search  "C:\Users\willc\Pictures\ukgood" "golden hour landscape" -k 10
autophotos search  "C:\Users\willc\Pictures\ukgood" "close-up of food"      -k 10

# a custom semantic axis, coldest -> hottest
autophotos axis    "C:\Users\willc\Pictures\ukgood" --pos "warm,cozy,sunset" --neg "cold,icy,blue" -k 8

# more like this photo (use a hash from `stats`/the ids.json; works even w/o CLIP)
autophotos similar "C:\Users\willc\Pictures\ukgood" <hash> -k 10

# 2D galaxy coords for the website -> writes .autophotos/cache/layout2d.json
autophotos layout  "C:\Users\willc\Pictures\ukgood"
```

## 6. Ratings round-trip (safe to try)

Writes a star rating to the XMP sidecar, non-destructively (darktable history is
preserved). Then re-sync into the cache:

```powershell
autophotos rate "C:\Users\willc\Pictures\ukgood\WE_08592.ARW" --stars 4
autophotos index "C:\Users\willc\Pictures\ukgood"
```

## What to report back

- Does `autophotos search` return sensible results for a few text queries?
- Do the `axis` poles order photos the way you'd expect?
- Aesthetic scores in `stats` (aesthetic scored: N) — do the top-ranked photos
  look like your better shots?

That feedback tells us whether ViT-L/14 + the LAION head is a good enough
zero-shot baseline before we invest in the personalized (PIAA/LoRA) head.

## v2 — the culling viewer (runs as a local web app now)

No Rust needed. The viewer is a FastAPI engine + browser UI.

```bash
pip install -e ".[api]"
export AUTOPHOTOS_LIBRARY="C:/Users/willc/Pictures/ukgood"
export AUTOPHOTOS_EMBED_MODEL=ViT-L-14 AUTOPHOTOS_EMBED_PRETRAINED=openai   # for search
uvicorn autophotos.api:app --port 8731
# open http://localhost:8731
```

Keyboard: ←/→ select · 1–5 stars · 0 clear · P pick · X reject · Space loupe.
Tabs: Cull / Stacks (the burst-review queue) / Picks (best-of-group ranking).
The search box does semantic text search. Ratings write straight to XMP sidecars.

Optional native wrapper (`src-tauri/`): with the Rust toolchain + Tauri CLI,
`cd src-tauri && cargo tauri dev` opens the same thing in a native window and
spawns the engine for you. The web app above is the faster path day-to-day.

## Best-of-group picks

```bash
autophotos picks "C:/Users/willc/Pictures/ukgood"   # collapses bursts to best frame
```

## The galaxy (interactive website)

```bash
python -m autophotos.export "C:/Users/willc/Pictures/bday-palmspringsandSC"
# writes .autophotos/gallery.html — open it in a browser
```

A self-contained page: 2D PCA map of your photos (similar shots cluster), borders
colored by aesthetic, hover to zoom, click any photo to highlight its nearest
neighbors, and X/Y axis dropdowns to re-arrange the galaxy.

## After the folder rename

You renamed the set to `bday-palmspringsandSC`. The cache moved with it (it lives
inside the folder) and identity is content-hash, so nothing re-embeds. Just run a
scan to refresh the stored paths:

```bash
autophotos scan "C:/Users/willc/Pictures/bday-palmspringsandSC"   # shows all as "moved", fixes paths
```

## Run the tests

```powershell
pip install pytest
pytest -q
```
(8 tests: XMP non-destructiveness, stack/burst classification, axis/layout math,
aesthetic-head folding.)
