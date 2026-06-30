# autophotos — detailed walkthrough

This is the engineering guide: architecture, data model, every module, the models
used, how to develop/test, and known constraints. For a high-level intro see
[README.md](README.md); for the original design notes see [docs/](docs/).

## 1. What it is

A headless Python **engine** that turns a folder of RAW photos into a queryable,
scored, grouped library, plus three thin clients on top: a **browser culling
viewer** (FastAPI), an **interactive galaxy** (static HTML), and an optional
**Tauri** native window that just wraps the viewer. The filesystem is the source
of truth; everything derived is a rebuildable cache.

## 2. Architecture

```
            ┌──────────────────────── engine (Python) ───────────────────────┐
 RAW files ─┤ scan → exif → raw(thumbs) → embed → group → score → (taste)     │
   + XMP    │ semantic (search/axes/layout) · pick (best-of-group) · export    │
            └───────────────▲───────────────────────────────┬─────────────────┘
                            │ reads cache + sidecars         │ HTTP (localhost)
                  ┌─────────┴─────────┐            ┌─────────▼──────────┐
                  │  galaxy gallery   │            │  FastAPI + culler  │◄─ Tauri
                  │  (static HTML)    │            │  (web/culler.html) │   (optional)
                  └───────────────────┘            └────────────────────┘
```

The engine is the only thing that understands RAW/EXIF/ML. Clients are thin: the
viewer is HTML talking to the API; the galaxy is precomputed and self-contained;
Tauri spawns `uvicorn` as a sidecar and opens a window at `localhost:8731`.

## 3. Data model (filesystem = source of truth)

Per library `<root>/`:

```
<root>/
  IMG_0001.ARW                  source RAW (never modified)
  IMG_0001.ARW.xmp              XMP sidecar — ratings/picks/labels (source of truth)
  .autophotos/
    decisions.json              user choices not in XMP (edit queue, confirmed stacks)
    gallery.html                exported galaxy (rebuildable)
    cache/                      100% rebuildable — safe to delete
      index.sqlite              photos / ratings (mirror) / scores / groups
      embeddings.npy + ids.json CLIP vectors, row-aligned to ids
      model.json                which embed model produced the cache
      thumb/<hash>/{256,1024,preview}.jpg
      aesthetic_head.npz        LAION head (fetched)
      taste_head.npz            your personalized head (trained)
      categories.json / clusters.json / captions.json
```

**Identity = content hash** (`hashing.content_hash`: xxh64 of size + 1 MiB head +
1 MiB tail). Moving/renaming files or the whole folder is a no-op; `scan`
reconciles by hash and just updates paths. Three tiers: source RAW (yours), user
decisions (XMP + decisions.json — never auto-deleted), derived cache (delete &
rebuild anytime). `AUTOPHOTOS_CACHE_DIR` relocates the cache off a network/FUSE
mount that can't host SQLite.

### SQLite tables
`photos(hash PK, path, filename, captured_at, camera_model, lens, focal_len,
aperture, shutter, iso, ev_bias, focus_dist, drive_mode, orientation, width,
height, …)`; `ratings(hash, rating, pick, label)` mirrored from XMP;
`scores(hash, aesthetic, sharpness, exposure_ok, personal)`;
`groups(group_id, kind_hint, t_start, t_end, n, confidence)` + `group_members`.

## 4. Module map (`autophotos/`)

| module | role |
|---|---|
| `config.py` | `Library` paths, env-configurable model/cache, thresholds |
| `hashing.py` | content-hash identity |
| `db.py` | SQLite schema + upserts |
| `exif.py` | EXIF via exifread; optional exiftool enrich (focus dist/drive mode) |
| `raw.py` | embedded-preview extraction, thumbnail pyramid, on-demand RAW decode |
| `scan.py` | filesystem↔DB reconciliation by hash |
| `embed.py` | `ClipEmbedder` (open_clip, forces QuickGELU for OpenAI) + `FallbackEmbedder` |
| `group.py` | timestamp-run + EXIF-delta + similarity → burst/bracket/focus/pano |
| `score.py` | sharpness, exposure, LAION `AestheticHead` (numpy) |
| `taste.py` | personalized ridge head (PIAA) from your ratings; review order |
| `crop.py` | aesthetic-guided crop search (CLIP+head) + heuristic + GAIC hook |
| `categories.py` | zero-shot CLIP tags, k-means clusters, BLIP `caption_library` |
| `semantic.py` | text/image search, custom pole-pair axes, PCA 2D layout |
| `pick.py` | best-of-group representatives, ranked by taste→aesthetic→sharpness |
| `export.py` | builds the self-contained galaxy `gallery.html` |
| `pipeline.py` | orchestration: scan→thumbs→embed→group→score, xmp sync |
| `assets.py` | fetch + fold the LAION aesthetic predictor into `aesthetic_head.npz` |
| `report.py` | one-shot `python -m autophotos.report <lib>` → report.json |
| `api.py` | FastAPI app | `cli.py` | argparse CLI | `web/culler.html` | the viewer UI |

## 5. Pipeline

`scan` walks the root, hashes each RAW, reconciles new/moved/deleted, reads EXIF.
`thumbs` extracts the embedded JPEG (fast) into 256/1024/preview; falls back to a
half-size RAW decode only if there's no embedded preview. `embed` runs CLIP (or
fallback) over the 1024 thumbs → `embeddings.npy`. `group` clusters by capture
time then splits on scene change (embedding similarity) and classifies a
`kind_hint` from EXIF. `score` writes technical + aesthetic. `train-taste` fits a
ridge map embedding→your star rating and writes `personal` scores. XMP sync
mirrors sidecar ratings into the cache.

## 6. Models & scoring

- **Backbone:** open_clip. Use `ViT-L-14` + `openai`; `ClipEmbedder` forces
  QuickGELU for OpenAI weights (otherwise embeddings + the LAION head are subtly
  wrong). `FallbackEmbedder` (color+gradient, non-semantic) keeps the pipeline
  runnable without torch — grouping/dedup work, text features don't.
- **Aesthetic:** LAION "improved-aesthetic-predictor" linear-MLP. It has no
  activations, so `assets.fetch_aesthetic` folds the layer chain into one exact
  affine map stored as `aesthetic_head.npz`; `score.AestheticHead` runs it in pure
  numpy (no torch at score time). Expects ViT-L/14 (768-d).
- **Personal taste (PIAA):** `taste.py` ridge regression (closed form,
  bias-unregularized) on your rated photos → `personal` score. Picks prefer
  personal > aesthetic > sharpness, so culling reflects your eye as you rate.
- **Crops:** `crop.suggest_crops` generates rule-of-thirds/saliency candidates,
  then *scores each actual crop* with CLIP+aesthetic (optionally + taste vector)
  and keeps the best. Heuristic-only fallback; GAIC/ProCrop hook for a real model.
- **Captions:** `categories.BlipCaptioner` (transformers BLIP) → `captions.json`;
  graceful no-op without weights.
- **Semantic:** axes are text pole-pairs (`mean(pos)-mean(neg)` direction,
  project + rank-normalize); search is cosine over cached vectors; layout is PCA.

## 7. Interfaces

CLI: see [docs/FEATURES.md](docs/FEATURES.md). API: `/api/photos`, `/thumb`,
`/preview`, `/rate`, `/groups`, `/picks`, `/review`, `/train-taste`, `/crops`,
`/search`, `/queue`, `/confirm-group`, `/caption`. Viewer keys: `←/→` select,
`1–5` stars, `0` clear, `P` pick, `X` reject, `Q` queue, `C` crops, `Space` loupe;
tabs Cull / Stacks / Picks / Queue; buttons Train-taste and Sort-by-taste.

## 8. Develop & test

```bash
pip install -e ".[all]"
pytest -q                      # 24 tests (no torch needed; uses fallback + mocks)
```
Env: `AUTOPHOTOS_EMBED_MODEL`, `AUTOPHOTOS_EMBED_PRETRAINED`,
`AUTOPHOTOS_CACHE_DIR`, `AUTOPHOTOS_LIBRARY` (for the API), `AUTOPHOTOS_PYTHON`
(Tauri sidecar). Tests cover XMP non-destructiveness, grouping classification,
best-of-group, taste recovery, crop ranking, categories math, decisions
round-trip, semantic axes/layout, aesthetic-head folding, export.

## 9. Constraints & gotchas

- **Camera-agnostic:** model/dimensions come from EXIF; nothing hardcodes A7RV
  (test data was an A7 III).
- **exiftool optional:** pure-Python EXIF covers bracket/burst/pano; focus-stack
  detection is sharper with exiftool present (Sony MakerNotes).
- **Tauri builds on your machine** (Windows app can't be cross-compiled from
  Linux); crates.io + HuggingFace must be reachable for Rust + BLIP weights.
- The repo history was assembled in an isolated checkout because the dev sandbox's
  mounted filesystem couldn't host a live `.git` (lockfile ops blocked) — see
  [docs/GIT_HISTORY.md](docs/GIT_HISTORY.md).

## 10. Roadmap

Wire a real GAIC/ProCrop model into the crop hook; attach a stronger VLM
captioner; LoRA-finetune the taste head once enough ratings accrue; expose taste
re-training and caption browsing in the viewer; ship the galaxy as a hosted site.
