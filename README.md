# autophotos

A local, filesystem-first pipeline to **cull, group, score, and explore** a large
photo library from a mirrorless workflow (built/tested on Sony `.ARW`). It ingests
RAWs, scores them for technical and aesthetic quality, learns *your* taste from
your star ratings, groups bursts/stacks, suggests crops, and gives you a fast
browser culling app plus an interactive "galaxy" gallery — all offline, with your
ratings written to standard XMP sidecars.

> Detailed walkthrough: **[CLAUDE.md](CLAUDE.md)**. Deep dives in **[docs/](docs/)**.

## Features

- **Fast ingest** — content-hash identity, embedded-JPEG previews (~6 ms/file), no re-decode.
- **Cull** — browser viewer: keyboard rating, loupe, reject/pick, edit queue.
- **Scoring** — sharpness + exposure (always); CLIP + LAION aesthetic (with CLIP);
  **personal taste** learned from your ratings (PIAA).
- **Grouping** — burst / bracket / focus-stack / pano candidates; best-of-group picks.
- **Crops** — aesthetic-guided crop suggestions (CLIP-scored), heuristic fallback.
- **Semantic** — text search, "more like this", custom axes, zero-shot tags, clusters, BLIP captions.
- **Galaxy** — a self-contained interactive `gallery.html` (2D map, aesthetic color, custom axes).
- **Safe** — filesystem is the source of truth; everything in `.autophotos/` is a rebuildable cache; ratings live in XMP (darktable/Lightroom-compatible).

## Install

```bash
git clone https://github.com/WeustiS/autophotos.git && cd autophotos
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[all]"                           # core + CLIP + API + captions
```
Python 3.10+. CLIP/captions pull in torch (a few hundred MB on first install).

## Quickstart

```bash
# recommended CLIP backbone (matches the aesthetic head)
export AUTOPHOTOS_EMBED_MODEL=ViT-L-14 AUTOPHOTOS_EMBED_PRETRAINED=openai

autophotos index "/path/to/photos"                       # scan+thumbs+embed+group+score
python -m autophotos.assets fetch-aesthetic "/path/to/photos" && autophotos score "/path/to/photos"

export AUTOPHOTOS_LIBRARY="/path/to/photos"
uvicorn autophotos.api:app --port 8731                   # cull at http://localhost:8731

python -m autophotos.export "/path/to/photos"            # build the galaxy gallery.html
```

After you've rated a couple dozen shots: `autophotos train-taste "/path/to/photos"`
to personalize the ranking.

## Commands (cheat-sheet)

| | |
|---|---|
| `index / scan / thumbs / embed / group / score` | pipeline stages |
| `stats / picks` | summary; best-of-group ranking |
| `rate <raw> --stars N [--reject]` | write XMP rating |
| `train-taste / review` | learn taste; sort unrated by predicted taste |
| `crops <hash>` | crop suggestions |
| `search "text" / similar <hash> / axis --pos a,b --neg c,d` | semantic |
| `tag / cluster / caption` | categories + captions |
| `queue {add\|rm\|list} / confirm-group` | edit queue, stack decisions |

Full reference: [docs/FEATURES.md](docs/FEATURES.md). Native window: [docs/TAURI.md](docs/TAURI.md).

## Project layout

```
autophotos/        engine package (scan, embed, group, score, taste, crop,
                   categories, semantic, pick, export, api, cli, web/culler.html)
tests/             unit tests (pytest)
src-tauri/         optional native window wrapper (Tauri v2)
docs/              PLAN, ARCHITECTURE, SETUP, FEATURES, TAURI, CHANGELOG
CLAUDE.md          detailed walkthrough
```

## License

MIT — see [LICENSE](LICENSE).
