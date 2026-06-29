# Changelog

## 0.2.0 — full feature pass

Engine
- Personalized aesthetics (PIAA): `taste.py` ridge head learns your star ratings
  (embeddings -> rating); `scores.personal`; picks now rank by personal taste
  first, then generic aesthetic, then sharpness.
- Active-learning review order: unrated photos ranked by predicted taste.
- Crop suggestions: `crop.py` torch-free saliency + rule-of-thirds proposer,
  with a pluggable GAIC/ProCrop hook.
- Categories: `categories.py` zero-shot CLIP tagging from a vocabulary, plus
  unsupervised k-means cluster discovery and a VLM caption hook.
- Edit-queue + confirmed-stack workflow in `decisions.json`.
- Galaxy export gained named semantic axes (cold->warm, nature->urban, ...) when
  CLIP is present; PCA axes always available.

Interfaces
- CLI: `train-taste`, `review`, `crops`, `tag`, `cluster`, `queue {add|rm|list}`,
  `confirm-group` added.
- API: `/api/review`, `/api/crops/{hash}`, `/api/queue`, `/api/confirm-group`,
  `/api/train-taste`; `/api/photos` now reports `personal` + `queued`.

Correctness
- ClipEmbedder forces QuickGELU for OpenAI weights (correct embeddings + aesthetic).
- 21 unit tests (xmp non-destructiveness, grouping, picks, taste, crop,
  categories, decisions, semantic axes/layout, aesthetic-head folding, export).

## 0.1.0 — initial
- Hash-keyed scan/reconcile, EXIF, RAW embedded-preview thumbnails.
- CLIP/fallback embeddings, burst/stack grouping, technical + LAION aesthetic.
- darktable-safe XMP ratings, best-of-group picks.
- Semantic search/axes/PCA layout, FastAPI+browser viewer, static galaxy, Tauri shell.
