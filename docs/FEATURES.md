# autophotos — command reference

## Pipeline
| command | what |
|---|---|
| `autophotos index <lib>` | scan + thumbs + embed + group + score + xmp sync |
| `autophotos scan/thumbs/embed/group/score <lib>` | individual stages |
| `autophotos stats <lib>` | counts (photos, groups, rated, aesthetic, personal) |

## Cull & taste
| command | what |
|---|---|
| `autophotos rate <raw> --stars N [--reject] [--label L]` | write XMP rating |
| `autophotos picks <lib> [-k]` | best-of-group ranking (personal > aesthetic) |
| `autophotos train-taste <lib> [--l2]` | fit personal taste head from your ratings |
| `autophotos review <lib> [-k]` | unrated photos ranked by predicted taste |

## Compose
| command | what |
|---|---|
| `autophotos crops <lib> <hash> [-k]` | rule-of-thirds crop suggestions |

## Semantic & website
| command | what |
|---|---|
| `autophotos search <lib> "text" [-k]` | semantic text search (CLIP) |
| `autophotos similar <lib> <hash> [-k]` | nearest neighbors |
| `autophotos axis <lib> --pos a,b --neg c,d` | rank along a custom axis |
| `autophotos tag <lib>` | zero-shot category tags |
| `autophotos cluster <lib> [-k]` | unsupervised clusters |
| `autophotos layout <lib>` | PCA 2D coords |
| `python -m autophotos.export <lib>` | build the interactive galaxy.html |

## Workflow
| command | what |
|---|---|
| `autophotos queue <lib> {add|rm|list} [hash...]` | edit queue |
| `autophotos confirm-group <lib> --members h1,h2 --kind K --action A` | confirm a stack |

## Viewer / report
- `uvicorn autophotos.api:app --port 8731` — browser culling viewer
- `python -m autophotos.report <lib>` — one-shot pipeline + report.json
