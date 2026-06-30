# Commit history

This sandbox couldn't host a live `.git` on the mounted folder (it blocks the
lock/rename/unlink operations git needs — same limitation that stops SQLite).
So the full history was built in an isolated checkout and delivered as a **git
bundle**: `autophotos.bundle` (9 commits).

A stray, broken `.git/` may exist from a failed `git init` attempt — delete it
first (you can, on your own machine): 

```bash
cd C:/Users/willc/code/autophotos
rm -rf .git                       # remove the broken stub (PowerShell: Remove-Item -Recurse -Force .git)
git clone autophotos.bundle .tmp_hist && mv .tmp_hist/.git . && rm -rf .tmp_hist
git checkout .                    # restore tracked files from history (optional)
git log --oneline                 # review
```

Or just inspect the history without touching your working tree:

```bash
git clone autophotos.bundle autophotos-history
cd autophotos-history && git log --oneline --stat
```

## Commits
- Initial import: engine, viewer, galaxy, docs
- feat(taste): personalized aesthetics (PIAA) ridge head
- feat(crop): torch-free saliency + rule-of-thirds crop suggestions
- feat(categories): zero-shot CLIP tags + k-means cluster discovery
- feat(decisions): edit queue + confirmed stack groups workflow
- feat(cli): wire taste, crops, tag, cluster, review, queue, confirm-group
- feat(api): endpoints for review, crops, queue, confirm-group, train-taste
- feat(export): named semantic axes in the galaxy + export test
- docs: CHANGELOG, FEATURES command reference, README status
