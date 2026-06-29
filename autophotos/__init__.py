"""autophotos engine: headless photo ingest / cull / group / score pipeline.

Filesystem is the source of truth; everything under <library>/.autophotos/cache/
is a rebuildable derived cache. See ARCHITECTURE.md.
"""
__version__ = "0.1.0"
