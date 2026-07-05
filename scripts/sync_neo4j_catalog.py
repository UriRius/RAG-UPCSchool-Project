#!/usr/bin/env python3
"""Wrapper: usa src/sync_graph.py (sustituye notebooks para sync Neo4j)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--from-drive", "--sync-neo4j"])
    from sync_graph import main

    raise SystemExit(main())
