#!/usr/bin/env python3
"""Unified local RECAP workflows; see docs/scripts.md for input/output contracts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recap_value.cli import main

if __name__ == '__main__':
    main()
