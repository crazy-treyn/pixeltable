#!/usr/bin/env python3
"""
Run FTS verification: executes the same automated tests as the feature plan (sample data + assertions).

Usage from the repository root (uses your Pixeltable environment / DB):

  python tool/verify_fts_feature.py

This delegates to pytest for the full suite in tests/test_fts_index.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    test_path = repo_root / 'tests' / 'test_fts_index.py'
    return pytest.main([str(test_path), '-v', '--tb=short'])


if __name__ == '__main__':
    raise SystemExit(main())
