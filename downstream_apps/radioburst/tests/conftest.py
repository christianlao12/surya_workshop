"""Test setup for the radio-burst app's own tests.

These tests live beside the app they cover rather than in the repo-level ``tests/``
folder, which holds the tests for the shared infrastructure. They still reuse the tiny
model fixtures from there — small enough to be fast, large enough to exercise both
backbone block types — so this makes ``tests/tiny_models.py`` importable, along with the
repo root that ``downstream_apps.*`` and ``workshop_infrastructure.*`` resolve against.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

for path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
