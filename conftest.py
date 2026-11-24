"""Ensure tests import modules from this repository root.

Burn_restore/ contains an __init__.py, so pytest's default rootdir insertion puts
the *parent* directory first on sys.path.  If a stale copy of api/ or inference/
exists there, it would shadow the code in this repo.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)
for _name in [m for m in sys.modules if m == "api" or m.startswith(("api.", "inference."))]:
    del sys.modules[_name]
