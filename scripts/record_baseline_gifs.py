"""Moved to ``scripts/benchmarks/record_baseline_gifs.py``.

A shim only, kept because a benchmark sweep started before the move
imports this module lazily, at each baseline's recording phase. Delete
it once no such run is in flight.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent
                       / "benchmarks"))

from record_baseline_gifs import (  # noqa: E402,F401
    DEFAULT_ENVS,
    MAX_FRAMES,
    record,
)
