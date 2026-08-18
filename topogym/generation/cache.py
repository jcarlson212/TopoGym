"""Process-wide layout cache.

Generation is deterministic and certified, so a (configuration, seed)
key names exactly one layout — generating it twice is pure waste, and
under vector environments (N copies of the same id) the waste is Nx.
The cache stores the first build and hands every later request an
independent copy: the base map and certified metadata are immutable
and shared, while the mutable containers (cell types, doors, features,
free cells) are copied so no environment instance can leak state into
another.

Bounded LRU: procedural mode (no fixed seed) generates a fresh layout
per episode, which would otherwise grow the cache without limit.
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from collections.abc import Callable

logger = logging.getLogger("topogym")

_MAX_ENTRIES = 32
_CACHE: OrderedDict = OrderedDict()


def _handoff(layout):
    """An independent copy sharing only the immutable parts."""
    out = copy.copy(layout)  # shares base map + certified metadata
    out.cell_types = dict(layout.cell_types)
    out.doors = dict(layout.doors)
    out.features = list(layout.features)
    out.free_cells = list(layout.free_cells)
    # The archive's cached fingerprint is per *object*; a copy earns
    # its own rather than inheriting one it might outgrow.
    out.__dict__.pop("_fingerprint", None)
    return out


def cached_layout(key: tuple, builder: Callable):
    """The layout for ``key``, building (and certifying) at most once
    per process. Every caller gets an independent copy."""
    layout = _CACHE.get(key)
    if layout is None:
        layout = builder()
        _CACHE[key] = layout
        if len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
        logger.debug("layout-cache: built %r (%d cached)", key[:2],
                     len(_CACHE))
    else:
        _CACHE.move_to_end(key)
    return _handoff(layout)


def clear() -> None:
    """Drop every cached layout (tests; memory pressure)."""
    _CACHE.clear()
