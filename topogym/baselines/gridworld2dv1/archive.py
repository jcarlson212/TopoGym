"""Archives of visited cells, and the Ray actor that shares them.

An archive belongs to a *world*, not to a process and not to a run.
Two workers exploring the same layout should contribute to one archive;
two workers on different layouts must never share one, however similar
the worlds look. Identity therefore comes from the layout's contents --
:func:`layout_fingerprint` -- rather than from an object reference,
which does not survive a process boundary, or from a registry id, which
several instances share.

Three pieces, deliberately separable:

- :class:`LayoutArchive` -- one world's archive. Plain Python, no Ray,
  so the algorithm can be tested without a cluster.
- :class:`ArchiveService` -- holds one archive per fingerprint and
  answers update-and-select. Plain Python too; :func:`remote_archive`
  turns it (or a subclass) into a Ray actor.
- :func:`layout_fingerprint` -- the key.

Phase 2 of Go-Explore needs an archive that also stores the trajectory
to each cell. That is a subclass of :class:`LayoutArchive` and a
subclass of :class:`ArchiveService` naming it -- not a fork of either,
which is why ``archive_class`` is a class attribute.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger("topogym")

#: The counted attributes of Go-Explore's A.5 cell score.
ATTRIBUTES = ("chosen", "seen", "chosen_since_new")

DEFAULTS = {
    "eps1": 0.001,
    "eps2": 0.00001,
    "w_a": 1.0,
    "p_a": 0.5,
    "w_n": 1.0,
}


def layout_fingerprint(layout) -> str:
    """A stable identity for a world.

    Covers what makes two worlds behave differently: which cells are
    what, where the doors are, where the agent starts and the goal
    sits, and the dynamics the extras carry (a seasonal schedule, a
    wormhole map). Two instances of the same configuration and seed
    fingerprint alike in any process; a different seed does not.

    Computed once per layout object and cached on it. Nothing that
    defines a world mutates in place -- the one in-place write,
    ``extras["optimal_actions"]``, is excluded below as derived -- and
    an archive method asks on every step, where serializing a whole
    world each time cost forty times the stepping it was keyed to.
    """
    cached = getattr(layout, "_fingerprint", None)
    if cached is not None:
        return cached
    parts = [
        repr(sorted(layout.cell_types.items(), key=repr)),
        repr(sorted(layout.doors, key=repr)),
        repr(layout.start),
        repr(layout.goal),
        repr(layout.base.layout_size()),
    ]
    extras = layout.extras or {}
    for key in sorted(extras):
        if key in ("optimal_actions",):  # derived, not defining
            continue
        parts.append(f"{key}={_stable(extras[key])}")
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
    layout._fingerprint = digest[:32]
    return layout._fingerprint


def _stable(value) -> str:
    """A repr that does not depend on set or dict ordering."""
    if isinstance(value, dict):
        return repr(sorted(
            ((k, _stable(v)) for k, v in value.items()), key=repr))
    if isinstance(value, (set, frozenset)):
        return repr(sorted(value, key=repr))
    if isinstance(value, (list, tuple)):
        return repr([_stable(v) for v in value])
    return repr(value)


class LayoutArchive:
    """One world's archive: cells, their counts, and selection.

    Scoring follows Go-Explore's Appendix A.5 (see
    :mod:`~topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1`).
    Adjacency is stored rather than a neighbour *function*, because the
    archive has to cross a process boundary and a bound method of a
    base map does not.
    """

    def __init__(self, params: dict | None = None, seed: int = 0,
                 adjacency=None, neighbors=None):
        self.params = {**DEFAULTS, **(params or {})}
        self.rng = np.random.default_rng(seed)
        #: A mapping when the archive must cross a process boundary, a
        #: callable when it stays in one -- an in-process caller
        #: already holds the base map's neighbour function and need not
        #: materialise adjacency for a 160,000-cell world.
        self.adjacency = adjacency if adjacency is not None else (
            neighbors if neighbors is not None else {})
        self.cells: dict = {}

    # -- extension points ---------------------------------------------

    def new_entry(self, cell: tuple) -> dict:
        """The record kept for a newly archived cell.

        Phase 2 overrides this to carry the trajectory that reached the
        cell, and overrides :meth:`observe` to fill it in.
        """
        return {"chosen": 0, "seen": 1, "chosen_since_new": 0}

    # -- the algorithm -------------------------------------------------

    def observe(self, visited, chosen_from=None, **kwargs) -> int:
        """Fold a finished episode in; returns how many cells were new."""
        fresh = 0
        for cell in visited:
            entry = self.cells.get(cell)
            if entry is None:
                self.cells[cell] = self.new_entry(cell)
                fresh += 1
            else:
                entry["seen"] += 1
        if fresh and chosen_from in self.cells:
            self.cells[chosen_from]["chosen_since_new"] = 0
        return fresh

    def neighbors(self, cell: tuple):
        if callable(self.adjacency):
            return self.adjacency(cell)
        return self.adjacency.get(cell, ())

    def score(self, cell: tuple) -> float:
        entry = self.cells[cell]
        params = self.params
        total = 0.0
        for attribute in ATTRIBUTES:
            total += params["w_a"] * (
                1.0 / (entry[attribute] + params["eps1"])
            ) ** params["p_a"] + params["eps2"]
        for neighbor in self.neighbors(cell):
            if neighbor not in self.cells:
                total += params["w_n"]
        return total + 1.0

    def select(self):
        if not self.cells:
            return None
        cells = list(self.cells)
        scores = np.array([self.score(c) for c in cells], dtype=float)
        chosen = cells[int(self.rng.choice(len(cells),
                                           p=scores / scores.sum()))]
        entry = self.cells[chosen]
        entry["chosen"] += 1
        entry["chosen_since_new"] += 1
        return chosen

    def summary(self) -> dict:
        return {"cells": len(self.cells),
                "chosen": sum(e["chosen"] for e in self.cells.values())}


class ArchiveService:
    """Holds one :class:`LayoutArchive` per world.

    The body of the Ray actor, kept as a plain class so it can be
    unit-tested and subclassed without a cluster running. Every method
    takes a fingerprint, so one service serves every world a sweep
    touches without them ever mixing.
    """

    #: Subclass and repoint this to archive something richer.
    archive_class = LayoutArchive

    def __init__(self, params: dict | None = None, seed: int = 0):
        self.params = dict(params or {})
        self.seed = seed
        self.archives: dict = {}

    def register(self, fingerprint: str, adjacency: dict) -> bool:
        """Ensure a world has an archive. True if one was created.

        Adjacency is sent once per world rather than per call: it is
        the world's geometry, and it does not change.
        """
        if fingerprint in self.archives:
            return False
        self.archives[fingerprint] = self.archive_class(
            self.params, self.seed, adjacency)
        logger.debug("archive registered for %s (%d cells of geometry)",
                     fingerprint[:8], len(adjacency))
        return True

    def observe(self, fingerprint: str, visited, chosen_from=None,
                **kwargs) -> int:
        return self.archives[fingerprint].observe(
            visited, chosen_from, **kwargs)

    def select(self, fingerprint: str):
        return self.archives[fingerprint].select()

    def update_and_select(self, fingerprint: str, visited,
                          chosen_from=None, **kwargs):
        """The episode-boundary operation, as one round trip.

        Observe then select is the order the algorithm requires, and
        doing both in one call keeps the cost at one RPC per episode
        rather than two.
        """
        self.observe(fingerprint, visited, chosen_from, **kwargs)
        return self.select(fingerprint)

    def summary(self, fingerprint: str | None = None) -> dict:
        if fingerprint is not None:
            return self.archives[fingerprint].summary()
        return {key: archive.summary()
                for key, archive in self.archives.items()}


def remote_archive(service_class: type = ArchiveService,
                   **actor_options) -> Callable:
    """Turn an :class:`ArchiveService` (or subclass) into a Ray actor.

    Ray's ``@ray.remote`` decorator does not compose with subclassing,
    so the class stays plain and is wrapped here instead -- which is
    also what keeps it testable without Ray.
    """
    import ray

    return ray.remote(**actor_options)(service_class) if actor_options \
        else ray.remote(service_class)


def adjacency_of(layout) -> dict:
    """The world's 4-adjacency over free cells, as plain data."""
    free = set(layout.free_cells)
    return {
        cell: [n for n in layout.base.neighbors(cell) if n in free]
        for cell in free
    }
