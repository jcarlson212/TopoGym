"""GridWorld2D-Top environments: the canonical corner-chamber layouts on
each edge-identification topology (see :mod:`topogym.generation.top`)."""

from __future__ import annotations

from topogym.envs.topo2d import TopoGrid2DEnv
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import Layout
from topogym.generation.top import TOPOLOGIES, build_top


class TopGrid2DEnv(TopoGrid2DEnv):
    """A :class:`TopoGrid2DEnv` running the canonical Top layout."""

    def __init__(self, topology: str, *, size: int = 50, **kwargs):
        if topology not in TOPOLOGIES:
            raise ValueError(
                f"unknown topology {topology!r}; choose from "
                f"{sorted(TOPOLOGIES)}"
            )
        # (self.topology is the certified-metadata property; the
        # identification lives under its own name.)
        self.identification = topology
        self._top_size = size
        kwargs.setdefault(
            "config",
            TopoGenConfig2D(base=TOPOLOGIES[topology], size=size),
        )
        super().__init__(**kwargs)

    def _generate(self, seed: int) -> Layout:
        from topogym.generation.cache import cached_layout

        return cached_layout(
            ("top", self.identification, self._top_size, seed),
            lambda: build_top(self.identification, seed,
                              size=self._top_size),
        )
