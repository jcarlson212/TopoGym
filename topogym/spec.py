"""Compositional topology specifications: build environments from topology.

A *spec* describes a space and its features; compiling it produces a
Gymnasium environment with certified metadata. Topology comes first, the
task is layered on, and specs compose::

    from topogym.spec import Torus

    env = Torus(15).holes(3).chambers(1).compile(seed=7)

Specs are immutable; every modifier returns a new spec, so partial builds
can be reused and swept::

    base = Torus(15).chambers(1)
    envs = [base.holes(k).compile(seed=0) for k in range(1, 5)]
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from topogym.complexes.cell_complex import CellComplex2D
from topogym.core.homology import free_complex_2d
from topogym.core.metadata import TopologyMetadata
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import Layout, generate_2d

if TYPE_CHECKING:
    from topogym.envs.topo2d import TopoGrid2DEnv

__all__ = [
    "Annulus", "Cylinder", "Klein", "Mobius", "RP2", "Spec2D", "Square",
    "Torus", "XHoles",
]


def _pair(value: int | tuple) -> tuple:
    return (value, value) if isinstance(value, int) else tuple(value)


@dataclass(frozen=True)
class Spec2D:
    """A 2D surface plus features; compiles to a ``TopoGrid2DEnv``."""

    cfg: TopoGenConfig2D

    @property
    def name(self) -> str:
        return self.cfg.base

    def _with(self, **kw) -> Spec2D:
        return Spec2D(dataclasses.replace(self.cfg, **kw))

    # -- features ------------------------------------------------------------

    def holes(self, n: int) -> Spec2D:
        """``n`` solid obstacles: +1 loop (b1) each."""
        return self._with(n_holes=n)

    def chambers(self, n: int) -> Spec2D:
        """``n`` rooms with hidden bump-doors: +1 loop each, gated inside."""
        return self._with(n_chambers=n)

    def decoys(self, n: int) -> Spec2D:
        """``n`` chamber look-alikes with no entrance."""
        return self._with(n_decoys=n)

    def partitions(self, n: int, gaps: int | tuple | None = None,
                   hidden: int | tuple | None = None,
                   material: str | None = None) -> Spec2D:
        """``n`` dividing lines with bridge passages (bottlenecks)."""
        kw: dict = {"n_partitions": n}
        if gaps is not None:
            kw["partition_gaps"] = _pair(gaps)
        if hidden is not None:
            kw["partition_hidden_gaps"] = _pair(hidden)
        if material is not None:
            kw["partition_material"] = material
        return self._with(**kw)

    # -- style / targets / task ----------------------------------------------

    def maze(self) -> Spec2D:
        return self._with(style="maze")

    def zigzag(self) -> Spec2D:
        return self._with(style="zigzag")

    def target_b1(self, b1: int) -> Spec2D:
        """Solve the number of holes so the free space has ``b1`` loops."""
        return self._with(target_b1=b1)

    def door_tries(self, lo: int, hi: int | None = None) -> Spec2D:
        return self._with(door_tries=(lo, hi if hi is not None else lo))

    def goal_in_chamber(self, flag: bool = True) -> Spec2D:
        return self._with(goal_in_chamber=flag)

    # -- realization ---------------------------------------------------------

    def layout(self, seed: int = 0) -> Layout:
        """Generate the layout for ``seed`` (certified metadata attached)."""
        return generate_2d(self.cfg, seed)

    def metadata(self, seed: int = 0) -> TopologyMetadata:
        return self.layout(seed).metadata

    def complex(self, seed: int = 0) -> CellComplex2D:
        """The free space's cell complex (faces keyed by env cells)."""
        lay = self.layout(seed)
        return free_complex_2d(
            (c, lay.base.face_cycle(c)) for c in lay.free_cells
        )

    def compile(self, seed: int | None = None, **env_kwargs) -> TopoGrid2DEnv:
        """A ``TopoGrid2DEnv``; ``seed=None`` regenerates per episode."""
        from topogym.envs import TopoGrid2DEnv

        return TopoGrid2DEnv(config=self.cfg, layout_seed=seed, **env_kwargs)


def _bare_2d(base: str, size: int | tuple, **kw) -> Spec2D:
    cfg = TopoGenConfig2D(
        base=base, size=size, n_holes=0, n_chambers=0, n_decoys=0, **kw
    )
    return Spec2D(cfg)


def Square(size: int | tuple = 15) -> Spec2D:
    """A disc: the filled square, one boundary circle."""
    return _bare_2d("square", size)


def Cylinder(size: int | tuple = 15) -> Spec2D:
    """S^1 x [0, 1]: wraps in x, walls in y."""
    return _bare_2d("cylinder", size)


def Torus(size: int | tuple = 15) -> Spec2D:
    """T^2 = S^1 x S^1: wraps both ways."""
    return _bare_2d("torus", size)


def Mobius(size: int | tuple = 15) -> Spec2D:
    """The Möbius band: crossing the x-seam mirrors the agent's frame."""
    return _bare_2d("mobius", size)


def Klein(size: int | tuple = 15) -> Spec2D:
    """The Klein bottle: closed, non-orientable, H1 torsion Z/2."""
    return _bare_2d("klein", size)


def RP2(size: int | tuple = 15) -> Spec2D:
    """The real projective plane: antipodal gluing on both seams."""
    return _bare_2d("rp2", size)


def Annulus(size: int | tuple = 15) -> Spec2D:
    """A disc with one large central hole: b1 = 1."""
    return _bare_2d("annulus", size)


def XHoles(size: int | tuple = 15, n: int = 4) -> Spec2D:
    """A disc with ``n`` large holes: b1 = n."""
    return _bare_2d("x_holes", size, n_base_holes=n)
