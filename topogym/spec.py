"""Compositional topology specifications: build environments from topology.

A *spec* describes a space and its features; compiling it produces a
Gymnasium environment with certified metadata. Topology comes first, the
task is layered on, and specs compose::

    from topogym.spec import Annulus, Circle, Torus

    env = Torus(15).holes(3).chambers(1).compile(seed=7)

    solid = Annulus(15) * Circle(8)          # a 3D product space
    env = solid.compile(seed=3)
    env.unwrapped.topology.product           # Künneth-certified provenance

Specs are immutable; every modifier returns a new spec, so partial builds
can be reused and swept::

    base = Torus(15).chambers(1)
    envs = [base.holes(k).compile(seed=0) for k in range(1, 5)]

Products
--------
``a * b`` (or ``a.product(b)``):

- 1D x 1D is the corresponding surface spec: ``Circle(m) * Circle(n)`` is
  a torus, ``Circle * Interval`` a cylinder, ``Interval * Interval`` a
  square — with every 2D modifier available on the result.
- (flip-free 2D) x 1D compiles to a real 3D environment: the 2D layout's
  obstacles are lifted along the third axis, and the metadata's homology is
  computed on the product free space *and* cross-checked against the
  Künneth formula applied to the factors' certified homology.
- Products whose 2D factor has an orientation-reversing seam (Möbius,
  Klein, RP^2) or is the cube-sphere still expose ``.complex()`` — GUDHI
  homology of the true product complex — but cannot be compiled to an
  environment yet (a non-orientable 3D environment needs 3D frame
  transport).

Doors are not liftable (a one-way column is not one door), so product
factors must be door-free; holes, mazes, and open partitions all lift.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from topogym.complexes.cell_complex import CellComplex1D, CellComplex2D
from topogym.complexes.product import ProductComplex, kunneth_betti
from topogym.core.basemap import Boundary, RectGluing2D, RectGluing3D
from topogym.core.constants import GOAL, HOLE, WALL
from topogym.core.homology import analyze_3d, free_complex_2d
from topogym.core.metadata import TopologyMetadata, homology_strings
from topogym.generation.config import TopoGenConfig2D, TopoGenConfig3D
from topogym.generation.generator import (
    GenerationError,
    Layout,
    generate_2d,
    generate_3d,
)
from topogym.generation.graph import asymmetry_block, connectivity_block

__all__ = [
    "Annulus", "Box", "Circle", "Cylinder", "Interval", "Klein", "Mobius",
    "Product", "RP2", "Shell", "SolidTorus", "Sphere", "Spec1D", "Spec2D",
    "Spec3D", "ProductSpec", "Square", "Torus", "Torus3", "XHoles",
]


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


# ---------------------------------------------------------------------------
# 1D
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec1D:
    """A 1D space: an interval or a circle of ``n`` cells."""

    kind: str  # "interval" | "circle"
    n: int

    def __post_init__(self):
        if self.kind not in ("interval", "circle"):
            raise ValueError(f"unknown 1D kind {self.kind!r}")
        if self.n < (1 if self.kind == "interval" else 2):
            raise ValueError(f"{self.kind} needs more cells than {self.n}")

    @property
    def name(self) -> str:
        return self.kind

    def complex(self) -> CellComplex1D:
        n = self.n
        if self.kind == "interval":
            return CellComplex1D((i, (i, i + 1)) for i in range(n))
        return CellComplex1D((i, (i, (i + 1) % n)) for i in range(n))

    def betti(self, field: int = 2) -> tuple:
        return (1, 1) if self.kind == "circle" else (1, 0)

    def __mul__(self, other):
        return Product(self, other)


def Interval(n: int = 8) -> Spec1D:
    """``n`` cells in a row — the unit interval."""
    return Spec1D("interval", n)


def Circle(n: int = 8) -> Spec1D:
    """``n`` cells around a loop — the circle S^1."""
    return Spec1D("circle", n)


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec2D:
    """A 2D surface plus features; compiles to a ``TopoGrid2DEnv``."""

    cfg: TopoGenConfig2D

    @property
    def name(self) -> str:
        return self.cfg.base

    def _with(self, **kw) -> Spec2D:
        return Spec2D(dataclasses.replace(self.cfg, **kw))

    # -- undirected features -------------------------------------------------

    def holes(self, n: int) -> Spec2D:
        """``n`` solid obstacles: +1 loop (b1) each."""
        return self._with(n_holes=n)

    def chambers(self, n: int) -> Spec2D:
        """``n`` rooms with hidden bump-doors: +1 loop each, gated inside."""
        return self._with(n_chambers=n)

    def decoys(self, n: int) -> Spec2D:
        """``n`` chamber look-alikes with no entrance."""
        return self._with(n_decoys=n)

    def partitions(self, n: int, gaps=None, hidden=None,
                   material=None) -> Spec2D:
        """``n`` dividing lines with bridge passages (bottlenecks)."""
        kw = {"n_partitions": n}
        if gaps is not None:
            kw["partition_gaps"] = _pair(gaps)
        if hidden is not None:
            kw["partition_hidden_gaps"] = _pair(hidden)
        if material is not None:
            kw["partition_material"] = material
        return self._with(**kw)

    # -- directed features ----------------------------------------------------

    def trap_rooms(self, n: int) -> Spec2D:
        """``n`` rooms with a one-way door inward (absorbing regions)."""
        return self._with(n_trap_rooms=n)

    def airlocks(self, n: int) -> Spec2D:
        """``n`` rooms with one-way in + one-way out (directed circuits)."""
        return self._with(n_airlocks=n)

    def trapdoor_rooms(self, n: int) -> Spec2D:
        """``n`` rooms entered by trapdoor, escaped by hidden door."""
        return self._with(n_trapdoor_rooms=n)

    # -- style / targets / task ------------------------------------------------

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

    # -- realization -------------------------------------------------------------

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

    def compile(self, seed: int | None = None, **env_kwargs):
        """A ``TopoGrid2DEnv``; ``seed=None`` regenerates per episode."""
        from topogym.envs import TopoGrid2DEnv

        return TopoGrid2DEnv(config=self.cfg, layout_seed=seed, **env_kwargs)

    def product(self, other):
        return Product(self, other)

    __mul__ = product


def _bare_2d(base: str, size, **kw) -> Spec2D:
    cfg = TopoGenConfig2D(
        base=base, size=size, n_holes=0, n_chambers=0, n_decoys=0, **kw
    )
    return Spec2D(cfg)


def Square(size=15) -> Spec2D:
    """A disc: the filled square, one boundary circle."""
    return _bare_2d("square", size)


def Cylinder(size=15) -> Spec2D:
    """S^1 x [0, 1]: wraps in x, walls in y."""
    return _bare_2d("cylinder", size)


def Torus(size=15) -> Spec2D:
    """T^2 = S^1 x S^1: wraps both ways."""
    return _bare_2d("torus", size)


def Mobius(size=15) -> Spec2D:
    """The Möbius band: crossing the x-seam mirrors the agent's frame."""
    return _bare_2d("mobius", size)


def Klein(size=15) -> Spec2D:
    """The Klein bottle: closed, non-orientable, H1 torsion Z/2."""
    return _bare_2d("klein", size)


def RP2(size=15) -> Spec2D:
    """The real projective plane: antipodal gluing on both seams."""
    return _bare_2d("rp2", size)


def Sphere(size=8) -> Spec2D:
    """S^2 as the surface of a cube (six ``size`` x ``size`` faces)."""
    return _bare_2d("sphere", size)


def Annulus(size=15) -> Spec2D:
    """A disc with one large central hole: b1 = 1."""
    return _bare_2d("annulus", size)


def XHoles(size=15, n: int = 4) -> Spec2D:
    """A disc with ``n`` large holes: b1 = n."""
    return _bare_2d("x_holes", size, n_base_holes=n)


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spec3D:
    """A 3D space plus features; compiles to a ``TopoGrid3DEnv``."""

    cfg: TopoGenConfig3D

    @property
    def name(self) -> str:
        return self.cfg.base

    def _with(self, **kw) -> Spec3D:
        return Spec3D(dataclasses.replace(self.cfg, **kw))

    def rings(self, n: int) -> Spec3D:
        """``n`` solid-torus obstacles: +1 loop and +1 shell each."""
        return self._with(n_rings=n)

    def blobs(self, n: int) -> Spec3D:
        """``n`` solid obstacles: +1 enclosing shell (b2) each."""
        return self._with(n_blobs=n)

    def chambers(self, n: int) -> Spec3D:
        return self._with(n_chambers=n)

    def decoys(self, n: int) -> Spec3D:
        return self._with(n_decoys=n)

    def partitions(self, n: int, gaps=None, hidden=None,
                   material=None) -> Spec3D:
        kw = {"n_partitions": n}
        if gaps is not None:
            kw["partition_gaps"] = _pair(gaps)
        if hidden is not None:
            kw["partition_hidden_gaps"] = _pair(hidden)
        if material is not None:
            kw["partition_material"] = material
        return self._with(**kw)

    def trap_rooms(self, n: int) -> Spec3D:
        return self._with(n_trap_rooms=n)

    def airlocks(self, n: int) -> Spec3D:
        return self._with(n_airlocks=n)

    def trapdoor_rooms(self, n: int) -> Spec3D:
        return self._with(n_trapdoor_rooms=n)

    def maze(self) -> Spec3D:
        return self._with(style="maze")

    def target_b1(self, b1: int) -> Spec3D:
        return self._with(target_b1=b1)

    def target_b2(self, b2: int) -> Spec3D:
        return self._with(target_b2=b2)

    def door_tries(self, lo: int, hi: int | None = None) -> Spec3D:
        return self._with(door_tries=(lo, hi if hi is not None else lo))

    def goal_in_chamber(self, flag: bool = True) -> Spec3D:
        return self._with(goal_in_chamber=flag)

    def layout(self, seed: int = 0) -> Layout:
        return generate_3d(self.cfg, seed)

    def metadata(self, seed: int = 0) -> TopologyMetadata:
        return self.layout(seed).metadata

    def compile(self, seed: int | None = None, **env_kwargs):
        """A ``TopoGrid3DEnv``; ``seed=None`` regenerates per episode."""
        from topogym.envs import TopoGrid3DEnv

        return TopoGrid3DEnv(config=self.cfg, layout_seed=seed, **env_kwargs)


def _bare_3d(base: str, size) -> Spec3D:
    cfg = TopoGenConfig3D(
        base=base, size=size, n_rings=0, n_blobs=0, n_chambers=0, n_decoys=0
    )
    return Spec3D(cfg)


def Box(size=12) -> Spec3D:
    """A solid box: contractible."""
    return _bare_3d("box", size)


def SolidTorus(size=12) -> Spec3D:
    """D^2 x S^1: wraps in x, walls in y and z."""
    return _bare_3d("solid_torus", size)


def Torus3(size=10) -> Spec3D:
    """The 3-torus T^3: wraps on all axes."""
    return _bare_3d("torus3", size)


def Shell(size=12) -> Spec3D:
    """A box with a large central void: b2 = 1."""
    return _bare_3d("shell", size)


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

_RECT_RULES_BY_BASE = {
    "square": (Boundary.WALL, Boundary.WALL),
    "cylinder": (Boundary.WRAP, Boundary.WALL),
    "torus": (Boundary.WRAP, Boundary.WRAP),
    "annulus": (Boundary.WALL, Boundary.WALL),
    "x_holes": (Boundary.WALL, Boundary.WALL),
}

_1D_TO_RULE = {"interval": Boundary.WALL, "circle": Boundary.WRAP}


def Product(a, b):
    """The product space ``a x b`` (also spelled ``a * b``)."""
    if isinstance(a, Spec1D) and isinstance(b, Spec1D):
        return _product_1d_1d(a, b)
    if isinstance(a, Spec2D) and isinstance(b, Spec1D):
        return ProductSpec(a, b)
    if isinstance(a, Spec1D) and isinstance(b, Spec2D):
        return ProductSpec(b, a)
    raise TypeError(
        f"unsupported product {type(a).__name__} x {type(b).__name__}; "
        "supported: 1D x 1D (a surface), 2D x 1D (a 3D space). For the "
        "homology of higher products, build "
        "topogym.complexes.ProductComplex(a.complex(), b.complex())."
    )


def _product_1d_1d(a: Spec1D, b: Spec1D) -> Spec2D:
    # Normalize so a wrap axis is x (matching the base-map convention).
    kinds = (a.kind, b.kind)
    if kinds == ("circle", "circle"):
        return _bare_2d("torus", (a.n, b.n))
    if kinds == ("interval", "interval"):
        return _bare_2d("square", (a.n, b.n))
    circle, interval = (a, b) if a.kind == "circle" else (b, a)
    return _bare_2d("cylinder", (circle.n, interval.n))


@dataclass(frozen=True)
class ProductSpec:
    """A (2D surface) x (1D space) product.

    ``complex()`` works for every factor pair. ``layout()``/``compile()``
    require a flip-free rectangular 2D factor with no doors: obstacles lift
    along the product axis, doors do not.
    """

    surface: Spec2D
    line: Spec1D

    @property
    def name(self) -> str:
        return f"{self.surface.name} x {self.line.name}"

    def complex(self, seed: int = 0) -> ProductComplex:
        """The product of the factors' cell complexes (any factors)."""
        return ProductComplex(self.surface.complex(seed), self.line.complex())

    def betti(self, field: int = 2, seed: int = 0,
              method: str = "kunneth") -> tuple:
        return self.complex(seed).betti(field, method=method)

    def product(self, other):
        return Product(self, other)  # raises with guidance: 4D+

    __mul__ = product

    # -- lifting to a 3D layout ------------------------------------------------

    def layout(self, seed: int = 0) -> Layout:
        """The lifted 3D layout, with Künneth-cross-checked metadata."""
        base_name = self.surface.cfg.base
        rules_2d = _RECT_RULES_BY_BASE.get(base_name)
        if rules_2d is None:
            raise NotImplementedError(
                f"cannot compile {self.name!r}: the 2D factor must be a "
                "flip-free rectangular base (square/cylinder/torus/annulus/"
                "x_holes); Möbius, Klein, RP^2 and sphere products expose "
                ".complex() for homology but have no 3D environment yet"
            )
        if self.line.n < 3:
            raise GenerationError(
                f"the 1D factor needs >= 3 cells to lift (got {self.line.n})"
            )
        lay2 = generate_2d(self.surface.cfg, seed)
        if lay2.doors:
            raise NotImplementedError(
                f"cannot lift {self.name!r}: the 2D factor has doors "
                "(chambers/trap rooms/hidden partition gaps); doors do not "
                "lift to a product — use holes, mazes, or open partitions"
            )
        base2 = lay2.base
        assert isinstance(base2, RectGluing2D)
        nz = self.line.n
        rules = (base2.rule_x, base2.rule_y, _1D_TO_RULE[self.line.kind])
        base3 = RectGluing3D((base2.width, base2.height, nz), rules)

        cell_types = {}
        for (x, y), t in lay2.cell_types.items():
            if t in (WALL, HOLE):
                for z in range(nz):
                    cell_types[(x, y, z)] = t
        free = [
            (x, y, z)
            for z in range(nz) for (x, y) in lay2.free_cells
        ]
        start = (*lay2.start, 0)
        goal_z = nz // 2 if self.line.kind == "circle" else nz - 1
        goal = (*lay2.goal, goal_z)
        cell_types[goal] = GOAL

        features = [
            dataclasses.replace(
                f,
                cells=tuple(
                    (x, y, z) for (x, y) in f.cells for z in range(nz)
                ),
                interior=(),
                doors=(),
            )
            for f in lay2.features
        ]

        layout = Layout(
            dim=3, base=base3, cell_types=cell_types, doors={},
            start=start, goal=goal, features=features, free_cells=free,
        )
        layout.metadata = self._metadata(lay2, base3, layout, seed)
        return layout

    def _metadata(self, lay2, base3, layout, seed) -> TopologyMetadata:
        meta2 = lay2.metadata
        line_betti = self.line.betti()

        summary = analyze_3d(
            base3.cube_corners(c) for c in layout.free_cells
        )
        expected = kunneth_betti(meta2.betti_z2, line_betti)
        if summary.betti_z2 != expected:
            raise GenerationError(
                f"product homology mismatch for {self.name!r}: computed "
                f"{summary.betti_z2}, Künneth expects {expected}"
            )

        # 2D free spaces have torsion-free homology unless they are a full
        # closed non-orientable surface — impossible for flip-free bases —
        # so the Künneth formula over Q has no Tor terms and betti_q is
        # certified whenever the factor's is.
        certified_q = meta2.certified.get("betti_q", False) and not (
            meta2.h1_torsion
        )
        betti_q = (
            kunneth_betti(meta2.betti_q, line_betti)
            if certified_q else None
        )

        free_set = set(layout.free_cells)
        asym = asymmetry_block(
            free_set, {}, base3.neighbors, layout.start, layout.goal
        )
        asym["feature_counts"] = {
            "trap_room": 0, "airlock": 0, "trapdoor_room": 0,
        }

        return TopologyMetadata(
            dim=3,
            base_map=self.name,
            base={
                k: getattr(base3.info, k)
                for k in base3.info.__dataclass_fields__
            },
            size=(base3.size[0], base3.size[1], base3.size[2]),
            style="product",
            layout_seed=seed,
            n_holes=meta2.n_holes,
            n_chambers=0,
            n_decoys=0,
            door_tries=(),
            n_cells=len(base3.cells()),
            n_free_cells=len(layout.free_cells),
            betti_z2=summary.betti_z2,
            euler_characteristic=summary.euler_characteristic,
            orientable=None, genus=None, demigenus=None,
            n_boundary_components=None,
            betti_q=betti_q,
            betti_q_expected=betti_q if betti_q is not None else
            kunneth_betti(meta2.betti_q_expected, line_betti),
            h1_torsion=() if certified_q else None,
            asymmetry=asym,
            connectivity=connectivity_block(free_set, base3.neighbors),
            n_partitions=meta2.n_partitions,
            product={
                "factors": [
                    {"name": self.surface.name, "dim": 2,
                     "betti_z2": list(meta2.betti_z2),
                     "betti_q": list(meta2.betti_q)
                     if meta2.betti_q else None},
                    {"name": self.line.name, "dim": 1,
                     "betti_z2": list(line_betti),
                     "betti_q": list(line_betti)},
                ],
                "kunneth_cross_check": "passed",
            },
            certified={
                "betti_z2": True,
                "betti_q": certified_q,
                "h1_torsion": certified_q,
                "asymmetry": True,
                "connectivity": True,
                "genus": False,
            },
            homology=homology_strings(
                betti_q, (), summary.betti_z2
            ),
        )

    def metadata(self, seed: int = 0) -> TopologyMetadata:
        return self.layout(seed).metadata

    def compile(self, seed: int = 0, **env_kwargs):
        """A ``TopoGrid3DEnv`` on the lifted product layout."""
        from topogym.envs import TopoGrid3DEnv

        return TopoGrid3DEnv(layout=self.layout(seed), **env_kwargs)
