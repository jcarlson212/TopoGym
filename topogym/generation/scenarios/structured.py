"""Ladders and BankRobber: textured bottleneck and nested regimes."""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import generate_2d
from topogym.generation.layout import (
    Layout,
)
from topogym.generation.scenarios._shared import (
    SCENARIO_SIZES,
    _door_cells,
    _interiors,
    _mark,
)


def build_ladders(seed: int) -> Layout:
    """Platforms, ladders, and bridges: the textured bottleneck regime.
    Vertical corridors are ladders, horizontal ones bridges; the gem sits
    on the top platform."""
    # The room tree claims every lattice node: the tower fills the
    # whole world rather than a corner of it.
    cfg = TopoGenConfig2D(
        base="square", size=SCENARIO_SIZES["ladders"], style="corridor",
        rooms=25, corridor_len=4, chamber_side=6,
        n_holes=0, n_chambers=0, n_decoys=0,
    )
    layout = generate_2d(cfg, seed)

    # The gem belongs on the *top* platform (smallest y); the climb
    # starts on a bottom-row platform.
    rooms_ = [f for f in layout.features if f.kind == "room"]
    top = min(rooms_, key=lambda f: (min(c[1] for c in f.interior), f.meta["node"]))
    bottom = max(rooms_, key=lambda f: (max(c[1] for c in f.interior),
                                        f.meta["node"]))
    old_goal = layout.goal
    layout.cell_types.pop(old_goal, None)
    rng = np.random.default_rng(seed)
    goal = tuple(top.interior[int(rng.integers(len(top.interior)))])
    layout.goal = goal
    layout.cell_types[goal] = C.GOAL
    layout.start = tuple(
        bottom.interior[int(rng.integers(len(bottom.interior)))]
    )

    textures: dict = {}
    for f in rooms_:
        _mark(textures, f.interior, C.TEX_PLATFORM)
    (corr,) = [f for f in layout.features if f.kind == "corridors"]
    free = set(layout.free_cells)
    for (x, y) in corr.meta["cells"]:
        vertical = ((x, y - 1) in free) or ((x, y + 1) in free)
        _mark(textures, [(x, y)],
              C.TEX_LADDER if vertical else C.TEX_BRIDGE)
    layout.extras = {"textures": textures}
    return layout


def build_bank_robber(seed: int) -> Layout:
    """Nested rooms with the money in the center: the textured nested
    regime. Doors and hallways advertise the sequential structure
    locally; the ordering constraint stays global."""
    cfg = TopoGenConfig2D(
        base="square", size=SCENARIO_SIZES["bank_robber"], style="nested",
        nested_depth=3, shell_spacing=3, chamber_side=7,
        door_kind="open", n_holes=0, n_chambers=1, n_decoys=0,
        goal_in_chamber=True,
    )
    layout = generate_2d(cfg, seed)
    textures: dict = {}
    _mark(textures, layout.free_cells, C.TEX_DIRT)
    core_interior = set(_interiors(layout, ("chamber",)))
    shell_region = set(_interiors(layout, ("shell",)))
    free = set(layout.free_cells)
    hallway = (shell_region - core_interior) & free
    _mark(textures, hallway, C.TEX_HALLWAY)
    _mark(textures, core_interior & free, C.TEX_INTERIOR)
    _mark(textures, _door_cells(layout), C.TEX_DOOR)
    layout.extras = {"textures": textures}
    return layout
