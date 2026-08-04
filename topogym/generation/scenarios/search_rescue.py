"""SearchRescue: one large persistent hole in a shrapnel field."""

from __future__ import annotations

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


def build_search_rescue(seed: int) -> Layout:
    """Search and rescue. A person is trapped in one large open chamber
    buried in a dense field of shrapnel. Every shard adds a small H1
    class; the victim's chamber is the only *large, persistent* hole —
    in the agent's own discovery filtration (the archive), the shards
    resolve as small transient bars while the chamber's enclosing class
    grows large and refuses to die. Persistence, not luck, finds the
    person."""
    cfg = TopoGenConfig2D(
        base="square", size=SCENARIO_SIZES["search_rescue"],
        style="rooms",
        n_holes=26, hole_shapes=("rect", "plus", "blob"),
        hole_size=(1, 2),
        n_chambers=1, n_decoys=0, chamber_side=15,
        door_kind="open", min_sep=2, goal_in_chamber=True,
    )
    layout = generate_2d(cfg, seed)
    textures: dict = {}
    free = set(layout.free_cells)
    _mark(textures, free, C.TEX_DIRT)
    _mark(textures, set(_interiors(layout)) & free, C.TEX_INTERIOR)
    _mark(textures, _door_cells(layout), C.TEX_DOOR)
    layout.extras = {"textures": textures, "person": layout.goal}
    return layout
