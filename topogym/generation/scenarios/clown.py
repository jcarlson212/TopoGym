"""ClownChase: the reward-deception scenario."""

from __future__ import annotations

import numpy as np

from topogym.core import constants as C
from topogym.generation.config import TopoGenConfig2D
from topogym.generation.generator import generate_2d
from topogym.generation.layout import (
    GenerationError,
    Layout,
)
from topogym.generation.scenarios._shared import (
    SCENARIO_SIZES,
    _door_cells,
    _interiors,
    _mark,
)


def build_clown_chase(seed: int) -> Layout:
    """A clown wanders near sealed decoys on one side of the map, paying
    a tiny reward for every step that closes the distance to it — from a
    budget that runs out after a few thousand rewarding steps. The
    treasure chamber sits on the opposite side."""
    size = SCENARIO_SIZES["clown_chase"]
    cfg = TopoGenConfig2D(
        base="square", size=size, style="rooms",
        n_holes=0, n_chambers=1, n_decoys=3,
        chamber_side=8, decoy_side=6, door_kind="open", min_sep=3,
        goal_in_chamber=True,
    )
    for attempt in range(40):
        layout = generate_2d(cfg, seed * 1013 + attempt)
        decoys = [f for f in layout.features if f.kind == "decoy"]
        (chamber,) = [f for f in layout.features if f.kind == "chamber"]

        def centroid(cells):
            return (sum(c[0] for c in cells) / len(cells),
                    sum(c[1] for c in cells) / len(cells))

        dx = centroid([c for f in decoys for c in f.cells])[0]
        cx = centroid(chamber.cells)[0]
        # The clown's side and the treasure's side must genuinely differ.
        if abs(dx - cx) < size / 3:
            continue

        free = set(layout.free_cells)
        anchor = min(
            (c for c in free), key=lambda c: (abs(c[0] - dx), repr(c))
        )
        # Start away from the clown's side, not inside any room.
        interiors = set(_interiors(layout))
        start_side = [
            c for c in sorted(free - interiors - set(layout.doors))
            if abs(c[0] - cx) < size / 4 and c != layout.goal
        ]
        if not start_side:
            continue
        rng = np.random.default_rng(seed)
        layout.start = start_side[int(rng.integers(len(start_side)))]

        textures: dict = {}
        _mark(textures, free, C.TEX_DIRT)
        _mark(textures, interiors & free, C.TEX_INTERIOR)
        _mark(textures, _door_cells(layout), C.TEX_DOOR)
        layout.extras = {
            "textures": textures,
            "clown": {
                "anchor": anchor,
                "radius": 8,
                "budget": 2.0,        # total distractor payout
                "step_reward": 0.001,  # ~2000 rewarding steps
            },
        }
        return layout
    raise GenerationError(f"could not build ClownChase for seed {seed}")
