"""Generator configurations.

A config plus a seed fully determines an environment layout — that pair is
the reproducibility unit used everywhere (benchmarks pin both).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: 2D base maps and presets accepted by :class:`TopoGenConfig2D.base`.
#: Presets: "annulus" = square + one large central hole; "x_holes" =
#: square + ``n_base_holes`` large holes.
BASES_2D = ("square", "cylinder", "torus", "mobius", "klein", "rp2",
            "annulus", "x_holes")


@dataclass(frozen=True)
class TopoGenConfig2D:
    """Configuration for the 2D environment generator."""

    base: str = "square"
    size: int | tuple = 15
    style: str = "rooms"  # "rooms" | "maze" | "zigzag" (controls)

    # -- undirected features ------------------------------------------------
    n_holes: int = 2
    n_chambers: int = 1
    n_decoys: int = 1
    n_base_holes: int = 4  # only used by the "x_holes" preset
    hole_shapes: tuple = ("rect", "disc", "blob", "plus")
    hole_size: tuple = (2, 4)  # inclusive scale range
    chamber_size: tuple = (4, 6)  # outer side length range
    door_tries: tuple = (1, 4)  # bumps to open a hidden door, inclusive range

    # -- partitions (bridge-finding) ------------------------------------------
    n_partitions: int = 0  # dividing lines across the world, with passages
    partition_gaps: tuple = (1, 2)  # passages per partition, inclusive range
    partition_hidden_gaps: tuple = (0, 1)  # of which, hidden bump-doors
    partition_material: str = "wall"  # "wall" (opaque) | "moat" (a pit:
    # blocks movement but not sight, so the far side is visible)

    # -- targets (override counts) ------------------------------------------
    target_b1: int | None = None  # solves n_holes if set

    # -- task ----------------------------------------------------------------
    goal_in_chamber: bool = False
    max_attempts: int = 80

    def to_dict(self) -> dict:
        return asdict(self)
