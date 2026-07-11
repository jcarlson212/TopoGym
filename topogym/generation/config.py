"""Generator configurations.

A config plus a seed fully determines an environment layout — that pair is
the reproducibility unit used everywhere (benchmarks pin both).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: 2D base maps and presets accepted by :class:`TopoGenConfig2D.base`.
#: Presets: "annulus" = square + one large central hole; "x_holes" =
#: square + ``n_base_holes`` large holes.
BASES_2D = ("square", "cylinder", "torus", "mobius", "klein", "rp2", "sphere",
            "annulus", "x_holes")

#: 3D base maps and presets. Preset: "shell" = box + one large central void.
BASES_3D = ("box", "solid_torus", "torus3", "shell")


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

    # -- directed features (asymmetric traversability) -----------------------
    n_trap_rooms: int = 0  # one-way door inward: absorbing region
    n_airlocks: int = 0  # one-way in + one-way out: directed circuit
    n_trapdoor_rooms: int = 0  # trapdoor in + hidden bump-door escape
    trapdoor_escape_tries: tuple = (3, 6)

    # -- targets (override counts) ------------------------------------------
    target_b1: int | None = None  # solves n_holes if set

    # -- task ----------------------------------------------------------------
    goal_in_chamber: bool = False
    max_attempts: int = 80

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TopoGenConfig3D:
    """Configuration for the 3D environment generator."""

    base: str = "box"
    size: int | tuple = 12
    style: str = "rooms"  # "rooms" | "maze" (control)

    # -- undirected features ------------------------------------------------
    n_rings: int = 1  # solid-torus obstacles: +1 loop (b1) and +1 shell (b2)
    n_blobs: int = 1  # solid obstacles: +1 enclosing shell (b2)
    n_chambers: int = 1
    n_decoys: int = 1
    blob_shapes: tuple = ("box", "ball", "blob")
    blob_size: tuple = (2, 3)
    ring_size: tuple = (3, 5)
    chamber_size: tuple = (4, 5)
    door_tries: tuple = (1, 4)

    # -- partitions (bridge-finding) -------------------------------------------
    n_partitions: int = 0  # dividing planes across the world, with tunnels
    partition_gaps: tuple = (1, 2)
    partition_hidden_gaps: tuple = (0, 1)
    partition_material: str = "wall"  # "wall" | "moat" (see 2D config)

    # -- directed features ---------------------------------------------------
    n_trap_rooms: int = 0
    n_airlocks: int = 0
    n_trapdoor_rooms: int = 0
    trapdoor_escape_tries: tuple = (3, 6)

    # -- targets --------------------------------------------------------------
    target_b1: int | None = None  # solves n_rings if set
    target_b2: int | None = None  # solves n_blobs if set

    # -- task ------------------------------------------------------------------
    goal_in_chamber: bool = False
    max_attempts: int = 80

    def to_dict(self) -> dict:
        return asdict(self)
