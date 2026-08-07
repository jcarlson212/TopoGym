"""Generator configurations.

A config plus a seed fully determines an environment layout — that pair is
the reproducibility unit used everywhere (the registry pins both).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: 2D base maps and presets accepted by :class:`TopoGenConfig2D.base`.
#: Presets: "annulus" = square + one large central hole; "x_holes" =
#: square + ``n_base_holes`` large holes.
BASES_2D = ("square", "cylinder", "torus", "mobius", "klein", "rp2",
            "annulus", "x_holes")

#: Generation styles (the spec's "modes"; "open" is an alias of "rooms").
STYLES_2D = ("rooms", "nested", "corridor", "maze", "zigzag")


@dataclass(frozen=True)
class TopoGenConfig2D:
    """Configuration for the 2D environment generator."""

    base: str = "square"
    size: int | tuple = 15
    #: "rooms" (the spec's "open" mode: features in a free field),
    #: "nested" (concentric shells), "corridor" (tree of rooms joined by
    #: width-1 corridors), "maze" (perfect maze, optionally braided),
    #: "zigzag" (serpentine control), "spiral" (one long corridor with
    #: chambers an episode apart).
    style: str = "rooms"

    # -- undirected features ------------------------------------------------
    n_holes: int = 2
    n_chambers: int = 1
    n_decoys: int = 1
    n_base_holes: int = 4  # only used by the "x_holes" preset
    hole_shapes: tuple = ("rect", "disc", "blob", "plus")
    hole_size: tuple = (2, 4)  # inclusive scale range

    # -- rooms (chambers and decoys) -----------------------------------------
    chamber_size: tuple = (4, 6)  # outer side range (when *_side is None)
    chamber_side: int | None = None  # exact outer side; overrides the range
    decoy_side: int | None = None  # exact decoy side; defaults to chamber's
    chamber_shape: str = "square"  # square | circle | triangle | star | mixed
    # -- placement policies ---------------------------------------------
    # The macro arrangement of a family is part of its identity, so the
    # registry pins it while seeds keep varying the micro detail (door
    # sides, goal cell, shapes). Set ``placement="random"`` to drop the
    # whole arrangement back into the sampled tier.
    chamber_placement: str = "random"  # "random" | "center" | "perimeter"
    decoy_placement: str = "random"  # "random" | "around" (ring about
    # the grid center, evenly spaced by angle)
    start_placement: str = "random"  # "random" | "bottom_left" | "center"
    placement_jitter: int = 0  # cells of uniform perturbation applied to
    # policy anchors; 0 in the registry (canonical specimens), > 0 in
    # benchmark splits so instances differ while the grammar holds
    placement: str | None = None  # master override; "random" ignores the
    # three policies above
    decoy_shape: str = "square"  # area-matched at equal side (never
    # confounds shape with size)
    min_sep: int = 2  # minimum pairwise Chebyshev separation between walls

    # -- doors ---------------------------------------------------------------
    door_kind: str = "bump"  # "bump" (hidden, opens after tries) | "open"
    # (a width-1 gap in the wall — the spec registry's door convention)
    doors_per_chamber: int = 1
    door_corridor_len: int = 0  # dead-end corridor outside each door (the
    # GiveUp mechanism; lowers the entry probability)
    door_tries: tuple = (1, 4)  # bump doors: bumps to open, inclusive range

    # -- nested style ---------------------------------------------------------
    nested_depth: int = 1  # concentric shells around the innermost chamber
    shell_spacing: int = 2  # free cells between consecutive shells

    # -- corridor style --------------------------------------------------------
    rooms: int = 6  # rooms in the tree
    corridor_len: int = 3  # width-1 corridor length between rooms

    # -- spiral style (EpicChase) -----------------------------------------
    spiral_arc: int = 0  # actions between consecutive chambers along the
    # corridor; also the episode budget the family is registered with, so
    # one episode reaches exactly one chamber
    spiral_width: int = 3  # corridor width in cells (odd; the arms widen
    # about their centreline, and the arm pitch grows to match)

    # -- maze style ------------------------------------------------------------
    braid: float = 0.0  # fraction of loop-opening candidates to open; each
    # opening encloses wall and adds one H1 class

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
