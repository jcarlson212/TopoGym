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
    # | "around" (ring about the grid center, evenly spaced by angle)
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
    #: Minimum graph distance, over free cells, between any two chamber
    #: doors -- and with it a guarantee that no two doors touch. Zero
    #: leaves placement unconstrained, which is every family that does
    #: not care. A positive value makes the *separation* the specimen's
    #: defining property rather than a by-product of wherever the
    #: placement policy happened to put things: it is enforced by
    #: rejecting attempts, so chambers can be placed at random (seeds
    #: vary the arrangement) while the guarantee holds for every seed
    #: that generates at all. Set it above an episode horizon and no
    #: single episode can reach two doors -- the premise a chamber-count
    #: experiment needs if the count is to be isolated from the geometry
    #: around it.
    min_door_distance: int = 0
    #: Radius of the "around" ring, in cells. Zero keeps the historical
    #: behaviour -- the largest radius that still fits inside the margin,
    #: which pins the ring to the world's edge and makes the two grow
    #: together. A positive value decouples them: the ring stays put
    #: while the world around it can be made as large as wanted, so the
    #: boundary can be pushed beyond anything a step budget reaches.
    #: That is the difference between a family where following the wall
    #: leads you from one chamber to the next and one where the wall is
    #: never seen, and only the structure the agent has encircled is
    #: there to be exploited.
    ring_radius: int = 0
    #: Mark every chamber door with the door texture slot, the way the
    #: Texture scenarios do. Off leaves a world with no local signal at
    #: all, which is what the chamber-count families were built as: the
    #: only thing telling an agent a door is there is having walked
    #: into it. On makes "this cell is a door" observable, so a method
    #: carrying a door-gated score has something to read.
    #:
    #: It is a separate specimen rather than a setting, because it
    #: changes what the family measures -- topology alone becomes
    #: topology plus a local cue -- and both questions are worth
    #: asking. Applied after generation, so a textured world and its
    #: plain twin share a layout cell for cell.
    door_textures: bool = False

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
