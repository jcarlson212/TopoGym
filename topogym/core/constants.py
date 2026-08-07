"""Cell-type, action, and observation codes shared across TopoGym."""

from dataclasses import dataclass, fields
from enum import Enum, IntEnum

# Actions. Both spaces are part of the public interface, so name them
# once here rather than leaving callers to pass bare integers:
# env.step(EgocentricAction.FORWARD) says what env.step(2) only
# implies, and a type checker can tell the two spaces apart.


class ActionMode(str, Enum):
    """Which action space an environment exposes.

    A ``str`` enum, so it compares and passes through as the plain
    string the environment already accepts -- existing
    ``actions="fourway"`` keeps working.
    """

    EGOCENTRIC = "egocentric"
    FOURWAY = "fourway"

    @property
    def actions(self) -> type:
        """The action enum belonging to this mode."""
        return (EgocentricAction if self is ActionMode.EGOCENTRIC
                else FourwayAction)


class EgocentricAction(IntEnum):
    """The default ``Discrete(3)`` space: the agent turns and advances.

    An ``IntEnum``, so members are the action integers and can be
    handed straight to ``env.step``.
    """

    TURN_LEFT = 0
    TURN_RIGHT = 1
    FORWARD = 2


class FourwayAction(IntEnum):
    """``Discrete(4)`` in *screen* directions (``actions="fourway"``):
    up decreases y whatever the agent is facing."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# Bare aliases, for callers who prefer names to enum members. Defined
# from the enums so there is one source of truth.
TURN_LEFT = int(EgocentricAction.TURN_LEFT)
TURN_RIGHT = int(EgocentricAction.TURN_RIGHT)
FORWARD = int(EgocentricAction.FORWARD)
MOVE_UP = int(FourwayAction.UP)
MOVE_DOWN = int(FourwayAction.DOWN)
MOVE_LEFT = int(FourwayAction.LEFT)
MOVE_RIGHT = int(FourwayAction.RIGHT)

#: Action index -> name, per space; handy for logs and debugging.
EGOCENTRIC_ACTION_NAMES = {
    int(a): a.name.lower() for a in EgocentricAction
}
FOURWAY_ACTION_NAMES = {int(a): a.name.lower() for a in FourwayAction}

# Layout cell types (what a cell *is*).
EMPTY = 0
WALL = 1
HOLE = 2  # visually distinct impassable cell ("definitely nothing inside")
DOOR = 3  # hidden door: observed as WALL until opened by repeated bumps
GOAL = 4
HAZARD = 5  # walkable but fatal: stepping on it ends the episode
WORMHOLE = 6  # walkable teleporter: stepping on it jumps to its partner

# Observation codes (what the agent *sees*). Closed doors are observed as
# OBS_WALL — doors are hidden until opened.
OBS_EMPTY = 0
OBS_WALL = 1
OBS_HOLE = 2
OBS_DOOR_OPEN = 3
OBS_GOAL = 4
OBS_OUT_OF_WORLD = 5  # beyond a WALL-type boundary of the base map
OBS_UNSEEN = 6  # occluded by walls in the local view
OBS_AGENT = 7  # only used in "global" observations / rendering
OBS_HAZARD = 8  # the drop: visibly distinct, fatally enterable
OBS_WORMHOLE = 9  # a teleporter ("all wormholes are purple"): visibly a
# wormhole, but never *which* one — destinations must be explored

OBS_MAX = 9

#: How many distinct observation codes exist, library-wide and
#: constant across every slice. Size an embedding to *this*, never to
#: the codes a particular training slice happened to contain: hazards
#: (8) and wormholes (9) occur only in Texture worlds, so a policy
#: trained on GridWorld2D meets them for the first time at evaluation.
#: Every environment declares ``Box(0, OBS_MAX, ...)`` regardless of
#: slice, so sizing from the observation space is also safe.
OBS_CODE_COUNT = OBS_MAX + 1

# The texture block. This dataclass is the *definition* of the block --
# its width, its slot order, and what each slot means. Everything below
# it (TEXTURE_DIM, the bare TEX_* integers, TEX_SLOT_NAMES) is derived,
# and every producer and consumer of a texture block sizes itself from
# TextureSlotMap.dim rather than from a literal, so the width is stated
# in exactly one place.


@dataclass(frozen=True)
class TextureSlotMap:
    """Slot name -> slot index for the texture block.

    The block is a *semantic overlay*, not a terrain map. What a cell
    physically is -- wall, goal, hazard, and crucially "out of world"
    and "not currently visible" -- is carried by the symbolic
    observation codes (:data:`OBS_EMPTY` ... :data:`OBS_WORMHOLE`), one
    per cell, in a separate channel. Nothing here encodes terrain or
    visibility, so no slot is ever spent on "out of map" or "unseen":
    those are codes 5 and 6, and a texture block of all zeros means
    "this cell carries no semantic annotation", never "this cell is
    absent".

    That separation is what keeps the two channels independently
    meaningful. A cell can be simultaneously ``OBS_EMPTY`` and
    ``water``; an occluded cell has code ``OBS_UNSEEN`` and an all-zero
    block, because the agent cannot know its semantics either.

    Slots 0-3 are directional blocker adjacency for the cell in
    question; slots 4 onward are per-scenario semantics, assigned
    library-wide so an agent transfers between Texture scenarios. The
    whole block is identically zero in GridWorld2D and Top -- those
    slices carry no textures at all.

    **Adding a slot.** Append a field with the next free index and an
    entry in :attr:`descriptions`; never renumber an existing one.
    Indices are the wire format: a policy, a recorded observation, and
    a scenario's documented slots all refer to them by number, so
    reordering silently reinterprets data that already exists, while
    appending leaves every existing slot meaning exactly what it did.
    ``tests/envs/test_observations.py`` pins the current assignment
    against exactly that mistake. Appending does widen the block, which
    changes observation shape and so retires trained checkpoints -- it
    is source-compatible, not weight-compatible.
    """

    blocked_left: int = 0
    blocked_right: int = 1
    blocked_above: int = 2
    blocked_below: int = 3
    water: int = 4
    platform: int = 5
    ladder: int = 6
    bridge: int = 7
    door: int = 8
    hallway: int = 9
    drop_adjacent: int = 10
    ground: int = 11
    room_interior: int = 12
    on_wormhole: int = 13
    clown_near: int = 14
    on_treasure: int = 15

    #: What each slot means, and which scenarios populate it.
    descriptions = {
        "blocked_left": "obstacle immediately left of this cell",
        "blocked_right": "obstacle immediately right of this cell",
        "blocked_above": "obstacle immediately above this cell",
        "blocked_below": "obstacle immediately below this cell",
        "water": "navigable open water (IceShip)",
        "platform": "platform / room floor (Ladders)",
        "ladder": "vertical corridor (Ladders)",
        "bridge": "horizontal corridor (Ladders)",
        "door": "a doorway cell",
        "hallway": "between-shell hallway (BankRobber)",
        "drop_adjacent": "adjacent to the drop (DontFall)",
        "ground": "plain ground",
        "room_interior": "room interior",
        "on_wormhole": "this cell is a wormhole (SpaceWarp)",
        "clown_near": "the clown is within one cell (ClownChase)",
        "on_treasure": "this cell holds the treasure",
    }

    @property
    def dim(self) -> int:
        """The block's width -- the one place it is defined.

        Size observation spaces, encoders, and buffers from this, never
        from a literal, so appending a slot propagates everywhere at
        once instead of leaving a 16 behind in some forgotten shape.
        """
        return len(fields(self))

    def as_dict(self) -> dict:
        """``{slot name: index}``, in slot order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def names(self) -> tuple:
        """Slot names, indexed by slot number."""
        ordered = sorted(self.as_dict().items(), key=lambda kv: kv[1])
        return tuple(name for name, _ in ordered)

    def describe(self, slot: int) -> str:
        """``"6 ladder -- vertical corridor (Ladders)"``."""
        name = self.names()[slot]
        return f"{slot} {name} -- {self.descriptions[name]}"


#: The canonical slot map. Prefer this to the bare ``TEX_*`` integers
#: when writing an encoder: it names what it indexes.
TEXTURE_SLOTS = TextureSlotMap()

#: Width of the texture block, derived from the slot map.
TEXTURE_DIM = TEXTURE_SLOTS.dim

# Bare aliases, for callers who prefer names to attribute access.
# Derived, so there is one source of truth for every index.
TEX_BLOCK_LEFT = TEXTURE_SLOTS.blocked_left
TEX_BLOCK_RIGHT = TEXTURE_SLOTS.blocked_right
TEX_BLOCK_ABOVE = TEXTURE_SLOTS.blocked_above
TEX_BLOCK_BELOW = TEXTURE_SLOTS.blocked_below
TEX_WATER = TEXTURE_SLOTS.water
TEX_PLATFORM = TEXTURE_SLOTS.platform
TEX_LADDER = TEXTURE_SLOTS.ladder
TEX_BRIDGE = TEXTURE_SLOTS.bridge
TEX_DOOR = TEXTURE_SLOTS.door
TEX_HALLWAY = TEXTURE_SLOTS.hallway
TEX_DROP_ADJ = TEXTURE_SLOTS.drop_adjacent
TEX_DIRT = TEXTURE_SLOTS.ground
TEX_INTERIOR = TEXTURE_SLOTS.room_interior
TEX_WORMHOLE = TEXTURE_SLOTS.on_wormhole
TEX_CLOWN_NEAR = TEXTURE_SLOTS.clown_near
TEX_TREASURE = TEXTURE_SLOTS.on_treasure

#: slot index -> human-readable name (TOPOGYM_DEBUG observation lines)
TEX_SLOT_NAMES = dict(enumerate(TEXTURE_SLOTS.names()))
