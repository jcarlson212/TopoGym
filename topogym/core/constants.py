"""Cell-type, action, and observation codes shared across TopoGym."""

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

# Universal observation vector (obs_mode="vector"): the agent's integer
# cell coordinates (x, y) followed by a texture block t in [0, 1]^16.
# Slots 0-3 are reserved library-wide for directional blocker adjacency
# (left, right, above, below); slots 4-15 carry per-environment semantic
# features. The block is identically zero outside the Texture variants.
TEXTURE_DIM = 16
TEX_BLOCK_LEFT, TEX_BLOCK_RIGHT, TEX_BLOCK_ABOVE, TEX_BLOCK_BELOW = 0, 1, 2, 3

# Semantic texture slots (4-15), assigned library-wide so agents transfer
# between Texture scenarios. Each scenario documents which slots it uses.
TEX_WATER = 4  # navigable open water (IceShip)
TEX_PLATFORM = 5  # platform / room floor (Ladders)
TEX_LADDER = 6  # vertical corridor (Ladders)
TEX_BRIDGE = 7  # horizontal corridor (Ladders)
TEX_DOOR = 8  # a doorway cell
TEX_HALLWAY = 9  # between-shell hallway (BankRobber)
TEX_DROP_ADJ = 10  # adjacent to the drop (DontFall)
TEX_DIRT = 11  # plain ground
TEX_INTERIOR = 12  # room interior
TEX_WORMHOLE = 13  # standing on a wormhole (SpaceWarp)
TEX_CLOWN_NEAR = 14  # the clown is within one cell (ClownChase)
TEX_TREASURE = 15  # standing on the treasure cell

#: slot index -> human-readable name (TOPOGYM_DEBUG observation lines)
TEX_SLOT_NAMES = {
    TEX_BLOCK_LEFT: "blocked_left", TEX_BLOCK_RIGHT: "blocked_right",
    TEX_BLOCK_ABOVE: "blocked_above", TEX_BLOCK_BELOW: "blocked_below",
    TEX_WATER: "water", TEX_PLATFORM: "platform", TEX_LADDER: "ladder",
    TEX_BRIDGE: "bridge", TEX_DOOR: "door", TEX_HALLWAY: "hallway",
    TEX_DROP_ADJ: "drop_adjacent", TEX_DIRT: "ground",
    TEX_INTERIOR: "room_interior", TEX_WORMHOLE: "on_wormhole",
    TEX_CLOWN_NEAR: "clown_near", TEX_TREASURE: "on_treasure",
}
