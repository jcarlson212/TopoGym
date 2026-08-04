"""Cell-type and observation codes shared across TopoGym."""

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
