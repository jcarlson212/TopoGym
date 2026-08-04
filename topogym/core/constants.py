"""Cell-type and observation codes shared across TopoGym."""

# Layout cell types (what a cell *is*).
EMPTY = 0
WALL = 1
HOLE = 2  # visually distinct impassable cell ("definitely nothing inside")
DOOR = 3  # hidden door: observed as WALL until opened by repeated bumps
GOAL = 4

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
OBS_DOOR_ONEWAY = 8  # a one-way door (valve); passable side discoverable by trying
OBS_TRAPDOOR = 9  # a passage that seals permanently after one use

OBS_MAX = 9

# Universal observation vector (obs_mode="vector"): the agent's integer
# cell coordinates (x, y) followed by a texture block t in [0, 1]^16.
# Slots 0-3 are reserved library-wide for directional blocker adjacency
# (left, right, above, below); slots 4-15 carry per-environment semantic
# features. The block is identically zero outside the Texture variants.
TEXTURE_DIM = 16
TEX_BLOCK_LEFT, TEX_BLOCK_RIGHT, TEX_BLOCK_ABOVE, TEX_BLOCK_BELOW = 0, 1, 2, 3
