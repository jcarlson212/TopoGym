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

OBS_MAX = 7
