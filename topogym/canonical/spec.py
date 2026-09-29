"""The canonical observation/action specification, as a reference
implementation.

This module is the spec. Environments that emit the canonical format
(TopoGym's grid envs with ``obs_mode="canonical"``, and any other
simulator that wants its episodes read by the same tools) take their
key names, units, frames, action vocabulary, text grammar, instruction
templates and category registry from here, and validate against it.

It depends on the standard library only, so it can be imported and
vendored without pulling in an environment.

Principles
----------
1. Egocentric RGB and language are the universal core; everything else
   is declared, optional and typed.
2. Nothing privileged is observed. Privileged signals (world pose, goal,
   distance to goal, region, topology) are *recorded* under
   ``privileged.*`` for auxiliary heads, evaluation and data selection.
3. Every stream is self-describing: names, units and frames live in a
   manifest, not in code.
4. Two action levels: a discrete word vocabulary for token-based
   policies and an egocentric waypoint shared across embodiments.

Frames
------
- **World**: the environment's own map frame, recorded only under
  ``privileged.world_pose``. For grid worlds, cell coordinates with x to
  the right and y *down* (screen convention), one unit per cell.
- **Ego** (``observation.state``): REP-103 body convention (x forward,
  y left, yaw counter-clockwise), expressed relative to the pose at the
  start of the episode. Its origin is per episode unless the manifest
  declares ``episode_frame: per_world`` (every episode of the world
  starts at the same pose).
- **Camera**: OpenCV (x right, y down, z forward) for perspective
  cameras. Grid envs render an orthographic top-down view centred on
  the agent, facing up; the manifest says so rather than inventing
  intrinsics.

Versioning
----------
:data:`CANONICAL_SPEC_VERSION` changes only additively (new optional
keys, new vocabulary). See COMPATIBILITY.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

CANONICAL_SPEC_VERSION = "1.2.0"

# -- observation keys -----------------------------------------------------------

IMAGE_PREFIX = "observation.images."
HEAD = "observation.images.head"
TOPDOWN = "observation.images.topdown"
INSTRUCTION = "observation.language.instruction"
STATE = "observation.state"
TEXT = "observation.text"
STRUCTURED = "observation.structured"
#: Env-specific observations that are allowed but not portable.
NATIVE_PREFIX = "observation.native."
#: Recorded, never observed.
PRIVILEGED_PREFIX = "privileged."

#: Privileged per-step fields (under ``info["privileged"]``).
PRIVILEGED_STEP_FIELDS = ("world_pose", "goal_pose", "d_goal",
                          "goal_visible", "region_id")
#: Privileged per-episode fields (reset ``info["privileged"]``).
PRIVILEGED_EPISODE_FIELDS = ("topology", "goals")

#: Default image side for the VLA preset.
DEFAULT_IMAGE_SIZE = 256

#: Characters the text channels may contain: printable ASCII plus
#: newline. Kept narrow so every tokenizer reads it the same way.
TEXT_CHARSET = frozenset(
    " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\n"
)
MAX_INSTRUCTION_LENGTH = 512
MAX_TEXT_LENGTH = 4096
MAX_STRUCTURED_LENGTH = 16384

#: ``observation.state`` fields for planar agents, in order.
STATE_NAMES_2D = ("ego.x", "ego.y", "ego.yaw")
STATE_UNITS_2D = {"ego.x": "cell", "ego.y": "cell", "ego.yaw": "rad"}


# -- actions --------------------------------------------------------------------

#: The shared word vocabulary, in id order. Ids 0-2 coincide with
#: TopoGym's egocentric action codes, so a word policy and an egocentric
#: policy agree on everything but ``stop``.
WORDS = ("turn_left", "turn_right", "move_forward", "stop")

#: What each word does, stated for any embodiment. Step sizes are
#: embodiment-specific and live in the manifest's ``body``.
WORD_SEMANTICS = {
    "turn_left": "rotate in place counter-clockwise by the body's turn step",
    "turn_right": "rotate in place clockwise by the body's turn step",
    "move_forward": "translate forward by the body's step, if not blocked",
    "stop": "declare the task done; ends the episode",
}

#: Words other embodiments may declare in addition (not used by grids).
OPTIONAL_WORDS = ("look_up", "look_down", "move_up", "move_down", "open",
                  "grab", "release", "operate", "report", "attack")

#: ``action.waypoint``: an ego-frame target, the cross-embodiment level.
WAYPOINT_NAMES = ("dx", "dy", "dz", "dyaw")
WAYPOINT_UNITS = {"dx": "cell", "dy": "cell", "dz": "cell", "dyaw": "rad"}


# -- categories -----------------------------------------------------------------

@dataclass(frozen=True)
class Category:
    """A goal-object category, anchored to WordNet so vocabularies
    from different environments can be joined."""

    label: str  # the noun instructions use
    synset: str  # WordNet synset id

#: Category registry. Append-only: labels and synsets never change.
CATEGORIES = {
    "treasure": Category("treasure", "chest.n.02"),
    "person": Category("person", "person.n.01"),
    "key": Category("key", "key.n.01"),
    "flag": Category("flag", "flag.n.01"),
    "lamp": Category("lamp", "lamp.n.01"),
    "bell": Category("bell", "bell.n.01"),
    "cave": Category("cave", "cave.n.01"),
    "body_of_water": Category("body of water", "body_of_water.n.01"),
}


# -- splits ---------------------------------------------------------------------

SPLITS = ("train", "val", "test", "tune")


def holdout_tag(kind: str, value) -> str:
    """``holdout:family=Maze``: holdouts are tags on a split, not splits."""
    return f"holdout:{kind}={value}"


# -- feature schema -------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSpec:
    """One observed or recorded stream."""

    key: str
    dtype: str  # numpy dtype name, or "string"
    shape: tuple
    names: tuple | None = None
    units: dict | None = None
    frame: str | None = None
    description: str = ""
    privileged: bool = False
    info: dict = field(default_factory=dict)
    #: How a dataset stores it: None (a column), "png"/"jpeg"/"video"
    #: (images), or one of ARRAY_STORAGES (2-D arrays as 16-bit PNG).
    storage: str | None = None

    @property
    def variable_length(self) -> bool:
        """True when some axis of the shape is ``None``."""
        return any(s is None for s in self.shape)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["shape"] = list(self.shape)
        out["names"] = list(self.names) if self.names is not None else None
        return out


def validate_observation(obs: dict, features: dict) -> list:
    """Problems with an observation against declared features (empty
    when valid). Checks presence, dtype, shape and that nothing
    privileged leaked in; values are the environment's business."""
    problems = []
    for key in obs:
        if key.startswith(PRIVILEGED_PREFIX) or key.startswith("privileged"):
            problems.append(f"{key}: privileged key in observation")
        elif key not in features:
            problems.append(f"{key}: undeclared key")
    for key, feat in features.items():
        if feat.privileged:
            continue
        if key not in obs:
            problems.append(f"{key}: declared but missing")
            continue
        value = obs[key]
        if feat.dtype == "string":
            if not isinstance(value, str):
                problems.append(f"{key}: expected str")
            elif not set(value) <= TEXT_CHARSET:
                problems.append(f"{key}: characters outside TEXT_CHARSET")
            continue
        dtype = getattr(value, "dtype", None)
        if dtype is None or str(dtype) != feat.dtype:
            problems.append(f"{key}: dtype {dtype} != {feat.dtype}")
        if tuple(getattr(value, "shape", ())) != tuple(feat.shape):
            problems.append(f"{key}: shape {getattr(value, 'shape', None)}"
                            f" != {tuple(feat.shape)}")
    return problems


# -- text grammar ---------------------------------------------------------------

_NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten")


def number_word(n) -> str:
    """Small whole numbers as words, others as digits (``2.5``)."""
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if isinstance(n, int):
        return _NUMBERS[n] if 0 <= n < len(_NUMBERS) else str(n)
    return f"{n:g}"


#: Distance units the grammar can speak, singular and plural.
DISTANCE_UNITS = {"cells": ("cell", "cells"), "metres": ("metre", "metres")}


def distance_phrase(n, units: str = "cells") -> str:
    """``"two cells"``, ``"one metre"``, ``"2.5 metres"``."""
    one, many = DISTANCE_UNITS[units]
    return f"{number_word(n)} " + (one if n == 1 else many)


def cells_phrase(n: int) -> str:
    return distance_phrase(n, "cells")


def offset_phrase(forward, right, units: str = "cells") -> str:
    """Where something is relative to the agent, in words:
    ``"two cells ahead and one to the left"``."""
    parts = []
    if forward > 0:
        parts.append(f"{distance_phrase(forward, units)} ahead")
    elif forward < 0:
        parts.append(f"{distance_phrase(-forward, units)} behind")
    if right:
        side = "right" if right > 0 else "left"
        n = abs(right)
        if parts:
            parts.append(f"{number_word(n)} to the {side}")
        else:
            parts.append(f"{distance_phrase(n, units)} to the {side}")
    return " and ".join(parts) if parts else "here"


#: Terrain words for the text channel, by terrain name.
TERRAIN_NOUNS = {
    "floor": "open floor",
    "wall": "a wall",
    "pit": "a pit",
    "doorway": "a doorway",
    "edge": "the edge of the world",
    "unseen": "something hidden",
    "drop": "a drop",
    "wormhole": "a wormhole",
    "water": "water",
    "obstacle": "an obstacle",
}

#: Terrains worth listing when visible beyond the agent's neighbours.
NOTABLE_TERRAINS = ("doorway", "pit", "drop", "wormhole", "water",
                    "obstacle")

#: Semantic tags -> how the text channel says the agent is on/near one.
SEMANTIC_ON = {
    "water": "on water",
    "platform": "on a platform",
    "ladder": "on a ladder",
    "bridge": "on a bridge",
    "door": "in a doorway",
    "hallway": "in a hallway",
    "drop_adjacent": "next to a drop",
    "ground": "on plain ground",
    "room_interior": "inside a room",
    "on_wormhole": "on a wormhole",
    "clown_near": "near a clown",
    "on_treasure": "at the treasure",
    "covered": "under cover",
    "underground": "underground",
}

#: Semantic tags -> the noun used when one is seen elsewhere.
SEMANTIC_NOUNS = {
    "water": "water",
    "platform": "a platform",
    "ladder": "a ladder",
    "bridge": "a bridge",
    "door": "a door",
    "hallway": "a hallway",
    "drop_adjacent": "the edge of a drop",
    "room_interior": "a room",
}

#: How many distant items the text lists before summarising the rest.
MAX_LISTED = 6


@dataclass(frozen=True)
class SeenCell:
    """One visible cell in the agent's frame (the grammar's input).

    ``forward``/``right`` are cell offsets from the agent (behind and
    left are negative)."""

    forward: int
    right: int
    terrain: str  # a key of TERRAIN_NOUNS
    semantics: tuple = ()  # keys of SEMANTIC_ON / SEMANTIC_NOUNS
    objects: tuple = ()  # category labels (see CATEGORIES)
    #: What blocks the cell, as a category key or label ("sofa"), for
    #: terrain "obstacle": the text says "a sofa" rather than "a wall".
    blocker: str | None = None


def _terrain_noun(c: SeenCell) -> str:
    if c.blocker:
        cat = CATEGORIES.get(c.blocker)
        label = cat.label if cat is not None else c.blocker
        article = "an" if label[:1].lower() in "aeiou" else "a"
        return f"{article} {label}"
    return TERRAIN_NOUNS[c.terrain]


def _join(items: list) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _order(c: SeenCell) -> tuple:
    # Nearest first; ties ahead before behind, left before right: a
    # fixed total order, so the rendering is deterministic.
    return (abs(c.forward) + abs(c.right), -c.forward, c.right)


def render_text(cells: list, here: tuple = (), *,
                units: str = "cells") -> str:
    """The ``observation.text`` rendering of an egocentric view.

    ``cells`` are the visible cells (the agent's own cell may be
    included, at offset (0, 0)); ``here`` the semantic tags of the
    agent's cell. ``units`` is what an offset of 1 means ("cells", or
    "metres" for continuous views, whose offsets may be fractional; the
    open-run sentence is spoken only for cells). Output is a pure
    function of the input.
    """
    by_offset = {(c.forward, c.right): c for c in cells}
    sentences = []

    around = []
    for label, off in (("ahead", (1, 0)), ("to the left", (0, -1)),
                       ("to the right", (0, 1)), ("behind", (-1, 0))):
        c = by_offset.get(off)
        if c is None:
            around.append(f"{TERRAIN_NOUNS['unseen']} {label}")
        elif c.terrain != "floor":
            around.append(f"{_terrain_noun(c)} {label}")
    sentences.append(
        ("There is " + _join(around) + ".") if around
        else "Open floor all around.")

    run = 0
    while units == "cells" and by_offset.get((run + 1, 0)) is not None \
            and by_offset[(run + 1, 0)].terrain in ("floor", "doorway"):
        run += 1
    if run:
        sentences.append(f"The way ahead is open for {cells_phrase(run)}.")

    on = [SEMANTIC_ON[s] for s in here if s in SEMANTIC_ON]
    if on:
        sentences.append("You are " + _join(on) + ".")

    listed = []
    for c in sorted(cells, key=_order):
        if (c.forward, c.right) == (0, 0):
            for obj in c.objects:
                listed.append(f"the {obj} is here")
            continue
        where = offset_phrase(c.forward, c.right, units)
        for obj in c.objects:
            listed.append(f"the {obj} {where}")
        if c.terrain in NOTABLE_TERRAINS and \
                abs(c.forward) + abs(c.right) > 1:
            listed.append(f"{_terrain_noun(c)} {where}")
    seen_sem = set()
    for c in sorted(cells, key=_order):
        if (c.forward, c.right) == (0, 0):
            continue
        for s in c.semantics:
            if s in SEMANTIC_NOUNS and s not in seen_sem:
                seen_sem.add(s)
                listed.append(f"{SEMANTIC_NOUNS[s]} "
                              f"{offset_phrase(c.forward, c.right, units)}")
    if listed:
        extra = len(listed) - MAX_LISTED
        shown = listed[:MAX_LISTED]
        text = "You see " + _join(shown)
        if extra > 0:
            text += f", and {number_word(extra)} more"
        sentences.append(text + ".")
    return " ".join(sentences)


def render_structured(cells: list, here: tuple = ()) -> str:
    """``observation.structured``: the same content as the text, as
    compact JSON. Plain floor cells without annotations are omitted."""
    rows = [
        {"f": c.forward, "r": c.right, "terrain": c.terrain,
         **({"semantics": list(c.semantics)} if c.semantics else {}),
         **({"objects": list(c.objects)} if c.objects else {}),
         **({"blocker": c.blocker} if c.blocker else {})}
        for c in sorted(cells, key=_order)
        if c.terrain != "floor" or c.semantics or c.objects
    ]
    return json.dumps({"here": list(here), "cells": rows},
                      separators=(",", ":"), sort_keys=True)


# -- instructions ---------------------------------------------------------------

#: Instruction templates by task. Index 0 is the canonical phrasing;
#: the rest are paraphrases, tagged ``paraphrase:<k>``. Append-only.
TEMPLATES = {
    "goto": (
        "Go to the {target}.",
        "Find the {target}.",
        "Navigate to the {target}.",
        "Your goal is the {target}.",
        "Make your way to the {target}.",
    ),
    "goto_place": (
        "Go to {place}.",
        "Find {place}.",
        "Navigate to {place}.",
        "Your goal is {place}.",
        "Make your way to {place}.",
    ),
    "explore": (
        "Explore the world.",
        "Look around and explore.",
        "Explore as much as you can.",
        "Search the whole area.",
        "Wander and explore.",
    ),
}


def phrasing_tag(k: int) -> str:
    return "canonical" if k == 0 else f"paraphrase:{k}"


def parse_phrasing(tag) -> int:
    """``"canonical"`` -> 0, ``"paraphrase:2"`` -> 2 (ints pass)."""
    if isinstance(tag, int):
        return tag
    if tag == "canonical":
        return 0
    if isinstance(tag, str) and tag.startswith("paraphrase:"):
        return int(tag.split(":", 1)[1])
    raise ValueError(f"unknown phrasing {tag!r}; expected 'canonical' "
                     "or 'paraphrase:<k>'")


def instruction(task: str, k: int = 0, **slots) -> str:
    """Fill template ``k`` of ``task``."""
    options = TEMPLATES[task]
    if not 0 <= k < len(options):
        raise ValueError(f"{task!r} has phrasings 0..{len(options) - 1}")
    return options[k].format(**slots)


def room_phrase(n_doors: int | None) -> str:
    """``"the room with two doors"`` (or a room, when not unique)."""
    if n_doors is None:
        return "one of the rooms"
    word = number_word(n_doors)
    return f"the room with {word} door" + ("" if n_doors == 1 else "s")

# == 1.2.0 additions ==============================================================

# -- more keys --------------------------------------------------------------------

DEPTH_PREFIX = "observation.depth."
SEGMENTATION_PREFIX = "observation.segmentation."
#: Multi-party messages: list of {t, speaker, to, text, volume?}.
CHAT = "observation.language.chat"
#: An observed (estimated) pose in a map frame -- odometry or SLAM,
#: drifted if the embodiment's would be. Unlike privileged.world_pose,
#: a policy may see it.
MAP_POSE = "observation.map_pose"
#: A long-range target in the episode (or map) frame, executed by the
#: env's planner; the manifest says whether that planner uses a
#: privileged map (``goto.uses_privileged_map``).
GOTO = "action.goto"


def depth_key(cam: str) -> str:
    return DEPTH_PREFIX + cam


def segmentation_key(cam: str) -> str:
    return SEGMENTATION_PREFIX + cam


#: Every word id, core then optional, in id order. Append-only.
def vocabulary() -> tuple:
    return WORDS + OPTIONAL_WORDS


def word_id(word: str) -> int:
    return vocabulary().index(word)


# -- 3D bodies and continuous worlds -----------------------------------------------
#
# Continuous worlds use the REP-103 world frame (right-handed, z up) in
# metres and radians; grid worlds keep their cell frame (x right, y
# down, cells). The names below are what ``observation.state`` and the
# privileged fields declare, so data from both can be mixed by name.

STATE_NAMES_3D = ("ego.x", "ego.y", "ego.z", "ego.yaw", "ego.pitch")
STATE_UNITS_3D = {"ego.x": "m", "ego.y": "m", "ego.z": "m",
                  "ego.yaw": "rad", "ego.pitch": "rad"}

#: A full 6-DoF pose: position, then a unit quaternion (scalar last),
#: of the child frame in the parent frame. For cameras and bodies that
#: pitch or roll (see :mod:`topogym.canonical.transforms`).
POSE7_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw")
POSE7_UNITS = {"x": "m", "y": "m", "z": "m", "qx": "1", "qy": "1",
               "qz": "1", "qw": "1"}

#: privileged.world_pose / goal_pose. Grids: cells in the grid frame.
#: Continuous worlds: metres in the REP-103 world frame, with height and
#: heading (a navigation pose; use pose7 where roll and pitch matter).
WORLD_POSE_NAMES_2D = ("x", "y", "yaw")
GOAL_POSE_NAMES_2D = ("x", "y")
WORLD_POSE_NAMES_3D = ("x", "y", "z", "yaw")
GOAL_POSE_NAMES_3D = ("x", "y", "z", "yaw")

#: privileged.d_goal: grids report an int (cells, graph distance),
#: continuous worlds a float (metres, path distance). Unreachable is
#: null at runtime; in dataset columns NaN (float) or the declared
#: sentinel (int, -1).
D_GOAL_UNITS = {"grid": "cell", "continuous": "m"}

#: privileged.region_id is an index into topology.regions. Grids label
#: every free cell (0 = open space); continuous worlds use -1 for "in
#: no region".
REGION_NONE = -1


# -- dataset conventions --------------------------------------------------------------
#
# Rows are decisions (one per action, with next.* fields for what
# followed); per-tick data rides in side streams. The clock is regular:
# timestamp = frame_index / fps. Missing values: NaN in float columns,
# a declared integer sentinel (FeatureSpec.info["missing"]) in integer
# columns, null in JSON metadata; never None in a column.

#: Depth images: z-depth in metres stored as 16-bit PNG codes of
#: DEPTH_UNIT_M, 0 = no valid depth, capped at DEPTH_MAX_CODE (65.534 m)
#: because LeRobot decodes 16-bit PNGs as int16. See
#: topogym.canonical.codecs.encode_depth / decode_depth.
DEPTH_UNIT_M = 0.002
DEPTH_MAX_CODE = 32767

#: Segmentation images: instance ids (0 = none) as 16-bit PNG, at most
#: SEGMENTATION_MAX_ID; ids mean what the episode's table says
#: ({id: {"category", "label"}} in the episode record).
SEGMENTATION_MAX_ID = 32767

#: Other integer 2-D arrays stored as 16-bit PNG: codes 0..ARRAY16_MAX.
ARRAY16_MAX = 32767

#: How 2-D array features can be stored (as images, never flattened).
ARRAY_STORAGES = ("depth_png16", "segmentation_png16", "array_png16")
#: How image features can be stored.
IMAGE_STORAGES = ("png", "jpeg", "video")


def array_storage_info(storage: str) -> dict:
    """The ``info`` a 2-D array storage declares in meta/info.json."""
    if storage == "depth_png16":
        return {"is_depth_map": True, "depth.encoding": "png16",
                "depth.unit_m": DEPTH_UNIT_M, "depth.max_code":
                DEPTH_MAX_CODE, "depth.invalid_code": 0,
                "depth.kind": "z_depth"}
    if storage == "segmentation_png16":
        return {"is_segmentation": True, "segmentation.encoding": "png16",
                "segmentation.max_id": SEGMENTATION_MAX_ID,
                "segmentation.none_id": 0}
    if storage == "array_png16":
        return {"array.encoding": "png16", "array.max_code": ARRAY16_MAX}
    raise ValueError(f"unknown array storage {storage!r}")


# -- the privileged extension namespace ---------------------------------------------

#: Producer-specific privileged fields live under one namespace:
#: ``privileged.ext.<producer>.<field>``, declared in the manifest's
#: privileged section as ``ext.<producer>``. This package does not
#: interpret them; assemble preserves them.
EXT_PREFIX = "ext."
PRIVILEGED_EXT_PREFIX = PRIVILEGED_PREFIX + EXT_PREFIX


def privileged_ext_key(producer: str, field: str) -> str:
    _check_producer(producer)
    return f"{PRIVILEGED_EXT_PREFIX}{producer}.{field}"


def _check_producer(producer: str) -> None:
    if not producer or not all(c.isalnum() or c in "_-" for c in producer):
        raise ValueError(f"producer names are [A-Za-z0-9_-]+, got "
                         f"{producer!r}")


#: Top-level key families a dataset feature may use.
_ROW_KEYS = ("is_first", "is_last", "is_terminal", "action")
_ROW_PREFIXES = ("observation.", "action.", "next.")

#: Privileged per-step fields beyond the grid's, in continuous worlds.
PRIVILEGED_STEP_FIELDS_EXTRA = ("camera_pose", "object_poses", "phase",
                                "progress")


def check_key(key: str) -> str | None:
    """Why ``key`` is not a canonical feature key, or None."""
    if key.startswith(PRIVILEGED_PREFIX):
        rest = key[len(PRIVILEGED_PREFIX):]
        if rest.startswith(EXT_PREFIX):
            parts = rest[len(EXT_PREFIX):].split(".", 1)
            if len(parts) != 2 or not parts[0] or not parts[1]:
                return "use privileged.ext.<producer>.<field>"
            return None
        field = rest.split(".", 1)[0]
        if field in PRIVILEGED_STEP_FIELDS + PRIVILEGED_EPISODE_FIELDS + \
                PRIVILEGED_STEP_FIELDS_EXTRA:
            return None
        return (f"{key!r} is not a spec privileged field; producer-specific "
                "fields go under privileged.ext.<producer>.<field>")
    if key in _ROW_KEYS or key.startswith(_ROW_PREFIXES):
        return None
    return (f"{key!r} is outside the canonical namespaces (observation.*, "
            "action[.*], next.*, is_*, privileged.*); env-specific data "
            "goes under observation.native.*")


def validate_features(features) -> list:
    """Problems with a list of :class:`FeatureSpec` (empty when valid):
    namespaces, duplicate keys, and storage/dtype/shape agreement."""
    problems, seen = [], set()
    for f in features:
        if f.key in seen:
            problems.append(f"{f.key}: declared twice")
        seen.add(f.key)
        why = check_key(f.key)
        if why:
            problems.append(why)
        shape = tuple(f.shape)
        if f.storage in IMAGE_STORAGES:
            if f.dtype != "uint8" or not (
                    len(shape) == 2 or (len(shape) == 3
                                        and shape[2] in (1, 3))):
                problems.append(f"{f.key}: {f.storage} images are uint8 "
                                "(H, W), (H, W, 1) or (H, W, 3)")
            if f.storage == "video" and (len(shape) != 3 or shape[2] != 3):
                problems.append(f"{f.key}: video is (H, W, 3) uint8")
        elif f.storage in ARRAY_STORAGES:
            if len(shape) not in (2, 3) or (len(shape) == 3
                                            and shape[2] != 1):
                problems.append(f"{f.key}: {f.storage} is (H, W) or "
                                "(H, W, 1)")
            if f.storage == "depth_png16" and not f.dtype.startswith("float"):
                problems.append(f"{f.key}: depth is float metres")
            if f.storage != "depth_png16" and "int" not in f.dtype:
                problems.append(f"{f.key}: {f.storage} holds integers")
        elif f.storage is not None:
            problems.append(f"{f.key}: unknown storage {f.storage!r}")
        elif None in shape and f.dtype == "string":
            problems.append(f"{f.key}: strings are scalars")
    return problems


def _flat(value):
    """Nested sequences to a flat list (duck-typed; no numpy here)."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            out.extend(_flat(v))
        return out
    return [value]


def check_value(feature: FeatureSpec, value) -> str | None:
    """Why ``value`` cannot be written as ``feature``, or None. Enforces
    the missing-value convention: no None in a numeric feature (NaN for
    floats, the declared sentinel for integers)."""
    if feature.dtype == "string":
        return None if isinstance(value, str) else "expected a str"
    if value is None:
        return ("None in a numeric feature; use NaN (float) or the "
                "declared missing sentinel (int)")
    if feature.storage in IMAGE_STORAGES + ARRAY_STORAGES:
        shape = tuple(getattr(value, "shape", ()))
        want = tuple(feature.shape)
        if shape != want and shape[:2] != want[:2]:
            return f"shape {shape} != {want}"
        return None
    flat = _flat(value)
    if any(v is None for v in flat):
        return ("None inside a numeric feature; use NaN (float) or the "
                "declared missing sentinel (int)")
    if not feature.variable_length:
        n = 1
        for s in feature.shape:
            n *= s
        if len(flat) != n:
            return f"{len(flat)} values for shape {tuple(feature.shape)}"
    return None


# -- registries --------------------------------------------------------------------
#
# Other producers extend the category and template registries at
# runtime rather than forking them. Their entries are namespaced
# ("<producer>:<key>"), and append-only per producer: registering a key
# again with a different category, or templates that do not extend the
# existing list, is refused. Unprefixed keys belong to this package.


def register_category(key: str, category: Category, *,
                      producer: str) -> str:
    """Add a category under ``<producer>:<key>``; returns the full key."""
    _check_producer(producer)
    if not isinstance(category, Category):
        raise TypeError("category must be a Category")
    full = f"{producer}:{key}"
    existing = CATEGORIES.get(full)
    if existing is not None and existing != category:
        raise ValueError(f"{full} is already registered as {existing}; "
                         "categories are append-only")
    CATEGORIES[full] = category
    return full


def register_templates(task: str, templates, *, producer: str) -> str:
    """Add instruction templates for ``<producer>:<task>`` (index 0 is
    the canonical phrasing). Registering again may only append."""
    _check_producer(producer)
    templates = tuple(templates)
    if not templates or not all(isinstance(t, str) for t in templates):
        raise ValueError("templates are a non-empty sequence of strings")
    full = f"{producer}:{task}"
    existing = TEMPLATES.get(full)
    if existing is not None and templates[:len(existing)] != existing:
        raise ValueError(f"{full}: templates are append-only; the existing "
                         f"{len(existing)} must stay first and unchanged")
    TEMPLATES[full] = templates
    return full
