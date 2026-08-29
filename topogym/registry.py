"""The TopoGym-v1 registry: named, pinned GridWorld2D environments.

Following the convention of existing gym libraries, the canonical
interface is a registry of named, pre-defined environments::

    import gymnasium as gym
    import topogym  # registers the ids

    env = gym.make("TopoGym/Dilution-50-v0", seed=3)

Each registry entry is a frozen configuration of the underlying
generator; the seed drives layout variation (placement, door positions,
shape assignment) within that configuration; and a small set of
documented parts stays modifiable as ``gym.make`` kwargs (``p_slip``,
``reward_mode``, ``complex``, and per-family knobs such as
``decoy_side``, ``rooms``, or ``braid``). Registry ids are stable across
releases; new families extend the registry rather than altering existing
entries.

A configuration serializes to a canonical string
(``TG-GridWorld2D-S{size}-...``) — the run-log key and the manifest row;
registry ids are aliases for canonical strings.
"""

from __future__ import annotations

import dataclasses

from topogym.generation.config import TopoGenConfig2D
from topogym.generation.modes import spiral_side
from topogym.generation.rooms import SHAPE_CODES

#: EpicChase: actions between consecutive chambers along the spiral, and
#: the basis of the family's episode budget.
_EPIC_ARC = 120

#: EnlargedChamberCount's episode budget. Doors are generated more than
#: this far apart, so no episode reaches two of them.
_ECC_HORIZON = 60

#: k -> world side. Measured rather than chosen: the smallest side at
#: which six consecutive seeds all place k chambers with doors above
#: the horizon, so a family member exists for every seed a split asks
#: for. k stops at 10 because 12 needs a world large enough that a
#: million-step study on it costs more than the point it makes.
_ECC_SIZES = {2: 60, 3: 60, 4: 80, 5: 100, 6: 120, 8: 150, 10: 180}

#: OpenFieldChamberCount's episode budget. Fixed at EnlargedChamber-
#: Count's so the two families are read against each other: same
#: horizon, same door guarantee, and the only thing that changes is
#: whether the world's edge is somewhere the agent can get to.
_OFCC_HORIZON = 60

#: k -> ring radius. The chambers sit evenly by angle on this ring, so
#: the arc between neighbours is 2*R*sin(pi/k) and the radius has to
#: grow with k to keep consecutive doors more than a horizon apart --
#: the same guarantee EnlargedChamberCount makes, stated on a circle.
#: Rejection still enforces it; these values only make the enforcement
#: likely to succeed rather than spin.
_OFCC_RINGS = {2: 40, 3: 45, 4: 50, 5: 55, 6: 65, 8: 85, 10: 105}

#: Free cells between the ring and the wall, on every side. This is the
#: whole point of the family, and it is a measured quantity rather than
#: a taste: it must exceed the furthest any method gets from the start
#: inside the step budget, so that the boundary is never reached and
#: cannot be followed from one chamber to the next. Measured in
#: EnlargedChamberCount, every arm is pinned against the wall by 500k
#: of a 1M budget -- which is what this family exists to prevent.
_OFCC_MARGIN = 200

#: Chamber counts EpicChase is registered at.
#:
#: A sweep rather than two points, because the family exists to
#: exhibit a scaling law: chambers sit an episode apart, so entering
#: all k of them is a conjunction of k independent-ish events, and the
#: theory separates methods by how the probability of that conjunction
#: falls with k. Two values cannot show a curve. Adding names here
#: never touches a published benchmark -- ``benchmarks.json`` is the
#: sole authority on membership and lists EpicChase as standalone.
_EPIC_CHAMBERS = (1, 2, 3, 4, 6, 8, 12)

#: EpicChase corridor width. Wide enough to move around in -- a
#: width-1 spiral is a queue, not a hallway -- and odd, so the arms
#: stay centred on the path the spacing is measured along.
_EPIC_WIDTH = 3

#: Chamber/decoy outer side used across the v1 registry: fixed so world
#: size (dilution) is never confounded with room size.
_SIDE = 8

#: EpicChase chambers are smaller: they hang inside the wall band
#: between spiral arms, and every extra cell of chamber widens the
#: whole spiral.
_SIDE_EPIC = 5


def _open_cfg(size: int, **kw) -> TopoGenConfig2D:
    """The registry's open-mode base: no holes, open width-1 doors."""
    base = TopoGenConfig2D(
        base="square", size=size, style="rooms",
        n_holes=0, n_chambers=1, n_decoys=0,
        chamber_side=_SIDE, decoy_side=_SIDE,
        door_kind="open", min_sep=2,
        # The spec's sparse-target convention: the goal sits inside a
        # designated chamber, so steps-to-first-reward coincides with
        # steps-to-first-entry.
        goal_in_chamber=True,
    )
    cfg = dataclasses.replace(base, **kw) if kw else base
    if cfg.n_chambers == 0 and cfg.goal_in_chamber:
        cfg = dataclasses.replace(cfg, goal_in_chamber=False)
    return cfg


def _build_registry() -> dict:
    entries: dict = {}

    def add(name: str, cfg: TopoGenConfig2D) -> None:
        entries[name] = cfg

    # Dilution: one chamber, no decoys; difficulty scales with the world.
    for size in (50, 200):
        add(f"Dilution-{size}", _open_cfg(
            size, chamber_placement="center",
            start_placement="bottom_left"))
    # Chambers2: two chambers, fixed geometry; the world-scaling family.
    for size in (50, 100, 200, 400):
        add(f"Chambers2-{size}", _open_cfg(
            size, n_chambers=2, chamber_placement="perimeter",
            start_placement="bottom_left"))
    # ChamberCount: k separated chambers at fixed world size.
    for k in (1, 2, 4, 8):
        add(f"ChamberCount{k}-200", _open_cfg(
            200, n_chambers=k, chamber_placement="perimeter",
            start_placement="center"))
    # EnlargedChamberCount: ChamberCount with the separation made a
    # guarantee instead of an accident. Every pair of doors is more than
    # _ECC_HORIZON apart over the free-cell graph, so no episode reaches
    # two of them and "entered j chambers" costs at least j chained
    # episodes -- the premise a chamber-count claim needs, and the one
    # ChamberCount cannot make because its doors land wherever the
    # perimeter policy happens to put them.
    #
    # The world grows with k because the guarantee demands it: a fixed
    # size would shrink the separation instead, which is the confound
    # this family exists to remove. Growth makes the task harder for
    # every method equally, so it favours none of them.
    for k, side in _ECC_SIZES.items():
        add(f"EnlargedChamberCount{k}-{side}", _open_cfg(
            side, n_chambers=k, chamber_placement="perimeter",
            placement_jitter=4, start_placement="center",
            min_door_distance=_ECC_HORIZON + 1))
    # OpenFieldChamberCount: EnlargedChamberCount with the wall moved
    # out of reach. The k chambers ride a ring about the start rather
    # than the world's perimeter, and _OFCC_MARGIN cells of open floor
    # sit between that ring and the boundary.
    #
    # The point is what it removes. When chambers sit on the perimeter,
    # a method that finds the wall can follow it from one chamber to the
    # next, and the score partly measures how efficiently it sweeps a
    # boundary. Here the boundary is never reached, so the only
    # exploitable structure is the ring the agent has itself encircled
    # -- which is the thing a topological signal claims to see.
    for k, radius in _OFCC_RINGS.items():
        side = 2 * (radius + _OFCC_MARGIN)
        add(f"OpenFieldChamberCount{k}-{side}", _open_cfg(
            side, n_chambers=k, chamber_placement="around",
            ring_radius=radius, placement_jitter=4,
            start_placement="center",
            min_door_distance=_OFCC_HORIZON + 1))
    # Decoys: one chamber among k sealed decoys.
    for k in (0, 1, 2, 4, 8):
        add(f"Decoys{k}-50", _open_cfg(
            50, n_decoys=k, chamber_placement="center",
            decoy_placement="around",
            start_placement="bottom_left"))
    # Shape: area-matched chamber shapes.
    for shape in ("square", "circle", "triangle", "star"):
        add(f"Shape{SHAPE_CODES[shape]}-50",
            _open_cfg(50, chamber_shape=shape,
                      chamber_placement="center",
                      start_placement="bottom_left"))
    # Nested: sequentially nested shells.
    for depth in (1, 2, 3):
        add(f"Nested{depth}-50", _open_cfg(
            50, style="nested", nested_depth=depth,
        ))
    # GiveUp: the door hides behind a dead-end corridor.
    for length in (1, 2, 4):
        add(f"GiveUp{length}-50", _open_cfg(
            50, door_corridor_len=length,
            chamber_placement="center", start_placement="bottom_left",
        ))
    # Bottleneck: a tree of rooms joined by width-1 corridors.
    for length in (3, 6):
        add(f"Bottleneck{length}-100", _open_cfg(
            100, style="corridor", corridor_len=length, rooms=6,
            n_chambers=0, chamber_side=24,
        ))
    # Maze: seeded perfect maze (braid opens loops).
    for size in (50, 100):
        add(f"Maze-{size}", _open_cfg(
            size, style="maze", n_chambers=0,
        ))
    # EpicChase: chambers an episode apart along one spiral corridor.
    # Deliberately outside every benchmark roster (see benchmarks.json):
    # it is a stress test for archive-based methods, not a graded task.
    for k in _EPIC_CHAMBERS:
        side = spiral_side(k, _EPIC_ARC, _SIDE_EPIC, _EPIC_WIDTH)
        add(f"EpicChase{k}-{side}", _open_cfg(
            side, style="spiral", n_chambers=k, n_decoys=0,
            spiral_arc=_EPIC_ARC, spiral_width=_EPIC_WIDTH,
            chamber_side=_SIDE_EPIC, start_placement="center",
        ))
    return entries


#: name -> frozen generator configuration (the registry itself).
REGISTRY: dict = _build_registry()

#: name -> extra ``gym.make`` kwargs. The horizon is normally derived
#: from the layout (side length, or slack over the optimal path), but
#: EpicChase inverts that: the budget is the *premise*, chosen so a
#: single episode reaches exactly one chamber, and the world is sized
#: to fit it.
EXTRA_KWARGS: dict = {
    name: {"max_steps": _EPIC_ARC + _EPIC_ARC // 2}
    for name in REGISTRY
    if name.startswith("EpicChase")
}
#: OpenFieldChamberCount takes the same fixed budget, so that a result
#: on it and one on EnlargedChamberCount differ in the boundary and in
#: nothing else.
EXTRA_KWARGS.update({
    name: {"max_steps": _OFCC_HORIZON}
    for name in REGISTRY
    if name.startswith("OpenFieldChamberCount")
})

#: EnlargedChamberCount inverts it the same way, and more strictly: the
#: budget is fixed at _ECC_HORIZON for every k, and generation
#: guarantees the doors sit further apart than that. Deriving the
#: horizon from the layout instead would let it grow with the world and
#: quietly dissolve the premise at large k.
EXTRA_KWARGS.update({
    name: {"max_steps": _ECC_HORIZON}
    for name in REGISTRY
    if name.startswith("EnlargedChamberCount")
})

#: The Top slice of TopoGym-v1: registry name -> topology.
TOP_TOPOLOGIES = {
    "TopPlane": "plane",
    "TopCylinder": "cylinder",
    "TopMobius": "mobius",
    "TopTorus": "torus",
    "TopKlein": "klein",
    "TopRP2": "rp2",
}

#: The Texture slice of TopoGym-v1: registry name -> scenario name.
TEXTURE_SCENARIOS = {
    "IceShip": "ice_ship",
    "Ladders": "ladders",
    "BankRobber": "bank_robber",
    "DontFall": "dont_fall",
    "SpaceWarp": "space_warp",
    "ClownChase": "clown_chase",
    "SearchRescue": "search_rescue",
    "EnvironmentalIceShip": "environmental_ice_ship",
}


def registry_ids() -> list:
    """All registry env ids, ``TopoGym/{Family}-{size}-v0`` plus the
    Texture scenarios ``TopoGym/{Scenario}-v0``."""
    return (
        [f"TopoGym/{name}-v0" for name in REGISTRY]
        + [f"TopoGym/{name}-50-v0" for name in TOP_TOPOLOGIES]
        + [f"TopoGym/{name}-v0" for name in TEXTURE_SCENARIOS]
    )


def _normalize(env_id: str) -> str:
    name = env_id
    if name.startswith("TopoGym/"):
        name = name[len("TopoGym/"):]
    if name.endswith("-v0"):
        name = name[: -len("-v0")]
    if name not in REGISTRY:
        raise KeyError(
            f"unknown registry entry {env_id!r}; see registry_ids()"
        )
    return name


def get_config(env_id: str) -> TopoGenConfig2D:
    """The frozen configuration behind a registry id (or family name)."""
    return REGISTRY[_normalize(env_id)]


def canonical_string(cfg: TopoGenConfig2D, seed: int,
                     p_slip: float = 0.0) -> str:
    """The canonical configuration string: the run-log key.

    ``TG-GridWorld2D-S{size}-C{c}-D{d}-cs{n}-ds{n}-sep{n}-shp{..}-{mode}
    -slip{p}-seed{n}``
    """
    size = cfg.size if isinstance(cfg.size, int) else max(cfg.size)
    shp = SHAPE_CODES.get(cfg.chamber_shape, "Sq")
    mode = "open" if cfg.style == "rooms" else cfg.style
    cs = cfg.chamber_side if cfg.chamber_side is not None else 0
    ds = cfg.decoy_side if cfg.decoy_side is not None else cs
    # Structural fields that define a family but are not implied by the
    # counts above. Omitted at their defaults, so plain configurations
    # keep the short form -- but present whenever they distinguish two
    # environments, because this string is the reproduction key.
    extras = ""
    if cfg.base != "square":
        extras += f"-b{cfg.base}"
    if cfg.n_holes:
        extras += f"-h{cfg.n_holes}"
    if cfg.base == "x_holes":
        extras += f"-bh{cfg.n_base_holes}"
    if cfg.doors_per_chamber > 1:
        extras += f"-dc{cfg.doors_per_chamber}"
    if cfg.door_corridor_len:
        extras += f"-cor{cfg.door_corridor_len}"
    if cfg.style == "nested":
        extras += f"-nd{cfg.nested_depth}"
        if cfg.shell_spacing != 2:
            extras += f"-ss{cfg.shell_spacing}"
    if cfg.style == "corridor":
        extras += f"-cl{cfg.corridor_len}-rm{cfg.rooms}"
    if cfg.style == "spiral":
        extras += f"-arc{cfg.spiral_arc}-sw{cfg.spiral_width}"
    if cfg.style == "maze" and getattr(cfg, "braid", 0):
        extras += f"-br{cfg.braid}"

    placement = ""
    placement += {"center": "-ctr", "perimeter": "-per",
                  "around": "-rng"}.get(cfg.chamber_placement, "")
    # A ring pinned to the world edge and one held at a fixed radius are
    # different specimens even at equal size, so the radius is part of
    # the key whenever it is not the historical fit-the-margin default.
    if cfg.ring_radius and (cfg.chamber_placement == "around"
                            or cfg.decoy_placement == "around"):
        placement += f"-rr{cfg.ring_radius}"
    placement += "-ring" if cfg.decoy_placement == "around" else ""
    placement += {"bottom_left": "-bl", "center": "-sc"}.get(
        cfg.start_placement, "")
    placement += f"-j{cfg.placement_jitter}" if cfg.placement_jitter else ""
    return (
        f"TG-GridWorld2D-S{size}-C{cfg.n_chambers}-D{cfg.n_decoys}"
        f"-cs{cs}-ds{ds}-sep{cfg.min_sep}-shp{shp}{extras}{placement}"
        f"-{mode}"
        f"-slip{p_slip:g}-seed{seed}"
    )


def register_all() -> None:
    """Register every entry with Gymnasium (idempotent)."""
    import gymnasium as gym
    from gymnasium.envs.registration import register

    for name, cfg in REGISTRY.items():
        env_id = f"TopoGym/{name}-v0"
        if env_id in gym.registry:
            continue
        register(
            id=env_id,
            entry_point="topogym.envs:TopoGrid2DEnv",
            kwargs={"config": cfg, **EXTRA_KWARGS.get(name, {})},
        )
    for name, topology in TOP_TOPOLOGIES.items():
        env_id = f"TopoGym/{name}-50-v0"
        if env_id in gym.registry:
            continue
        register(
            id=env_id,
            entry_point="topogym.envs:TopGrid2DEnv",
            kwargs={"topology": topology, "size": 50},
        )
    for name, scenario in TEXTURE_SCENARIOS.items():
        env_id = f"TopoGym/{name}-v0"
        if env_id in gym.registry:
            continue
        register(
            id=env_id,
            entry_point="topogym.envs:TextureGrid2DEnv",
            kwargs={"scenario": scenario},
        )


def manifest(seed: int = 0, ids: list | None = None) -> list:
    """One row per registry entry: id, canonical string, validity, and
    satisfied assumptions — generated (and therefore certified) at the
    given seed. ``ids`` restricts to a subset of registry ids."""
    from topogym.generation.generator import GenerationError, generate_2d

    if ids is not None:
        wanted = {_normalize(i) for i in ids}
        items = [(n, c) for n, c in REGISTRY.items() if n in wanted]
    else:
        items = list(REGISTRY.items())
    rows = []
    for name, cfg in items:
        row = {
            "id": f"TopoGym/{name}-v0",
            "canonical": canonical_string(cfg, seed),
            "config": cfg.to_dict(),
            "seed": seed,
        }
        try:
            metadata = generate_2d(cfg, seed).metadata
            assert metadata is not None  # always set by generate_2d
            row["valid"] = True
            row["betti_z2"] = list(metadata.betti_z2)
            row["assumptions"] = {
                "well_composed": True,  # enforced at generation
                "door_width": 1,
                "min_sep": cfg.min_sep,
                "exterior_4_connected": True,  # enforced at generation
            }
        except GenerationError as exc:
            row["valid"] = False
            row["error"] = str(exc)
        rows.append(row)
    return rows
