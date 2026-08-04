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
from topogym.generation.rooms import SHAPE_CODES

#: Chamber/decoy outer side used across the v1 registry: fixed so world
#: size (dilution) is never confounded with room size.
_SIDE = 8


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
        add(f"Dilution-{size}", _open_cfg(size))
    # Chambers2: two chambers, fixed geometry; the world-scaling family.
    for size in (50, 100, 200, 400):
        add(f"Chambers2-{size}", _open_cfg(size, n_chambers=2))
    # ChamberCount: k separated chambers at fixed world size.
    for k in (1, 2, 4, 8):
        add(f"ChamberCount{k}-200", _open_cfg(200, n_chambers=k))
    # Decoys: one chamber among k sealed decoys.
    for k in (0, 1, 2, 4, 8):
        add(f"Decoys{k}-50", _open_cfg(50, n_decoys=k))
    # Shape: area-matched chamber shapes.
    for shape in ("square", "circle", "triangle", "star"):
        add(f"Shape{SHAPE_CODES[shape]}-50",
            _open_cfg(50, chamber_shape=shape))
    # Nested: sequentially nested shells.
    for depth in (1, 2, 3):
        add(f"Nested{depth}-50", _open_cfg(
            50, style="nested", nested_depth=depth,
        ))
    # GiveUp: the door hides behind a dead-end corridor.
    for length in (1, 2, 4):
        add(f"GiveUp{length}-50", _open_cfg(
            50, door_corridor_len=length,
        ))
    # Bottleneck: a tree of rooms joined by width-1 corridors.
    for length in (3, 6):
        add(f"Bottleneck{length}-100", _open_cfg(
            100, style="corridor", corridor_len=length, rooms=6,
            n_chambers=0,
        ))
    # Maze: seeded perfect maze (braid opens loops).
    for size in (50, 100):
        add(f"Maze-{size}", _open_cfg(
            size, style="maze", n_chambers=0,
        ))
    return entries


#: name -> frozen generator configuration (the registry itself).
REGISTRY: dict = _build_registry()

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
    return (
        f"TG-GridWorld2D-S{size}-C{cfg.n_chambers}-D{cfg.n_decoys}"
        f"-cs{cs}-ds{ds}-sep{cfg.min_sep}-shp{shp}-{mode}"
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
            kwargs={"config": cfg},
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
