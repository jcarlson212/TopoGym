"""The canonical manifest: what an env's canonical streams are.

Keys, shapes, dtypes, units and frames of every observed stream; what
is recorded as privileged; the body; the action vocabulary and its
semantics; the goal categories and instruction templates; the split.
Consumers read this rather than hard-coding any of it.
"""

from __future__ import annotations

import math

from topogym.canonical import spec
from topogym.canonical.spec import FeatureSpec


def _adapter(env):
    """The canonical adapter behind ``env`` (a wrapped or in-env one)."""
    e = env
    while e is not None:
        adapter = getattr(e, "adapter", None)
        if adapter is not None:
            return adapter
        e = getattr(e, "env", None)
    adapter = getattr(env.unwrapped, "_canonical", None)
    if adapter is None:
        raise ValueError(
            "env does not run the canonical layer; build it with "
            'obs_mode="canonical" or wrap it with topogym.canonical.wrap')
    return adapter


def features(adapter) -> dict:
    """``key -> FeatureSpec`` for the observed streams."""
    s = adapter.image_size
    n = 2 * adapter.env.view_radius + 1
    out = {
        spec.HEAD: FeatureSpec(
            spec.HEAD, "uint8", (s, s, 3), ("height", "width", "channels"),
            frame="ego_topdown",
            description=f"egocentric {n}x{n}-cell occluded view, agent at "
                        "the centre facing up"),
        spec.INSTRUCTION: FeatureSpec(
            spec.INSTRUCTION, "string", (1,),
            description="the episode's task in natural language"),
        spec.STATE: FeatureSpec(
            spec.STATE, "float32", (3,), spec.STATE_NAMES_2D,
            units=dict(spec.STATE_UNITS_2D), frame="ego",
            description="pose relative to the start of the episode"),
        spec.TEXT: FeatureSpec(
            spec.TEXT, "string", (1,),
            description="templated description of the view"),
        spec.STRUCTURED: FeatureSpec(
            spec.STRUCTURED, "string", (1,),
            description="the view as JSON: {here, cells: [{f, r, terrain, "
                        "semantics?, objects?}]}"),
    }
    if adapter.topdown:
        out[spec.TOPDOWN] = FeatureSpec(
            spec.TOPDOWN, "uint8", (s, s, 3), ("height", "width", "channels"),
            frame="world",
            description="the observed map (cells seen and believed free) "
                        "in world coordinates; reveals absolute heading")
    return out


def privileged_features() -> dict:
    """What ``info["privileged"]`` records (never observed)."""
    return {
        "world_pose": {"shape": [3], "names": ["x", "y", "yaw"],
                       "units": {"x": "cell", "y": "cell", "yaw": "rad"},
                       "frame": "world", "per": "step"},
        "goal_pose": {"shape": [2], "names": ["x", "y"], "units": "cell",
                      "frame": "world", "per": "step",
                      "nullable": True},
        "d_goal": {"shape": [], "units": "cell",
                   "description": "free-cell graph distance to the active "
                                  "goal, doors passable",
                   "per": "step", "nullable": True},
        "goal_visible": {"shape": [], "dtype": "bool", "per": "step"},
        "region_id": {"shape": [], "dtype": "int64", "per": "step",
                      "description": "index into topology.regions"},
        "topology": {"per": "episode",
                     "description": "certified homology, the region table "
                                    "(open, chamber, room, door, hallway, "
                                    "...) and every layout feature"},
        "goals": {"per": "episode",
                  "description": "every goal's cell and category"},
    }


def _ext_entries(ext: dict | None) -> dict:
    out = {}
    for producer, fields in (ext or {}).items():
        spec.privileged_ext_key(producer, "x")  # validates the name
        out[spec.EXT_PREFIX + producer] = dict(fields)
    return out


def split_info(env) -> dict:
    """The split this world's seed belongs to, and holdout tags.

    Splits map TopoGym's seed bands as they are (tune/train/val/test),
    which is what keeps them disjoint. A world outside every band (the
    canonical specimen, seed 0) is in no split. Holdouts are tags: the
    held-out families and the sizes beyond the extrapolation boundary
    of the default benchmark.
    """
    from topogym import benchmarks

    base = env.unwrapped
    seed = base.layout_seed
    split = benchmarks.split_of(seed) if seed is not None else None
    tags = []
    env_id = getattr(getattr(env, "spec", None), "id", None) or \
        getattr(getattr(base, "spec", None), "id", None)
    if env_id and env_id.startswith("TopoGym/"):
        name = env_id[len("TopoGym/"):].rsplit("-v", 1)[0]
        family = benchmarks.family_of(name)
        held = benchmarks.declared_family(family)
        if held in benchmarks.HELD_OUT_FAMILIES:
            tags.append(spec.holdout_tag("family", held))
        if base.layout is not None:
            size = max(base.layout.base.layout_size())
            if benchmarks.in_benchmark(name) and \
                    size > benchmarks.EXTRAPOLATION_TRAIN_MAX:
                tags.append(spec.holdout_tag("size", size))
    return {"split": split, "tags": tags, "seed": seed}


def manifest(env, *, ext: dict | None = None) -> dict:
    """The canonical manifest of ``env`` (built with
    ``obs_mode="canonical"`` or wrapped with :func:`wrap`).

    ``ext`` declares producer-specific privileged fields: ``{producer:
    {field: description}}`` becomes ``privileged["ext.<producer>"]``,
    matching ``privileged.ext.<producer>.<field>`` columns. This package
    records it and does not interpret it.
    """
    import topogym

    adapter = _adapter(env)
    base = adapter.env
    n = 2 * base.view_radius + 1
    px = adapter.image_size // n
    fixed_start = base._motion == "fourway" and base.layout_seed is not None \
        and not base.teleport
    env_id = getattr(getattr(env, "spec", None), "id", None) or \
        getattr(getattr(base, "spec", None), "id", None)
    cameras = {
        "head": {
            "model": "orthographic_topdown", "facing": "up",
            "cells": [n, n], "px_per_cell": px,
            "intrinsics": None, "T_body_cam": None,
            "note": "a rendered occupancy view, not a perspective camera",
        },
    }
    if adapter.topdown:
        cameras["topdown"] = {"model": "orthographic_topdown",
                              "frame": "world", "intrinsics": None}
    from topogym.canonical import world

    categories = (world.primary_category(base),) + \
        world.EXTRA_CATEGORIES[:adapter.n_goals - 1]
    goals = [{"label": spec.CATEGORIES[k].label,
              "synset": spec.CATEGORIES[k].synset}
             for k in categories] if base.goal_exists else None
    return {
        "spec_version": spec.CANONICAL_SPEC_VERSION,
        "producer": {"name": "topogym", "version": topogym.__version__},
        "env_id": env_id,
        "features": {k: v.to_dict() for k, v in features(adapter).items()},
        "privileged": {**privileged_features(), **_ext_entries(ext)},
        "frames": {
            "world": {"axes": "x right, y down", "units": "cell",
                      "yaw": "from +x toward +y"},
            "ego": {"convention": "REP-103", "axes": "x forward, y left",
                    "yaw": "counter-clockwise", "units": "cell, rad",
                    "origin": "the pose at the start of the episode",
                    "on_glued_worlds": "dead reckoned (a covering-space "
                                       "pose, not a map position)"},
            "episode_frame": "per_world" if fixed_start else "per_episode",
        },
        "cameras": cameras,
        "body": {"kind": "grid_agent", "dof": 3, "can_fly": False,
                 "radius_cells": 0.5,
                 "step": {"forward_cells": 1, "turn_rad": math.pi / 2},
                 "motion": base._motion},
        "actions": {
            "level": adapter.level,
            "words": {"vocab": list(spec.WORDS),
                      "semantics": dict(spec.WORD_SEMANTICS)},
            "waypoint": {"names": list(spec.WAYPOINT_NAMES),
                         "units": dict(spec.WAYPOINT_UNITS),
                         "frame": "ego at issue",
                         "range_cells": base.view_radius,
                         "dz": "ignored (planar body)",
                         "planner": "breadth-first over observed free cells",
                         "uses_privileged_map": False},
            "stop": {"stop_to_succeed": adapter.stop_to_succeed,
                     "goal_radius_cells": 1,
                     "default": "the goal pays on arrival; stop ends the "
                                "episode without success"},
        },
        "goals": {"n_goals": adapter.n_goals, "categories": goals},
        "instructions": {"phrasing": adapter.phrasing,
                         "templates": {k: list(v) for k, v in
                                       spec.TEMPLATES.items()}},
        "split": split_info(env),
    }
