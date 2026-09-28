"""Hindsight goal descriptions: any place as an instruction.

:func:`describe_goal` phrases a cell the way the environment's own
instructions phrase goals -- same templates, same phrasing tags -- so a
trajectory can be relabelled with the place it actually reached and
still read like the data it sits beside.
"""

from __future__ import annotations

from topogym.canonical import spec, world


def _adapter(env):
    from topogym.canonical.manifest import _adapter as find

    try:
        return find(env)
    except ValueError:
        return None


def describe_goal(env, cell=None, *, phrasing="canonical") -> dict:
    """An instruction for going to ``cell`` (default: where the agent
    stands), with the goal it corresponds to.

    In order of preference, the instruction names: a goal object at or
    next to the cell (exactly the instruction that goal gets); the room,
    doorway or hallway the cell is in; or, in the open, the cell's
    offset from where the episode started.

    Returns ``{"instruction", "phrasing", "goal": {"cell", "category",
    "region_id", "region_kind"}}``.
    """
    base = env.unwrapped
    if base.layout is None:
        raise RuntimeError("reset the env first")
    adapter = _adapter(env)
    here = base._state.cell
    cell = here if cell is None else tuple(cell)
    k = spec.parse_phrasing(phrasing)
    of, table = world.regions(base.layout)
    rid = of.get(cell, world.OPEN)
    goal = {"cell": list(cell), "category": None, "region_id": rid,
            "region_kind": table[rid]["kind"]}
    goals = adapter._goals if adapter is not None and adapter._goals else \
        world.place_goals(base, 1) if base.goal_exists else ()
    for gcell, category in goals:
        near = world.distance_field(base.layout, gcell).get(cell)
        if near is not None and near <= 1:
            goal["category"] = category
            text = spec.instruction("goto", k, target=world.goal_target(
                base.layout, gcell, category))
            return {"instruction": text, "phrasing": spec.phrasing_tag(k),
                    "goal": goal}
    place = world.place_phrase(base.layout, cell)
    if place is None:
        place = _offset_place(base, adapter, cell)
    text = spec.instruction("goto_place", k, place=place)
    return {"instruction": text, "phrasing": spec.phrasing_tag(k),
            "goal": goal}


def _offset_place(base, adapter, cell) -> str:
    """``"the spot two cells ahead and one to the left of where you
    started"``, in the episode's ego frame."""
    if adapter is not None and cell == base._state.cell:
        x, y, _ = adapter._pose
        forward, right = round(x), -round(y)
    elif adapter is not None and adapter._square():
        px, py = base.layout.base.layout_coords(cell)
        fx, fy = adapter._f0
        dx, dy = px - adapter._p0[0], py - adapter._p0[1]
        forward, right = dx * fx + dy * fy, -(dx * fy - dy * fx)
    else:
        return "the open floor"
    if (forward, right) == (0, 0):
        return "the spot where you started"
    return (f"the spot {spec.offset_phrase(forward, right)} of where you "
            "started")
