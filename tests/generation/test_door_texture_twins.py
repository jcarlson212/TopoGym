"""Door-textured twins: the same world, with the door made visible.

The three standalone chamber families carry no textures at all, which
was the point -- nothing tells an agent a door is there but walking
into it. That also means a method whose score reads the door texture
slot has nothing to read, so on these families such an arm is inert
and an arm combining it with another subscore is a duplicate of that
subscore alone.

The twins switch the door slot on so those arms become live. The claim
that makes the comparison controlled is that *only* that changes: the
marking runs after generation and consumes no randomness, so a twin
and its original agree cell for cell. These tests hold them to it,
which is also what licenses the registry-wide horizon test to skip a
twin rather than re-derive an optimal route it already has.
"""

from __future__ import annotations

import pytest

import topogym  # noqa: F401  (registers the ids)
from topogym.baselines.gridworld2dv1.archive import layout_fingerprint
from topogym.core.constants import TEX_DOOR
from topogym.generation.generator import generate_2d
from topogym.registry import EXTRA_KWARGS, REGISTRY

#: Small members of each family: the property is structural, and the
#: large worlds cost minutes apiece to generate.
PAIRS = [
    ("EnlargedChamberCount2-60", "EnlargedChamberCountDoors2-60"),
    ("EnlargedChamberCount4-80", "EnlargedChamberCountDoors4-80"),
    ("EpicChase2-70", "EpicChaseDoors2-70"),
]
SEEDS = (0, 3)


def _twins():
    return sorted(n for n in REGISTRY if REGISTRY[n].door_textures)


def test_every_family_has_a_twin_and_every_twin_an_original():
    twins = _twins()
    assert twins, "no door-textured twin registered"
    for twin in twins:
        plain = twin.replace("Doors", "", 1)
        assert plain in REGISTRY, f"{twin} has no original"
        assert not REGISTRY[plain].door_textures
    # And the relation is onto: nothing textured was left behind.
    for plain in [n for n in REGISTRY if not REGISTRY[n].door_textures]:
        if not plain.startswith(("EpicChase", "EnlargedChamberCount",
                                 "OpenFieldChamberCount")):
            continue
        family = plain.rstrip("0123456789-")
        expected = plain.replace(family, family + "Doors", 1)
        assert expected in REGISTRY, f"{plain} has no twin"


@pytest.mark.parametrize("plain,twin", PAIRS)
@pytest.mark.parametrize("seed", SEEDS)
def test_a_twin_is_its_original_cell_for_cell(plain, twin, seed):
    """The controlled part of the comparison. If this ever fails, the
    twins stop being the same experiment with one variable moved."""
    a = generate_2d(REGISTRY[plain], seed=seed)
    b = generate_2d(REGISTRY[twin], seed=seed)
    assert sorted(map(tuple, a.free_cells)) == sorted(map(tuple, b.free_cells))
    assert sorted(a.doors) == sorted(b.doors)
    assert tuple(a.start) == tuple(b.start)
    assert tuple(a.goal) == tuple(b.goal)
    assert (sorted(a.cell_types.items(), key=repr)
            == sorted(b.cell_types.items(), key=repr))


@pytest.mark.parametrize("plain,twin", PAIRS)
@pytest.mark.parametrize("seed", SEEDS)
def test_the_twin_marks_every_door_and_the_original_none(plain, twin, seed):
    a = generate_2d(REGISTRY[plain], seed=seed)
    b = generate_2d(REGISTRY[twin], seed=seed)
    assert not (a.extras or {}).get("textures"), (
        f"{plain} is supposed to carry no local signal")
    marked = (b.extras or {}).get("textures") or {}
    assert b.doors, "a chamber family with no doors is not one"
    for cell in b.doors:
        assert TEX_DOOR in (marked.get(cell) or ()), (
            f"{twin} left door {cell} unmarked")


@pytest.mark.parametrize("plain,twin", PAIRS)
def test_the_two_are_different_worlds_to_an_archive(plain, twin):
    """Same geometry, but not interchangeable: an archive keyed by
    fingerprint must not reuse one world's entries on the other."""
    a = generate_2d(REGISTRY[plain], seed=0)
    b = generate_2d(REGISTRY[twin], seed=0)
    assert layout_fingerprint(a) != layout_fingerprint(b)


def test_a_twin_inherits_the_budget_it_is_compared_under():
    """A twin scored under a different horizon would not be a control."""
    for twin in _twins():
        plain = twin.replace("Doors", "", 1)
        assert (EXTRA_KWARGS.get(twin, {}).get("max_steps")
                == EXTRA_KWARGS.get(plain, {}).get("max_steps")), twin
