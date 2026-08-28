"""The ``min_door_distance`` guarantee.

A family whose subject is the chamber *count* has to hold everything
else fixed, and the thing most easily confounded with count is how far
apart the chambers are: pack them close and one episode reaches two,
and the experiment measures crowding rather than counting. The
constraint here makes the separation a property of the specimen, so
these tests check the property rather than the code path -- every one
of them re-derives the distance from the layout instead of trusting
what generation recorded.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools

import pytest

import topogym  # noqa: F401  (registers the ids)
from topogym.generation.generator import GenerationError, generate_2d
from topogym.registry import _open_cfg

#: Small and quick: these run in the pre-commit gate. The exhaustive
#: sweep over thousands of seeds lives in
#: ``scripts/soak_door_separation.py``, which is too slow for here.
SEEDS = range(8)


def _cfg(k: int, size: int, distance: int, **kw):
    settings = {"min_door_distance": distance, "max_attempts": 200, **kw}
    return dataclasses.replace(
        _open_cfg(size, n_chambers=k, chamber_placement="perimeter",
                  placement_jitter=4, start_placement="center"),
        **settings)


def _doors(layout) -> list:
    return [tuple(spec.cell)
            for f in layout.features if f.kind == "chamber"
            for spec in f.doors]


def _walk_distances(layout, source):
    """BFS over free cells, computed here rather than imported, so a
    bug in the generator's own helper cannot hide behind itself."""
    free = set(map(tuple, layout.free_cells))
    seen = {source: 0}
    queue = collections.deque([source])
    while queue:
        x, y = queue.popleft()
        for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbour in free and neighbour not in seen:
                seen[neighbour] = seen[(x, y)] + 1
                queue.append(neighbour)
    return seen


def _min_pairwise(layout) -> int | None:
    doors = _doors(layout)
    if len(doors) < 2:
        return None
    best = None
    for one, other in itertools.combinations(doors, 2):
        reach = _walk_distances(layout, one).get(other)
        if reach is not None:
            best = reach if best is None else min(best, reach)
    return best


# -- the guarantee ----------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_every_seed_meets_the_requested_separation(seed):
    """The property the family is named for, re-derived from geometry."""
    want = 61
    layout = generate_2d(_cfg(3, 60, want), seed=seed)
    assert _min_pairwise(layout) >= want


@pytest.mark.parametrize("seed", SEEDS)
def test_no_two_doors_touch_even_diagonally(seed):
    """Graph distance alone would let two doors sit corner to corner:
    non-adjacent over 4-connected free cells, adjacent to the eye."""
    layout = generate_2d(_cfg(3, 60, 61), seed=seed)
    for one, other in itertools.combinations(_doors(layout), 2):
        assert max(abs(one[0] - other[0]), abs(one[1] - other[1])) > 1


@pytest.mark.parametrize("seed", SEEDS)
def test_the_metadata_reports_what_the_geometry_says(seed):
    """The certificate is auditable: a reader who trusts the metadata
    and a reader who measures the layout must agree."""
    layout = generate_2d(_cfg(3, 60, 61), seed=seed)
    assert (layout.metadata.connectivity["min_door_distance"]
            == _min_pairwise(layout))


def test_the_separation_exceeds_the_horizon_it_was_chosen_for():
    """The premise the experiment rests on: with doors more than L
    apart, no episode of L steps reaches two of them."""
    horizon = 60
    layout = generate_2d(_cfg(3, 60, horizon + 1), seed=0)
    assert _min_pairwise(layout) > horizon


# -- randomisation ----------------------------------------------------

def test_seeds_move_the_chambers():
    """Randomised placement, not a fixed arrangement wearing a seed."""
    arrangements = {tuple(sorted(_doors(generate_2d(_cfg(3, 60, 61),
                                                    seed=s))))
                    for s in SEEDS}
    assert len(arrangements) > 1


def test_seeds_move_the_achieved_separation_too():
    """If every seed produced the same distance the constraint would be
    pinning the layout rather than bounding it."""
    seen = {_min_pairwise(generate_2d(_cfg(3, 60, 61), seed=s))
            for s in SEEDS}
    assert len(seen) > 1


@pytest.mark.parametrize("seed", SEEDS)
def test_generation_stays_deterministic_under_the_constraint(seed):
    """Rejection sampling must not consume entropy differently between
    identical calls, or the benchmark's determinism claim dies."""
    first = generate_2d(_cfg(3, 60, 61), seed=seed)
    second = generate_2d(_cfg(3, 60, 61), seed=seed)
    assert _doors(first) == _doors(second)
    assert first.start == second.start and first.goal == second.goal
    assert set(map(tuple, first.free_cells)) == set(
        map(tuple, second.free_cells))


# -- the boundaries ---------------------------------------------------

def test_zero_leaves_placement_alone():
    """Every family that does not ask for the constraint must generate
    exactly as it did before it existed."""
    without = generate_2d(_cfg(3, 60, 0), seed=0)
    assert "min_door_distance" in without.metadata.connectivity
    # Recorded (it is measurable either way) but not enforced: the
    # unconstrained arrangement is free to be closer than any bound.
    assert (without.metadata.connectivity["min_door_distance"]
            == _min_pairwise(without))


def test_an_impossible_separation_fails_loudly():
    """Better a GenerationError than a world quietly missing the
    property its family is defined by."""
    with pytest.raises(GenerationError):
        generate_2d(_cfg(8, 40, 500, max_attempts=6), seed=0)


def test_one_chamber_has_no_pair_to_separate():
    layout = generate_2d(_cfg(1, 60, 61), seed=0)
    assert _min_pairwise(layout) is None
    assert "min_door_distance" not in layout.metadata.connectivity
