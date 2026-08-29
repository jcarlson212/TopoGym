"""The open-field ring: a fixed radius, and a boundary out of reach.

``EnlargedChamberCount`` guarantees the doors are far apart, but puts
them on the world's perimeter -- so a method that finds the wall can
follow it from one chamber to the next, and the score partly measures
how well it sweeps a boundary. Measured in that family, every arm is
pinned against the wall by half of a 1M step budget.

``OpenFieldChamberCount`` removes the wall as a resource by moving the
chambers onto a ring about the start and leaving open floor between
that ring and the boundary. These tests check the two properties that
makes true -- the ring is where it was asked for, and the boundary is
far beyond it -- plus the regression that matters, since the ring
policy is shared with the ``Decoys`` family: a zero radius still means
what it always meant.

Sizes here are the smallest that exercise the geometry, because these
run in the commit gate: k=3 needs only a radius of 36 to hold a 61-cell
door gap, where k=6 would need 62 and a world four times the area. The
registered family's own worlds are checked by
``scripts/soak_door_separation.py``, which is too slow for here.
"""

from __future__ import annotations

import dataclasses

import pytest

import topogym  # noqa: F401  (registers the ids)
from topogym.generation.generator import generate_2d
from topogym.registry import EXTRA_KWARGS, REGISTRY, _open_cfg

SEEDS = range(4)

#: k=3 at radius 40 gives a 69-cell chord, clearing the 61 the doors
#: must be apart, in a world of 240 rather than 530.
K, RADIUS, SIZE = 3, 40, 240


def _cfg(k: int, size: int, radius: int, **kw):
    settings = {"ring_radius": radius, "min_door_distance": 61,
                "max_attempts": 200, **kw}
    return dataclasses.replace(
        _open_cfg(size, n_chambers=k, chamber_placement="around",
                  placement_jitter=4, start_placement="center"),
        **settings)


def _extent(layout) -> tuple:
    xs = [c[0] for c in layout.free_cells]
    ys = [c[1] for c in layout.free_cells]
    return max(xs) + 1, max(ys) + 1


def _door_radii(layout) -> list:
    """Chebyshev distance from the world centre to each chamber door."""
    w, h = _extent(layout)
    cx, cy = w // 2, h // 2
    return [max(abs(spec.cell[0] - cx), abs(spec.cell[1] - cy))
            for f in layout.features if f.kind == "chamber"
            for spec in f.doors]


# -- the ring is where it was asked for --------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_the_ring_sits_at_the_requested_radius(seed):
    """Not at the largest radius that fits, which is all the policy
    could do before a radius could be named."""
    layout = generate_2d(_cfg(K, SIZE, RADIUS), seed=seed)
    # Jitter and the door's position on its chamber wall both move a
    # door off the exact ring, by no more than the chamber plus jitter.
    assert all(abs(r - RADIUS) <= 20 for r in _door_radii(layout))


@pytest.mark.parametrize("seed", SEEDS)
def test_open_floor_lies_beyond_the_ring(seed):
    """The property the family exists for: the boundary is not a rail
    running between the chambers."""
    layout = generate_2d(_cfg(K, SIZE, RADIUS), seed=seed)
    w, h = _extent(layout)
    assert min(w // 2, h // 2) - max(_door_radii(layout)) > 60


def test_growing_the_world_leaves_the_ring_alone():
    """Radius and world size are decoupled -- the whole point of the
    knob. Two sizes, one radius, the same ring."""
    small = _door_radii(generate_2d(_cfg(K, 180, RADIUS), seed=0))
    large = _door_radii(generate_2d(_cfg(K, SIZE, RADIUS), seed=0))
    assert abs(max(small) - max(large)) <= 6


# -- the registered family --------------------------------------------

def test_every_registered_world_puts_the_wall_beyond_reach():
    """Checked on the configuration rather than by generating, because
    these worlds are 230k-372k cells and the gate is not the place for
    that. The claim is arithmetic anyway: the wall sits at half the
    side, and it has to be far beyond one episode."""
    names = sorted(n for n in REGISTRY
                   if n.startswith("OpenFieldChamberCount"))
    assert names, "the family did not register"
    for name in names:
        cfg = REGISTRY[name]
        horizon = EXTRA_KWARGS[name]["max_steps"]
        assert cfg.ring_radius > 0, name
        # Open floor between ring and wall, and a wall many episodes out.
        margin = cfg.size // 2 - cfg.ring_radius
        assert margin > 3 * horizon, (name, margin, horizon)
        assert cfg.min_door_distance > horizon, name


# -- the regression that matters --------------------------------------

def test_zero_radius_keeps_the_old_ring():
    """``Decoys`` rings its decoys with the same policy. A zero radius
    has to keep meaning 'the largest that fits', or that family's
    published layouts move under it."""
    fitted = generate_2d(_cfg(K, 180, 0), seed=0)
    named = generate_2d(_cfg(K, 180, RADIUS), seed=0)
    w, h = _extent(fitted)
    assert max(_door_radii(fitted)) > max(_door_radii(named))
    assert max(_door_radii(fitted)) >= min(w // 2, h // 2) - 20


def test_an_oversized_radius_clamps_instead_of_escaping():
    """Asking for a ring larger than the world degrades to the fitted
    ring rather than placing chambers out of bounds."""
    layout = generate_2d(_cfg(K, 180, 10_000), seed=0)
    w, h = _extent(layout)
    assert all(r <= min(w // 2, h // 2) for r in _door_radii(layout))
