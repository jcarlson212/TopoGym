"""Ollivier-Ricci curvature: exactness and the coverage stat."""

import itertools

import gymnasium as gym
import pytest

import topogym  # noqa: F401
from topogym.curvature import hungarian, ollivier_ricci
from topogym.stats import StatsRecorder


def test_hungarian_matches_brute_force():
    costs = [
        [[3.0]],
        [[1, 2], [2, 1]],
        [[4, 1, 3], [2, 0, 5], [3, 2, 2]],
    ]
    for cost in costs:
        n = len(cost)
        brute = min(
            sum(cost[i][p[i]] for i in range(n))
            for p in itertools.permutations(range(n))
        )
        assert hungarian(cost) == pytest.approx(brute)


def test_corridors_are_negatively_curved():
    env = gym.make("TopoGym/Grid2D-v0", base="square", size=15, actions="fourway",
                   n_holes=0, n_chambers=1, n_decoys=0,
                   layout_seed=1).unwrapped
    env.reset(seed=0)
    ricci = ollivier_ricci(set(env.layout.free_cells),
                           env.layout.base.neighbors)
    (door,) = env.layout.doors
    # An open-field interior cell (all 8 surrounding cells free).
    free = set(env.layout.free_cells)
    interior = next(
        c for c in sorted(free)
        if all((c[0] + dx, c[1] + dy) in free
               for dx in (-1, 0, 1) for dy in (-1, 0, 1))
    )
    assert ricci[door] < 0  # the doorway is a bottleneck
    assert ricci[door] < ricci[interior]
    assert abs(ricci[interior]) < 0.3  # flat-ish open space


def test_curvature_coverage_toggle_and_accessor():
    env = StatsRecorder(
        gym.make("TopoGym/Grid2D-v0", base="square", size=13, actions="fourway",
                 n_holes=0, n_chambers=1, n_decoys=0, layout_seed=1),
        track_curvature=True,
    )
    env.reset(seed=0)
    for a in (0, 3, 1, 2, 0, 3):
        env.step(a)
    m = env.metrics()
    assert m.curvature_coverage_below_zero is not None
    assert 0 <= m.curvature_coverage_below_zero <= 1
    # Parametrized accessor works regardless of the toggle.
    off = StatsRecorder(
        gym.make("TopoGym/Grid2D-v0", base="square", size=13, actions="fourway",
                 n_holes=0, n_chambers=1, n_decoys=0, layout_seed=1),
    )
    off.reset(seed=0)
    off.step(0)
    assert off.metrics().curvature_coverage_below_zero is None  # off
    assert 0 <= off.curvature_coverage(0.0) <= 1  # explicit call ok
    # env.ollivier_ricci is cached per layout.
    core = off.env.unwrapped
    assert core.ollivier_ricci() is core.ollivier_ricci()
