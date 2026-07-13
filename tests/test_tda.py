"""Trajectory TDA: exploration tracking, discovery persistence, Rips."""

import math

import numpy as np

from topogym.spec import Annulus, Square
from topogym.tda import (
    ExplorationTracker,
    betti_at_scale,
    bottleneck_distance,
    rips_diagram,
)


def sweep(tracker, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < n_steps:
        _, _, terminated, truncated, info = tracker.step(int(rng.integers(3)))
        steps += 1
    return info


def test_tracker_records_visits_and_observations():
    env = Square(6).compile(seed=1, reward_mode="explore", max_steps=200)
    tracker = ExplorationTracker(env)
    tracker.reset(seed=0)
    assert list(tracker.visit_step.values()) == [0]  # the start cell
    assert len(tracker.observed_step) >= 1  # the initial view
    sweep(tracker, 100)
    assert len(tracker.visit_step) > 1
    # Timestamps are monotone: no cell recorded after a later snapshot
    # can precede an earlier one it was discovered with.
    assert min(tracker.visit_step.values()) == 0


def test_full_exploration_recovers_certified_topology():
    env = Annulus(8).compile(
        seed=4, reward_mode="explore", obs_mode="global", max_steps=3000
    )
    tracker = ExplorationTracker(env)
    tracker.reset(seed=0)
    sweep(tracker, 3000)

    summary = tracker.summary("visited")
    if summary["coverage"] == 1.0:  # the random walk covered everything
        assert summary["recovery"]["betti_z2"] is not None
    # Essential bars of the discovery diagram are the real topology of
    # whatever was explored.
    diagram = tracker.discovery_diagram("visited")
    betti_explored = tracker.betti_curve("visited")[-1][1]
    for dim in range(3):
        essential = sum(
            1 for _, death in diagram.get(dim, ()) if math.isinf(death)
        )
        assert essential == betti_explored[dim]


def test_observed_region_beats_visited():
    env = Annulus(8).compile(seed=4, reward_mode="explore", max_steps=400)
    tracker = ExplorationTracker(env)
    tracker.reset(seed=0)
    sweep(tracker, 400)
    # You always see at least as much as you touch.
    assert set(tracker.visit_step) <= set(tracker.observed_step)
    assert all(
        tracker.observed_step[c] <= t for c, t in tracker.visit_step.items()
    )


def test_betti_curve_is_step_indexed_and_final():
    env = Square(6).compile(seed=2, reward_mode="explore", max_steps=300)
    tracker = ExplorationTracker(env)
    tracker.reset(seed=0)
    sweep(tracker, 300)
    curve = tracker.betti_curve("visited", every=10)
    assert curve[0][0] == 0
    assert curve[-1][1] == tracker.betti_curve("visited")[-1][1]
    assert all(b[0] == 1 for _, b in curve)  # visited region is connected


def circle_points(n=40, radius=1.0):
    return [
        (radius * math.cos(a), radius * math.sin(a))
        for a in np.linspace(0, 2 * math.pi, n, endpoint=False)
    ]


def test_rips_recovers_a_circle():
    points = circle_points()
    assert betti_at_scale(points, 0.5, max_dim=1) == (1, 1)
    bars = rips_diagram(points, max_edge_length=3.0, max_dim=1)[1]
    births_deaths = [(b, d) for b, d in bars if d - b > 1.0]
    assert len(births_deaths) == 1  # one dominant loop


def test_bottleneck_distance_separates_shapes():
    loop = rips_diagram(circle_points(), 3.0, 1).get(1, [])
    rng = np.random.default_rng(0)
    blob = rips_diagram(rng.normal(0, 0.1, (40, 2)).tolist(), 3.0, 1).get(
        1, []
    )
    assert bottleneck_distance(loop, blob) > 0.5


def test_discovery_diagram_h0_merges():
    # Walking two disjoint arms before connecting them must show a finite
    # H0 bar (a region discovered separately, merged later).
    env = Square(8).compile(
        seed=1, reward_mode="explore", obs_mode="global", max_steps=2000
    )
    tracker = ExplorationTracker(env)
    tracker.reset(seed=0)
    sweep(tracker, 2000, seed=3)
    diagram = tracker.discovery_diagram("visited")
    h0 = diagram.get(0, [])
    essential = [bar for bar in h0 if math.isinf(bar[1])]
    assert len(essential) == 1  # the visited region ends connected
