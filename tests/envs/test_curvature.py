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


def test_incremental_ricci_update_equals_full_recompute():
    """The archive-growth path: updating only the edges near newly
    observed cells must reproduce the full field exactly, at every
    stage of growth. Exactness is the contract that lets a method
    maintain curvature incrementally instead of paying the whole
    observed set per step."""
    import gymnasium as gym

    from topogym.curvature import (
        ollivier_ricci,
        ollivier_ricci_edges,
        per_cell_curvature,
        update_ricci_edges,
    )

    env = gym.make("TopoGym/Grid2D-v0", base="square", size=17,
                   actions="fourway", n_holes=1, n_chambers=1,
                   n_decoys=0, layout_seed=3).unwrapped
    env.reset(seed=0)
    free = sorted(map(tuple, env.layout.free_cells))
    neighbors = env.layout.base.neighbors

    # Grow the observed set in uneven bites, as an archive would.
    stops = [5, 6, 19, 40, 41, 90, len(free)]
    observed: set = set()
    edges = {}
    for stop in stops:
        added = set(free[len(observed):stop])
        observed |= added
        edges = update_ricci_edges(edges, observed, added, neighbors)
        assert per_cell_curvature(edges) == ollivier_ricci(observed,
                                                           neighbors)
    assert edges == ollivier_ricci_edges(observed, neighbors)


# -- oracle tests for the hand-rolled optimal transport ----------------

def test_hungarian_matches_brute_force_on_small_matrices():
    """The assignment cost is the one hand-rolled numeric kernel every
    curvature number flows through; check it against exhaustive
    enumeration, which cannot be wrong."""
    import itertools
    import random

    from topogym.curvature import hungarian

    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 6)
        cost = [[rng.uniform(0, 10) for _ in range(n)] for _ in range(n)]
        best = min(sum(cost[i][p[i]] for i in range(n))
                   for p in itertools.permutations(range(n)))
        assert abs(hungarian(cost) - best) < 1e-9


def test_hungarian_matches_scipy_on_larger_matrices():
    import random

    import numpy as np
    scipy_opt = pytest.importorskip("scipy.optimize")

    from topogym.curvature import hungarian

    rng = random.Random(11)
    for _ in range(50):
        n = rng.randint(2, 12)
        cost = [[rng.uniform(0, 100) for _ in range(n)]
                for _ in range(n)]
        rows, cols = scipy_opt.linear_sum_assignment(np.asarray(cost))
        reference = float(np.asarray(cost)[rows, cols].sum())
        assert abs(hungarian(cost) - reference) < 1e-9


def test_uncapped_field_matches_reference_library():
    """With the distance cap lifted out of the way, our field is
    standard Ollivier-Ricci (alpha=0, uniform neighbour measures,
    exact W1) and must agree with the reference implementation.
    Skips unless GraphRicciCurvature is installed."""
    grc = pytest.importorskip("GraphRicciCurvature.OllivierRicci")
    nx = pytest.importorskip("networkx")

    import topogym.curvature as curvature

    free = {(x, y) for x in range(5) for y in range(5)} - {(2, 2)}

    def neighbors(cell):
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    old_cap = curvature._DIST_CAP
    curvature._DIST_CAP = 10_000
    try:
        edges = curvature.ollivier_ricci_edges(free, neighbors)
    finally:
        curvature._DIST_CAP = old_cap

    graph = nx.Graph()
    for (u, v) in edges:
        graph.add_edge(u, v)
    orc = grc.OllivierRicci(graph, alpha=0.0, method="OTD",
                            verbose="ERROR")
    orc.compute_ricci_curvature()
    for (u, v), ours in edges.items():
        theirs = orc.G[u][v]["ricciCurvature"]
        assert abs(ours - theirs) < 1e-6, ((u, v), ours, theirs)


def test_uncapped_curvature_matches_analytic_ground_truth():
    """Hand-derived Ollivier-Ricci values (alpha=0, uniform neighbour
    measures, exact W1), cap lifted so the capped metric is the true
    one. Flat, positively curved, and negatively curved -- the last
    being the bottleneck signature the curvature term exists to find.
    """
    import topogym.curvature as curvature

    def field(adjacency):
        cells = set(adjacency)
        return cells, lambda cell: list(adjacency[cell])

    old_cap = curvature._DIST_CAP
    curvature._DIST_CAP = 10_000
    try:
        # Interior edge of a large 4-grid: flat, kappa = 0.
        grid = {(x, y) for x in range(9) for y in range(9)}
        def nbrs(cell):
            x, y = cell
            return [c for c in ((x + 1, y), (x - 1, y),
                                (x, y + 1), (x, y - 1)) if c in grid]
        assert curvature.edge_curvature((4, 4), (4, 5), grid, nbrs) == 0.0

        # Triangle: the shared neighbour keeps half the mass in place,
        # the other half moves one step: W1 = 1/2, kappa = +1/2.
        tri, tri_n = field({'a': 'bc', 'b': 'ac', 'c': 'ab'})
        assert curvature.edge_curvature('a', 'b', tri, tri_n) == 0.5

        # Barbell bridge: two hubs of leaves joined by one edge. Best
        # coupling moves two of three units one step and one unit
        # three steps: W1 = 5/3, kappa = -2/3. Negative curvature is
        # exactly the doorway/bottleneck signature.
        bar, bar_n = field({
            'c1': ['l1', 'l2', 'c2'], 'c2': ['c1', 'l3', 'l4'],
            'l1': ['c1'], 'l2': ['c1'], 'l3': ['c2'], 'l4': ['c2'],
        })
        value = curvature.edge_curvature('c1', 'c2', bar, bar_n)
        assert abs(value - (1.0 - 5.0 / 3.0)) < 1e-9
    finally:
        curvature._DIST_CAP = old_cap


def test_ricci_field_grow_equals_fresh_build():
    """The class-shaped contract: a field grown in uneven bites is
    bit-identical -- edges and per-cell -- to one built fresh over the
    same cells."""
    import gymnasium as gym

    from topogym.curvature import RicciField

    env = gym.make("TopoGym/Grid2D-v0", base="square", size=17,
                   actions="fourway", n_holes=1, n_chambers=1,
                   n_decoys=0, layout_seed=3).unwrapped
    env.reset(seed=0)
    free = sorted(map(tuple, env.layout.free_cells))
    neighbors = env.layout.base.neighbors

    field = RicciField(free[:5], neighbors)
    for stop in (6, 19, 40, 41, 90, len(free)):
        field.grow(free[len(field.free):stop])
        fresh = RicciField(free[:stop], neighbors)
        assert field.edges == fresh.edges
        assert field.per_cell() == fresh.per_cell()


def test_ricci_field_contracts():
    """The class's small print: growing by already-known cells is a
    no-op; per-cell results are refreshed after growth; and the
    functional wrapper never mutates the caller's table."""
    from topogym.curvature import (
        RicciField,
        ollivier_ricci_edges,
        update_ricci_edges,
    )

    cells = {(x, y) for x in range(4) for y in range(4)}

    def neighbors(cell):
        x, y = cell
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    field = RicciField(sorted(cells)[:8], neighbors)
    before = dict(field.edges)
    field.grow(sorted(cells)[:8])          # nothing new
    assert field.edges == before

    first = field.per_cell()
    field.grow(sorted(cells)[8:])
    assert field.per_cell() != first       # cache invalidated by grow
    assert field.free == cells

    # Wrapper parity and no-mutation.
    table = ollivier_ricci_edges(set(sorted(cells)[:8]), neighbors)
    snapshot = dict(table)
    grown = update_ricci_edges(table, cells, cells - field.free
                               | set(sorted(cells)[8:]), neighbors)
    assert table == snapshot               # input untouched
    assert grown == field.edges
