"""Structure accessors: betti dataclass, graph, paths, bottlenecks."""

import logging

import gymnasium as gym
import pytest

import topogym  # noqa: F401
from topogym import BettiNumbers


def _env(env_id="TopoGym/Decoys2-50-v0", **kw):
    env = gym.make(env_id, seed=1, **kw).unwrapped
    env.reset(seed=0)
    return env


def test_betti_numbers_dataclass_conventions():
    env = _env()
    walkable = env.topology.betti_numbers_doors_dont_count_as_walls()
    sealed = env.topology.betti_numbers_doors_count_as_walls()
    assert isinstance(walkable, BettiNumbers)
    assert walkable.as_tuple() == tuple(env.topology.betti_z2)
    assert tuple(sealed) == tuple(env.topology.betti_z2_sealed)
    assert walkable.b0 == 1  # doors walkable: one component
    assert sealed.b0 == walkable.b0 + 1  # the chamber interior seals off
    assert sealed.b1 == walkable.b1
    assert str(walkable).startswith("b0=")


def test_graph_is_networkx_over_free_cells():
    nx = pytest.importorskip("networkx")
    env = _env()
    g = env.graph()
    assert isinstance(g, nx.Graph)
    assert g.number_of_nodes() == len(env.layout.free_cells)
    assert nx.is_connected(g)  # doors passable: one component


def test_shortest_path_defaults_start_to_goal():
    env = _env()
    path = env.shortest_path()
    assert path[0] == env.layout.start and path[-1] == env.layout.goal
    for a, b in zip(path, path[1:]):
        assert b in env.layout.base.neighbors(a)
    with pytest.raises(ValueError):
        env.shortest_path((0, 0), next(iter(
            c for c, t in env.layout.cell_types.items() if t == 1
        )))


def test_bottlenecks_are_straight_through_cells():
    env = _env("TopoGym/Dilution-50-v0")
    necks = env.bottlenecks()
    (door,) = env.layout.doors
    assert door in necks  # a doorway can only be passed straight through
    free = set(env.layout.free_cells)
    for cell in necks:
        nbrs = [n for n in env.layout.base.neighbors(cell) if n in free]
        assert len(nbrs) == 2
    # Corridor worlds are full of them.
    env2 = _env("TopoGym/Bottleneck6-100-v0")
    assert len(env2.bottlenecks()) >= 5 * 6  # every corridor cell


def test_debug_logs_bottlenecks_on_reset(monkeypatch, caplog):
    monkeypatch.setenv("TOPOGYM_DEBUG", "1")
    env = gym.make("TopoGym/Dilution-50-v0", seed=1).unwrapped
    with caplog.at_level(logging.DEBUG, logger="topogym"):
        env.reset(seed=0)
    assert "bottlenecks=" in caplog.text
