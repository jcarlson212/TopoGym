"""Every frozen benchmark entry must generate and carry sane metadata."""

import gymnasium as gym
import pytest

import topogym  # noqa: F401
from topogym import benchmarks as bench


def test_collections_present():
    names = bench.benchmark_names()
    assert "2d_bench_grid_small" in names
    assert "3d_bench_grid_small" in names
    assert "2d_bench_grid_small_directed" in names
    assert "3d_bench_grid_small_directed" in names
    assert len(bench.get_benchmark("2d_bench_grid_small")) == 16
    assert len(bench.get_benchmark("3d_bench_grid_small")) == 8


@pytest.mark.parametrize(
    "entry",
    [e for c in bench.BENCHMARKS.values() for e in c],
    ids=lambda e: e.env_id,
)
def test_entry_generates_with_certified_metadata(entry):
    md = entry.metadata()
    assert md.certified["betti_z2"] is True
    assert md.betti_z2[0] == 1
    assert md.layout_seed == entry.layout_seed
    directed = entry.collection.endswith("_directed")
    assert md.asymmetry["is_symmetric"] == (not directed)
    if directed:
        assert md.asymmetry["mechanisms"]
    if entry.name.startswith("control-"):
        assert all(b == 0 for b in md.betti_z2[1:])
    # Deterministic: regeneration yields identical metadata.
    assert entry.metadata().to_dict() == md.to_dict()


def test_entries_make_and_reset():
    entry = bench.get_benchmark("2d_bench_grid_small")[0]
    env = entry.make()
    obs, info = env.reset(seed=0)
    assert info["topology"]["betti_z2"] == [1, 3, 0]
    env2 = gym.make(entry.env_id)
    obs2, info2 = env2.reset(seed=0)
    assert info2["topology"] == info["topology"]


def test_benchmark_metadata_sweep():
    rows = bench.benchmark_metadata("2d_bench_grid_small")
    assert len(rows) == 16
    assert all("betti_z2" in r and "name" in r and "env_id" in r for r in rows)
    by_name = {r["name"]: r for r in rows}
    assert by_name["plane-6holes"]["betti_z2"] == [1, 6, 0]
    assert by_name["control-maze"]["betti_z2"] == [1, 0, 0]
    assert by_name["klein-rooms"]["orientable"] is False
    assert by_name["torus-rooms"]["genus"] == 1
