"""Parallel evaluation must be a scheduling detail, never a result."""

import pytest

from topogym.baselines.gridworld2dv1.evaluate import evaluate_split, instance_seed
from topogym.baselines.gridworld2dv1.instances import load_split
from topogym.baselines.gridworld2dv1.parallel import default_workers, map_instances
from topogym.baselines.gridworld2dv1.random_walk import RandomPolicyFactory

COMPARED = ("instance", "success_rate", "unique_states", "interactions",
            "lifetime_coverage")


def _rows(n=6):
    return load_split("test")[:n]


def test_instance_seed_depends_on_the_instance_not_the_order():
    rows = _rows(3)
    seeds = [instance_seed(0, row) for row in rows]
    assert len(set(seeds)) == len(seeds)          # distinct per instance
    assert seeds == [instance_seed(0, r) for r in reversed(rows)][::-1]
    assert instance_seed(1, rows[0]) != seeds[0]  # and on the base seed


def test_parallel_and_serial_agree_exactly():
    """Sharding must not change a number. A policy built once and
    reused would carry its random stream between instances, making
    results depend on which worker took which row."""
    rows = _rows(6)
    serial = evaluate_split(rows, None, episodes=2,
                            policy_factory=RandomPolicyFactory(0),
                            workers=1)
    parallel = evaluate_split(rows, None, episodes=2,
                              policy_factory=RandomPolicyFactory(0),
                              workers=4)
    assert [{k: r[k] for k in COMPARED} for r in serial] == \
        [{k: r[k] for k in COMPARED} for r in parallel]


def test_results_follow_submission_order():
    rows = _rows(5)
    records = evaluate_split(rows, None, episodes=1,
                             policy_factory=RandomPolicyFactory(0),
                             workers=4)
    assert [r["instance"] for r in records] == \
        [f"{row['unit']}@{row['seed']}" for row in rows]


def test_map_instances_handles_degenerate_input():
    assert map_instances(str, [], workers=4) == []
    assert map_instances(str, [1], workers=4) == ["1"]
    assert map_instances(str, [1, 2], workers=1) == ["1", "2"]
    assert default_workers() >= 1


@pytest.mark.parametrize("workers", [1, 3])
def test_policy_factory_is_optional(workers):
    """Without a factory the run stays serial rather than failing: a
    policy wrapping a torch module cannot cross a process boundary."""
    from topogym.baselines.gridworld2dv1.random_walk import RandomBaseline

    records = evaluate_split(_rows(2), RandomBaseline().policy(),
                             episodes=1, workers=workers)
    assert len(records) == 2
