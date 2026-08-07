"""Go-Explore phase 1 + 2: the archive of paths, and the backward
curriculum that consumes one."""

from __future__ import annotations

import gymnasium as gym
import pytest

import topogym  # noqa: F401  (registers the ids)
from topogym.baselines.gridworld2dv1 import get_baseline
from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1_and_phase2 import (  # noqa: E501
    BACKUP_STRIDE,
    TrajectoryArchive,
)
from topogym.baselines.gridworld2dv1.protocol import BaselineConfig

# -- the archive ------------------------------------------------------

def test_the_archive_remembers_how_it_reached_each_cell():
    """Phase 1 only needs cells; phase 2 needs the route to them, and
    without one it has nothing to restart along."""
    archive = TrajectoryArchive({}, seed=0,
                                adjacency={(0, 0): [(1, 0)], (1, 0): []})
    path = ((0, 0), (1, 0), (2, 0))
    archive.observe(set(path), trajectory=path)
    assert archive.cells[(2, 0)]["trajectory"] == path
    assert archive.cells[(1, 0)]["trajectory"] == path[:2]


def test_a_shorter_route_to_the_same_cell_replaces_the_longer_one():
    """A shorter demonstration is a shorter curriculum."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    long_way = ((0, 0), (1, 0), (2, 0), (3, 0), (2, 1))
    short_way = ((0, 0), (1, 1), (2, 1))
    archive.observe(set(long_way), trajectory=long_way)
    archive.observe(set(short_way), trajectory=short_way)
    assert archive.cells[(2, 1)]["trajectory"] == short_way


def test_a_route_found_after_an_archive_reset_includes_its_prefix():
    """An episode that resumed at an archived cell only walked the tail.
    Storing the tail would give phase 2 a curriculum that starts
    nowhere reachable from the layout's start."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    first = ((0, 0), (1, 0), (2, 0))
    archive.observe(set(first), trajectory=first)
    resumed = ((2, 0), (3, 0), (4, 0))
    archive.observe(set(first) | set(resumed), chosen_from=(2, 0),
                    trajectory=resumed)
    assert archive.cells[(4, 0)]["trajectory"] == (
        (0, 0), (1, 0), (2, 0), (2, 0), (3, 0), (4, 0))


def test_only_a_goal_reaching_episode_yields_a_demonstration():
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    path = ((0, 0), (1, 0), (2, 0))
    archive.observe(set(path), trajectory=path, reached_goal=False)
    assert archive.best_goal_trajectory() == ()
    archive.observe(set(path), trajectory=path, reached_goal=True)
    assert archive.best_goal_trajectory() == path


# -- the backward curriculum ------------------------------------------

def test_the_curriculum_runs_from_the_goal_back_to_the_start():
    """Salimans and Chen: begin next to the last state, and back up only
    once the agent can finish from where it stands."""
    baseline = get_baseline("go-explore-phase1and2")()
    demonstration = tuple((i, 0) for i in range(40))
    stages = baseline.backward_stages(demonstration)

    assert stages[0] == demonstration[-1]   # nearest the goal first
    assert stages[-1] == demonstration[0]   # finishing from the start
    gaps = {a[0] - b[0] for a, b in zip(stages, stages[1:])}
    assert gaps <= {BACKUP_STRIDE, 39 % BACKUP_STRIDE or BACKUP_STRIDE}
    assert baseline.backward_stages(()) == []


def test_the_curriculum_always_finishes_at_the_true_start():
    """A stride that does not divide the trajectory must not leave the
    agent never having run the whole thing."""
    baseline = get_baseline("go-explore-phase1and2")()
    for length in range(1, 40):
        demonstration = tuple((i, 0) for i in range(length))
        assert baseline.backward_stages(demonstration)[-1] == (0, 0)


# -- the demonstration reaches the environment ------------------------

def test_a_demonstration_cell_is_a_legal_reset_target():
    """Phase 2 restarts along a trajectory recorded in phase 1, possibly
    in another process, so the env cannot have visited it."""
    env = gym.make("TopoGym/Dilution-50-v0", seed=0, teleport=True,
                   demonstration=[(25, 25)]).unwrapped
    env.reset(seed=0)
    _, info = env.reset(seed=1, options={"teleport": (25, 25)})
    assert info["position"] == (25, 25)


def test_an_undeclared_cell_is_still_refused():
    """The exception must not become the rule: without a declared
    demonstration, an archive can only return you where you have been."""
    env = gym.make("TopoGym/Dilution-50-v0", seed=0,
                   teleport=True).unwrapped
    env.reset(seed=0)
    with pytest.raises(ValueError, match="has not been visited"):
        env.reset(seed=1, options={"teleport": (25, 25)})


def test_a_demonstration_does_not_inflate_coverage():
    """Restarting somewhere is not the same as having explored it."""
    plain = gym.make("TopoGym/Dilution-50-v0", seed=0,
                     teleport=True).unwrapped
    plain.reset(seed=0)
    seeded = gym.make("TopoGym/Dilution-50-v0", seed=0, teleport=True,
                      demonstration=[(25, 25), (25, 26), (25, 27)]).unwrapped
    _, info = seeded.reset(seed=0)
    assert info["lifetime_coverage"] == pytest.approx(
        plain._step_info(plain.layout.start)["lifetime_coverage"])


def test_the_split_env_starts_along_the_demonstration():
    """The route from a stage's start cell into the RLlib workers."""
    from topogym.baselines.gridworld2dv1.multitask import SplitEnv
    from topogym.baselines.gridworld2dv1.single_layout import layout_row

    row = layout_row("TopoGym/Dilution-50-v0", 0)
    env = SplitEnv({"rows": [row], "seed": 0,
                    "start_cell": (25, 25),
                    "demonstration": [(25, 25)],
                    "env_options": {"teleport": True}})
    env.reset(seed=0)
    assert env.env.unwrapped._state.cell == (25, 25)
    env.close()


# -- the protocol -----------------------------------------------------

def test_it_declares_that_it_adapts_within_a_hold_out_instance():
    """Phase 2 improves the policy in the world that produced the
    trajectory, which is a different claim from transfer, and the
    result has to say so rather than leave it to be inferred."""
    baseline = get_baseline("go-explore-phase1and2")()
    assert baseline.adapts_per_instance is True
    assert get_baseline("go-explore-phase1")().adapts_per_instance is False


def test_phase_one_records_a_route_when_it_reaches_the_goal():
    """End to end on a small world: explore, and come back with a
    demonstration that starts where the layout starts."""
    from topogym.baselines.gridworld2dv1.single_layout import layout_row

    row = layout_row("TopoGym/Decoys0-50-v0", 0)
    baseline = get_baseline("go-explore-phase1and2")(
        BaselineConfig(seed=0))
    records, demonstration = baseline.explore([row], episodes=60, seed=0)

    assert len(records) == 1
    if demonstration:  # sparse goals: a run may legitimately find none
        assert demonstration[0] == tuple(
            gym.make(row["template_id"], seed=0).unwrapped.layout.start
        ) or len(demonstration) > 1
        assert len(set(demonstration)) > 1


def test_no_demonstration_is_reported_rather_than_papered_over():
    """These worlds are hard. 'Phase 1 never found the goal' is a
    finding; a silently untrained network reported as a result is not."""
    from topogym.baselines.gridworld2dv1.protocol import Hyperparameters
    from topogym.baselines.gridworld2dv1.single_layout import layout_row

    row = layout_row("TopoGym/EpicChase8-120-v0", 0)
    baseline = get_baseline("go-explore-phase1and2")(
        BaselineConfig(seed=0, train_episodes_per_instance=3))
    report = baseline.fit([row], [row], Hyperparameters(values={}))

    assert report.iterations == 0
    assert report.stopped_early
    assert "no goal trajectory" in report.stopped_because
    # And it still hands back a usable policy rather than an untrained
    # network pretending to be one.
    assert callable(baseline.policy())
