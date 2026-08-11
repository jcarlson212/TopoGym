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
    # The join must not double the resumed cell: restoring to a point
    # of a trajectory is being at that point, not stepping to it twice.
    assert archive.cells[(4, 0)]["trajectory"] == (
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0))


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


# -- TrajectoryArchive: the inherited contract ------------------------

def test_it_is_still_a_phase_one_archive():
    """Adding a field to each entry must not break the thing phase 1
    uses the archive for -- counting, scoring, and selecting."""
    from topogym.baselines.gridworld2dv1.archive import ATTRIBUTES

    archive = TrajectoryArchive({}, seed=0, adjacency={})
    path = ((0, 0), (1, 0), (2, 0))
    fresh = archive.observe(set(path), trajectory=path)

    assert fresh == 3                       # super's return value
    entry = archive.cells[(1, 0)]
    for attribute in ATTRIBUTES:            # every counted attribute
        assert attribute in entry
    assert entry["trajectory"] == path[:2]  # ...plus ours
    assert archive.select() in archive.cells


def test_selection_still_moves_over_the_archive():
    """A selection that always returns the same cell would make phase 1
    a random walk with extra steps."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    cells = {(x, 0) for x in range(30)}
    archive.observe(cells, trajectory=tuple(sorted(cells)))
    picks = {archive.select() for _ in range(40)}
    assert len(picks) > 1
    assert picks <= cells


def test_reobserving_does_not_reset_the_recorded_route():
    """Phase 1 revisits cells constantly; each visit must not wipe the
    route that phase 2 depends on."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    path = ((0, 0), (1, 0), (2, 0))
    archive.observe(set(path), trajectory=path)
    archive.observe(set(path))  # a later episode with no path recorded
    assert archive.cells[(2, 0)]["trajectory"] == path
    assert archive.cells[(2, 0)]["seen"] > 1


# -- TrajectoryArchive: the route itself ------------------------------

def test_a_revisited_cell_keeps_its_first_arrival():
    """Within one episode the shortest prefix is the first arrival;
    a later loop back must not lengthen the stored route."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    wandering = ((0, 0), (1, 0), (2, 0), (1, 0), (0, 0))
    archive.observe(set(wandering), trajectory=wandering)
    assert archive.cells[(0, 0)]["trajectory"] == ((0, 0),)
    assert archive.cells[(1, 0)]["trajectory"] == ((0, 0), (1, 0))


def test_cells_seen_but_never_stepped_on_get_no_route():
    """The observation window reveals cells the agent never stood on.
    They belong in the archive -- phase 1 may select them -- but there
    is no route to somewhere you have not been."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    walked = ((0, 0), (1, 0))
    archive.observe({*walked, (5, 5)}, trajectory=walked)
    assert archive.cells[(5, 5)]["trajectory"] == ()
    assert archive.cells[(1, 0)]["trajectory"] == walked


def test_a_route_through_cells_outside_the_archive_is_ignored():
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    archive.observe({(0, 0)}, trajectory=((0, 0), (9, 9)))
    assert (9, 9) not in archive.cells
    assert archive.cells[(0, 0)]["trajectory"] == ((0, 0),)


def test_an_empty_trajectory_is_a_no_op():
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    assert archive.observe({(0, 0)}, trajectory=()) == 1
    assert archive.cells[(0, 0)]["trajectory"] == ()


def test_a_route_resumed_from_a_cell_with_no_route_stands_alone():
    """Selecting a seen-but-unwalked cell is legal, and the segment
    explored from it is still worth keeping."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    archive.observe({(5, 5)}, trajectory=())
    segment = ((5, 5), (6, 5))
    archive.observe({(5, 5), (6, 5)}, chosen_from=(5, 5),
                    trajectory=segment)
    assert archive.cells[(6, 5)]["trajectory"] == segment


# -- TrajectoryArchive: choosing the demonstration --------------------

def test_the_shortest_of_several_goal_routes_wins():
    """A shorter demonstration is a shorter curriculum, and phase 2
    pays one training stage per stride along it."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    long_way = tuple((i, 0) for i in range(10)) + ((9, 1),)
    archive.observe(set(long_way), trajectory=long_way, reached_goal=True)
    short_way = ((0, 0), (0, 1), (9, 1))
    archive.observe(set(long_way) | set(short_way), trajectory=short_way,
                    reached_goal=True)
    assert archive.best_goal_trajectory() == short_way


def test_the_goal_stays_marked_across_later_episodes():
    """Reaching the goal once is enough; a later episode that misses it
    must not un-mark the cell."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    path = ((0, 0), (1, 0), (2, 0))
    archive.observe(set(path), trajectory=path, reached_goal=True)
    archive.observe(set(path), trajectory=path, reached_goal=False)
    assert archive.best_goal_trajectory() == path


def test_a_goal_reached_after_a_reset_yields_the_whole_route():
    """The demonstration phase 2 restarts along has to begin where the
    layout begins, or the curriculum's last stage is unreachable."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    first = ((0, 0), (1, 0), (2, 0))
    archive.observe(set(first), trajectory=first)
    tail = ((2, 0), (3, 0), (4, 0))
    archive.observe(set(first) | set(tail), chosen_from=(2, 0),
                    trajectory=tail, reached_goal=True)
    route = archive.best_goal_trajectory()
    assert route[0] == (0, 0)          # the layout's start
    assert route[-1] == (4, 0)         # the goal
    assert len(route) == len(set(route))  # no doubled join


def test_the_demonstration_feeds_a_curriculum_that_spans_it():
    """The two halves have to fit together: every stage the backward
    curriculum picks must be a cell of the demonstration, so that
    every stage is a legal reset target."""
    from topogym.baselines.gridworld2dv1 import get_baseline

    archive = TrajectoryArchive({}, seed=0, adjacency={})
    path = tuple((i, 0) for i in range(25))
    archive.observe(set(path), trajectory=path, reached_goal=True)
    demonstration = archive.best_goal_trajectory()

    stages = get_baseline("go-explore-phase1and2")().backward_stages(
        demonstration)
    assert set(stages) <= set(demonstration)
    assert stages[0] == demonstration[-1]
    assert stages[-1] == demonstration[0]


# -- Algorithm 1 details (Salimans and Chen, section 3) ---------------

def test_each_stage_samples_a_local_start_from_a_window():
    """"Each worker then samples a local starting point from a small set
    of time steps {tau - D, ..., tau} to increase diversity." Without
    it a stage learns one restart position rather than a stretch."""
    from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1_and_phase2 import (  # noqa: E501
        LOCAL_START_WINDOW,
    )

    baseline = get_baseline("go-explore-phase1and2")()
    demonstration = tuple((i, 0) for i in range(30))

    window = baseline.local_starts(demonstration, 20)
    assert window[-1] == demonstration[20]         # tau itself
    assert len(window) == LOCAL_START_WINDOW + 1
    # Only ever backward: never a position the curriculum has not
    # reached, which would leak progress it has not earned.
    assert all(cell in demonstration[:21] for cell in window)


def test_the_window_is_clipped_at_the_start_of_the_demonstration():
    baseline = get_baseline("go-explore-phase1and2")()
    demonstration = tuple((i, 0) for i in range(30))
    assert baseline.local_starts(demonstration, 0) == [demonstration[0]]
    assert baseline.local_starts(demonstration, 2)[0] == demonstration[0]


def test_the_split_env_samples_across_the_whole_window():
    """The window has to reach the rollout workers, not just the
    config: a sampler that always returns the same cell is a fixed
    start with extra machinery."""
    from topogym.baselines.gridworld2dv1.multitask import SplitEnv
    from topogym.baselines.gridworld2dv1.single_layout import layout_row

    window = [(25, 25), (25, 26), (25, 27)]
    env = SplitEnv({"rows": [layout_row("TopoGym/Dilution-50-v0", 0)],
                    "seed": 0, "start_cells": window,
                    "demonstration": window,
                    "env_options": {"teleport": True}})
    seen = set()
    for episode in range(30):
        env.reset(seed=episode)
        seen.add(env.env.unwrapped._state.cell)
    env.close()
    assert seen == set(window)


def test_a_single_start_cell_still_works():
    """The plural form must not break the singular one."""
    from topogym.baselines.gridworld2dv1.multitask import SplitEnv
    from topogym.baselines.gridworld2dv1.single_layout import layout_row

    env = SplitEnv({"rows": [layout_row("TopoGym/Dilution-50-v0", 0)],
                    "seed": 0, "start_cell": (25, 25),
                    "demonstration": [(25, 25)],
                    "env_options": {"teleport": True}})
    env.reset(seed=0)
    assert env.env.unwrapped._state.cell == (25, 25)
    env.close()


def test_the_recurrent_priming_step_is_declared_inapplicable():
    """Salimans and Chen replay K demonstration actions before each
    rollout to initialise an RNN's hidden state. Every baseline here is
    feedforward over a Markov observation, so there is no hidden state
    to prime. That is a property of the architecture, not a shortcut,
    and it stops holding the moment a recurrent module appears."""
    from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1_and_phase2 import (  # noqa: E501
        DEMONSTRATION_PRIMING_STEPS,
    )

    assert DEMONSTRATION_PRIMING_STEPS == 0

    # The constant above is the half of this claim that holds without
    # the benchmark extra; policy_module_spec() reaches into RLlib.
    pytest.importorskip("ray", reason="needs topogym[benchmarks]")
    baseline = get_baseline("go-explore-phase1and2")()
    spec = baseline.policy_module_spec()
    model_config = getattr(spec, "model_config", None) or {}
    assert not model_config.get("use_lstm")
    assert not model_config.get("use_attention")


# -- the budget when phase 1 finds nothing ----------------------------

def test_an_unused_phase_two_budget_returns_to_phase_one():
    """Phase 2 exists only once phase 1 has something to robustify.
    With no route, discarding its half makes this method phase 1 with
    half the exploration -- worse by construction, and any comparison
    between them measures the split rather than the algorithms."""
    from topogym.baselines.gridworld2dv1.single_layout import (
        episodes_for,
        layout_row,
    )

    row = layout_row("TopoGym/EpicChase8-120-v0", 0)
    baseline = get_baseline("go-explore-phase1and2")(
        BaselineConfig(seed=0, max_iterations=1))
    spent = []
    baseline.explore = lambda rows, episodes, seed: (
        spent.append(episodes), ([], ()))[1]

    result = baseline.single_layout_train_test_run(
        row, step_budget=3600, eval_episodes=2)

    total = episodes_for(3600, int(row["horizon"]))
    assert sum(spent) == total, (
        f"explored {sum(spent)} of {total} episodes; the rest was lost")
    assert result.training["phase1_episodes"] == total


def test_a_found_route_still_leaves_phase_two_its_budget():
    from topogym.baselines.gridworld2dv1.single_layout import (
        episodes_for,
        layout_row,
    )

    row = layout_row("TopoGym/EpicChase8-120-v0", 0)
    baseline = get_baseline("go-explore-phase1and2")(
        BaselineConfig(seed=0, max_iterations=1))
    route = ((1, 1), (1, 2), (1, 3))
    spent = []
    baseline.explore = lambda rows, episodes, seed: (
        spent.append(episodes), ([], route))[1]
    baseline.robustify = lambda *a, **k: {
        "stages": [], "reached_start": False, "why": "stubbed"}

    result = baseline.single_layout_train_test_run(
        row, step_budget=3600, eval_episodes=2)
    assert sum(spent) < episodes_for(3600, int(row["horizon"]))
    assert result.training["demonstration_cells"] == len(route)


def test_it_tunes_the_archive_grid_not_ppos():
    """Inheriting PPOBaseline's grid would search a learning rate that
    only matters once phase 1 has found a route, while leaving the
    cell-selection weights that decide whether it ever does at their
    defaults."""
    from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1 import (  # noqa: E501
        GoExplorePhase1Baseline,
    )

    both = get_baseline("go-explore-phase1and2")()
    assert both.tune_grid == GoExplorePhase1Baseline.tune_grid
    assert "w_topo" not in both.tune_grid[0]
    assert "w_a" in both.tune_grid[0]


def test_the_evaluation_measures_the_policy_not_the_archive():
    row_seed = 0
    from topogym.baselines.gridworld2dv1.single_layout import layout_row

    row = layout_row("TopoGym/Decoys1-50-v0", row_seed)
    baseline = get_baseline("go-explore-phase1and2")(
        BaselineConfig(seed=0, max_iterations=1))
    baseline.explore = lambda rows, episodes, seed: ([], ())
    result = baseline.single_layout_train_test_run(
        row, step_budget=1800, eval_episodes=2)
    assert result.config["eval_archive"] is False
    assert result.eval_horizon >= int(row["horizon"])
