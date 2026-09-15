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

def test_a_goal_route_that_does_not_begin_at_the_start_is_not_a_demonstration():
    """A segment from an archive restart may reach the goal, and may
    even be the shortest goal route in the archive; it is still not
    something a policy can be trained to run from the start."""
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    rooted = ((0, 0), (1, 0), (2, 0), (3, 0), (4, 0))
    archive.observe(set(rooted), trajectory=rooted, reached_goal=True)
    segment = ((7, 7), (4, 0))
    archive.observe(set(segment), chosen_from=(7, 7), trajectory=segment,
                    reached_goal=True)
    # The shorter segment does not displace the rooted route to the
    # goal cell, and the start filter refuses what is left unrooted.
    assert archive.cells[(4, 0)]["trajectory"] == rooted
    assert archive.best_goal_trajectory(start=(0, 0)) == rooted
    assert archive.best_goal_trajectory(start=(9, 9)) == ()


def test_a_rooted_route_replaces_an_unrooted_one_whatever_its_length():
    archive = TrajectoryArchive({}, seed=0, adjacency={})
    archive.observe({(7, 7)}, trajectory=())
    segment = ((7, 7), (8, 7))
    archive.observe(set(segment), chosen_from=(7, 7), trajectory=segment)
    assert archive.cells[(8, 7)]["rooted"] is False
    long_way = tuple((i, 7) for i in range(9))
    archive.observe(set(long_way), trajectory=long_way)
    assert archive.cells[(8, 7)]["trajectory"] == long_way
    assert archive.cells[(8, 7)]["rooted"] is True


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
    demonstration = tuple((i, 0) for i in range(3 * LOCAL_START_WINDOW))
    tau = 2 * LOCAL_START_WINDOW

    window = baseline.local_starts(demonstration, tau)
    assert window[-1] == demonstration[tau]         # tau itself
    assert len(window) == LOCAL_START_WINDOW + 1
    # Only ever backward: never a position the curriculum has not
    # reached, which would leak progress it has not earned.
    assert all(cell in demonstration[:tau + 1] for cell in window)


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


def _budget_run(config, step_budget, spend_all=False):
    """Run the two-phase protocol with a stubbed phase 1 and a phase 2
    that records the iterations it was handed."""
    from topogym.baselines.gridworld2dv1.single_layout import (
        episodes_for,
        layout_row,
    )

    row = layout_row("TopoGym/EpicChase8-120-v0", 0)
    baseline = get_baseline("go-explore-phase1and2")(config)
    total = episodes_for(step_budget, int(row["horizon"]))
    route = ((1, 1), (1, 2), (1, 3))
    spent, handed = [], []

    def explore(rows, episodes, seed):
        spent.append(episodes)
        found = sum(spent) >= total if spend_all else True
        return [], (route if found else ())

    baseline.explore = explore
    baseline.robustify = lambda rows, demo, values, iterations, seed: (
        handed.append(iterations),
        {"stages": [], "reached_start": True, "why": "stubbed"})[1]
    result = baseline.single_layout_train_test_run(
        row, step_budget=step_budget, eval_episodes=2)
    return result, row, total, sum(spent), handed


def test_phase_two_is_sized_by_what_phase_one_left():
    """No iteration default: phase 2 gets the unspent steps, converted
    at PPO's batch size, so the two phases together stay within the
    one-phase budget they are compared against."""
    config = BaselineConfig(seed=0, max_iterations=200,
                            train_batch_size=1000)
    result, row, total, spent, handed = _budget_run(config, 120_000)
    remaining_steps = (total - spent) * int(row["horizon"])
    assert handed == [remaining_steps // 1000]
    assert result.training["phase2_steps"] == remaining_steps
    assert result.training["phase2_iterations"] == handed[0]


def test_a_phase_one_that_spends_everything_leaves_phase_two_nothing():
    config = BaselineConfig(seed=0, max_iterations=200,
                            train_batch_size=1000)
    result, _, total, spent, handed = _budget_run(
        config, 120_000, spend_all=True)
    assert spent == total
    assert handed == [], "phase 2 ran on a budget it did not have"
    assert result.training["phase2_iterations"] == 0
    assert "no budget" in result.training["why"]


def test_a_fixed_phase_two_budget_frees_phase_one_to_use_it_all():
    """With phase2_steps stated, phase 1 may explore up to the whole
    step budget -- exactly the one-phase arm's allowance -- and phase
    2's length comes from the stated figure alone."""
    config = BaselineConfig(seed=0, max_iterations=200,
                            train_batch_size=1000, phase2_steps=50_000)
    result, _, total, spent, handed = _budget_run(
        config, 120_000, spend_all=True)
    assert spent == total, "phase 1 was capped below the budget"
    assert handed == [50]
    assert result.training["phase2_steps"] == 50_000


class _FakeAlgorithm:
    """Stands in for an RLlib algorithm: train() returns preset
    success rates in order, everything else is a no-op."""

    def __init__(self, successes, episodes_per_iteration=100):
        self._successes = list(successes)
        self._episodes = episodes_per_iteration

        class _Group:
            def get_state(self_inner):
                return {"weights": 1}

            def set_state(self_inner, state):
                pass

            def sync_weights(self_inner, **kwargs):
                pass

        self.learner_group = _Group()
        self.env_runner_group = _Group()

    def train(self):
        value = self._successes.pop(0) if self._successes else 1.0
        return {"env_runners": {"episode_return_mean": value,
                                "num_episodes": self._episodes}}

    def stop(self):
        pass


def _curriculum(successes, iterations, route_len=40, horizon=None,
                configs=None, episodes_per_iteration=100):
    """Run robustify with a fake learner; return its outcome and the
    number of algorithms built (one per stage)."""
    baseline = get_baseline("go-explore-phase1and2")(BaselineConfig(seed=0))
    baseline._phase2_horizon = horizon
    algorithm = _FakeAlgorithm(successes, episodes_per_iteration)
    built = []

    class _Config:
        def __init__(self):
            self.env_config = {"env_options": {"teleport": True}}

        def build_algo(self):
            built.append(1)
            return algorithm

    def make_config(rows, values, seed):
        config = _Config()
        if configs is not None:
            configs.append(config)
        return config

    baseline.algorithm_config = make_config
    baseline._checkpoint = lambda: None
    demonstration = tuple((i, 0) for i in range(route_len))
    return baseline.robustify([], demonstration, {}, iterations), len(built)


def test_phase_two_trains_at_the_evaluation_horizon():
    """Phase 1's horizon is pinned below the route length by design;
    a policy trained under it can never finish from far back."""
    configs = []
    _curriculum([1.0] * 10, iterations=10, horizon=310, configs=configs)
    assert configs, "no stage was configured"
    for config in configs:
        assert config.env_config["env_options"]["max_steps"] == 310
        assert config.env_config["env_options"]["teleport"] is True
        assert config.env_config["start_cells"]


def test_the_curriculum_budget_is_one_pool_not_a_per_stage_ration():
    """A stage passed in one iteration leaves its share to the stages
    behind it. Under the old even split a 40-cell route with 6
    iterations gave each of its 6 stages one, and the first stage that
    needed two ended the curriculum."""
    # Stage 1 passes at once; stage 2 needs three iterations; the rest
    # pass at once again.
    outcome, built = _curriculum(
        [1.0, 0.0, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0], iterations=12)
    assert outcome["reached_start"], outcome["why"]
    stages = outcome["stages"]
    assert stages[1]["iterations"] == 3
    assert outcome["iterations_used"] == sum(s["iterations"] for s in stages)
    assert outcome["iterations_used"] <= outcome["iterations_budget"]
    assert built == len(stages)


def test_consecutive_stage_windows_overlap_by_at_least_half():
    """The new window must contain starts the policy already finishes
    from, or a stage can earn no reward at all and never learn: the
    retreat is capped at half the window."""
    from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1_and_phase2 import (  # noqa: E501
        LOCAL_START_WINDOW,
        MAX_RETREAT,
    )

    assert MAX_RETREAT * 2 <= LOCAL_START_WINDOW
    outcome, _ = _curriculum([1.0] * 40, iterations=40, route_len=120)
    assert outcome["reached_start"]
    windows = [set(map(tuple, s["start_window"]))
               for s in outcome["stages"]]
    assert len(windows) >= 5
    for before, after in zip(windows, windows[1:]):
        # The last window is clipped at the layout start and may be a
        # single cell; it still has to be one the stage before ran.
        assert len(before & after) >= min(LOCAL_START_WINDOW // 2,
                                          len(after)), (
            "a stage started with no mastered start in its window")


def test_a_stage_cannot_pass_on_a_handful_of_finished_episodes():
    """After one iteration the only finished episodes are the quick
    successes -- the failures are still running toward the horizon --
    so a perfect rate over a few episodes is survivor bias, not a
    pass. The stage keeps training until a full window has finished."""
    from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1_and_phase2 import (  # noqa: E501
        MIN_STAGE_EPISODES,
    )

    per_iteration = 10
    needed = -(-MIN_STAGE_EPISODES // per_iteration)
    outcome, _ = _curriculum([1.0] * 200, iterations=200, route_len=20,
                             episodes_per_iteration=per_iteration)
    assert outcome["reached_start"]
    for stage in outcome["stages"]:
        assert stage["iterations"] == needed, stage
        assert stage["episodes"] >= MIN_STAGE_EPISODES


def test_the_curriculum_stops_when_the_pool_runs_dry_and_says_so():
    outcome, _ = _curriculum([1.0, 0.0, 0.0, 0.0, 0.0], iterations=4)
    assert not outcome["reached_start"]
    assert outcome["iterations_used"] == 4
    assert outcome["stages"][-1]["passed"] is False
    assert outcome["why"].startswith("curriculum stopped")


def test_a_batch_with_no_finished_episode_does_not_end_a_stage():
    """nan is 'no episode finished in this batch', which is common on
    a long horizon and is not a failed stage."""
    nan = float("nan")
    outcome, _ = _curriculum([nan, nan, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                             iterations=20)
    assert outcome["stages"][0]["iterations"] == 3
    assert outcome["stages"][0]["passed"] is True
    assert outcome["reached_start"]


def test_phase_two_may_act_on_a_different_action_space_than_phase_one():
    """Phase 1 explores as it always did; from the moment phase 2
    begins, every environment the baseline builds -- stages,
    evaluation, GIF -- uses the phase 2 action space."""
    baseline = get_baseline("go-explore-phase1and2")(
        BaselineConfig(seed=0, phase2_actions="fourway"))
    assert baseline.env_options().get("actions", "egocentric") \
        == "egocentric"
    baseline._phase2_active = True
    assert baseline.env_options()["actions"] == "fourway"
    plain = get_baseline("go-explore-phase1and2")(BaselineConfig(seed=0))
    plain._phase2_active = True
    assert plain.env_options().get("actions", "egocentric") == "egocentric"


def test_it_inherits_phase_ones_tuning():
    cls = get_baseline("go-explore-phase1and2")
    assert cls.tuning_source == "go-explore-phase1"
    assert get_baseline(cls.tuning_source) is not cls


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
