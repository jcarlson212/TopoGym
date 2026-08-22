"""Phase 2's inputs, pinned.

Every test here is the negation of a bug that shipped: demonstrations
that did not root at the layout start (chunk-boundary stitching),
demonstrations that stopped one cell short of the goal (pre-step
recording), a curriculum whose stages relearned from scratch (no
weight threading). Subclass-archive wiring is pinned beside the
subclass it belongs to.
"""

import numpy as np
import pytest

from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1_and_phase2 import (
    DEFAULTS,
    GoExplorePhase12Baseline,
    _Session,
)
from topogym.baselines.gridworld2dv1.instances import load_split, make_instance
from topogym.baselines.gridworld2dv1.protocol import BaselineConfig
from topogym.stats import StatsRecorder


@pytest.fixture(autouse=True)
def _fresh_sessions():
    """The session registry is process-global by design (one study per
    process in production); tests share a process, so each gets a
    clean registry or they bleed archives into one another."""
    from topogym.baselines.gridworld2dv1.concrete_baselines import (
        goexplore_phase1_and_phase2 as ge12,
    )

    ge12._SESSIONS.clear()
    yield
    ge12._SESSIONS.clear()


def _row(unit="Decoys0-50", seed=4000):
    return [r for r in load_split("test")
            if r["unit"] == unit and int(r["seed"]) == seed][0]


def _walk(session, env, steps):
    obs, info = env.reset(seed=0)
    core = env.unwrapped
    for _ in range(steps):
        action = session.act(obs, core)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


# -- chunk boundaries --------------------------------------------------

def test_chunked_exploration_roots_every_route_at_the_start():
    """Two explore() chunks over one live world: every route the
    archive holds must begin at the layout start. Before end_chunk,
    the second chunk's first episode -- which really does start at the
    layout start -- inherited the previous chunk's stale teleport
    prefix, and 'demonstrations' were born that no agent ever ran."""
    row = _row()
    baseline = GoExplorePhase12Baseline(BaselineConfig(seed=0))
    baseline._archive_params = dict(DEFAULTS)
    env = StatsRecorder(make_instance(row, **baseline.env_options()))
    baseline.bind_env(env)
    try:
        for chunk_seed in (0, 7, 21):
            baseline.explore([row], 8, chunk_seed)
        session = baseline.phase1_probe()
        start = tuple(env.unwrapped.layout.start)
        routes = [entry["trajectory"]
                  for entry in session.archive.cells.values()
                  if entry.get("trajectory")]
        assert routes, "exploration stored no routes at all"
        assert all(tuple(route[0]) == start for route in routes)
    finally:
        env.close()


def test_end_chunk_flushes_the_dangling_episode_and_clears_state():
    row = _row()
    baseline = GoExplorePhase12Baseline(BaselineConfig(seed=0))
    baseline._archive_params = dict(DEFAULTS)
    env = StatsRecorder(make_instance(row, **baseline.env_options()))
    baseline.bind_env(env)
    try:
        baseline.explore([row], 3, 0)
        session = baseline.phase1_probe()
        # evaluate_split fires the probe before every episode but its
        # first, so the last episode is still dangling here.
        assert session.trajectory, "expected a dangling final episode"
        dangling = tuple(map(tuple, session.trajectory))
        session.end_chunk(env)
        assert session.trajectory == []
        assert session.chosen_from is None
        stored = session.archive.cells[dangling[-1]]["trajectory"]
        assert tuple(stored[-1]) == dangling[-1]
    finally:
        env.close()


# -- the goal cell -----------------------------------------------------

def test_a_goal_reaching_route_ends_on_the_goal_cell():
    """act() records the cell it stands on before each action, so the
    step onto the goal was never recorded and every demonstration
    ended one cell short of the reward phase 2 exists to reach."""
    row = _row()
    session = _Session(dict(DEFAULTS), seed=0)
    env = StatsRecorder(make_instance(row))
    try:
        _walk(session, env, 40)
        core = env.unwrapped
        # Stand the agent on the goal, as a terminated episode does,
        # and let the boundary probe see a goal_reached info.
        core._state = core.layout.base.initial_state(core.layout.goal)
        core._visited.add(tuple(core.layout.goal))  # as a real step does
        session.choose_reset(core, {"goal_reached": True})
        goal = tuple(core.layout.goal)
        routes = [entry["trajectory"]
                  for entry in session.archive.cells.values()
                  if entry.get("reached_goal")]
        assert routes and tuple(routes[0][-1]) == goal
        assert tuple(session.archive.best_goal_trajectory()[-1]) == goal
    finally:
        env.close()


# -- weight threading --------------------------------------------------

@pytest.mark.slow
def test_curriculum_stages_continue_the_same_weights():
    """Stage N+1 must start from stage N's weights: the Backward
    Algorithm fine-tunes one policy, and a stage rebuilt from scratch
    relearns the whole tail of the route inside a budget sized for
    one increment."""
    pytest.importorskip("ray")
    row = _row()
    baseline = GoExplorePhase12Baseline(BaselineConfig(
        seed=0, num_env_runners=1, num_envs_per_runner=1,
        train_batch_size=400))
    values = baseline.default_hyperparameters()

    def tensors(state):
        out = {}
        def collect(node, path):
            if isinstance(node, dict):
                for key, value in node.items():
                    collect(value, f"{path}/{key}")
            elif isinstance(node, np.ndarray):
                out[path] = node
            elif hasattr(node, "detach"):
                out[path] = node.detach().cpu().numpy()
        collect(state, "")
        return out

    first = baseline.algorithm_config([row], values, 0).build_algo()
    try:
        first.train()
        carried = first.learner_group.get_state()
        reference = tensors(carried)
        assert reference, "no tensors found in learner state"
        second = baseline.algorithm_config([row], values, 1).build_algo()
        try:
            fresh = tensors(second.learner_group.get_state())
            assert any(not np.array_equal(fresh[k], reference[k])
                       for k in reference if k in fresh), \
                "fresh build unexpectedly matches trained weights"
            second.learner_group.set_state(carried)
            second.env_runner_group.sync_weights(
                from_worker_or_learner_group=second.learner_group)
            restored = tensors(second.learner_group.get_state())
            for key, value in reference.items():
                assert key in restored
                assert np.array_equal(restored[key], value), key
        finally:
            second.stop()
    finally:
        first.stop()


def test_archive_values_filters_to_declared_knobs_only():
    """The filter keeps what the *baseline* declares, not what GE's
    defaults happen to contain: an optimizer key is dropped, GE's own
    selection weights pass, and a subclass widens the vocabulary by
    widening default_hyperparameters()."""
    baseline = GoExplorePhase12Baseline(BaselineConfig(seed=0))
    merged = baseline.archive_values({"w_a": 3.0, "p_a": 1.0, "lr": 9.0})
    assert merged["w_a"] == 3.0 and merged["p_a"] == 1.0
    assert "lr" not in merged
