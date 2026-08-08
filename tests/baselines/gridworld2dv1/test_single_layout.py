"""Single-layout studies: one world, one long interaction budget."""

from __future__ import annotations

import pytest

from topogym.baselines.gridworld2dv1 import get_baseline
from topogym.baselines.gridworld2dv1.protocol import BaselineConfig
from topogym.baselines.gridworld2dv1.single_layout import (
    episodes_for,
    layout_row,
    run_single_layout,
)


def test_the_budget_is_steps_not_episodes():
    """Layout horizons here span 130 to 6760 steps. Equal episode
    counts would hand some methods fifty times the experience of
    others, which would make the comparison meaningless."""
    assert episodes_for(1_000_000, 180) == 5555
    assert episodes_for(1_000_000, 6760) == 147
    assert episodes_for(100, 6760) == 1  # never zero
    assert episodes_for(1_000_000, 0) == 1_000_000  # no division by zero


def test_a_row_can_be_built_for_any_registry_id():
    """Swapping the world under study must be an argument, not a code
    change -- including for families no split carries."""
    row = layout_row("TopoGym/EpicChase8-120-v0", 0)
    assert row["template_id"] == "TopoGym/EpicChase8-120-v0"
    assert row["unit"] == "EpicChase8-120"
    assert row["family"] == "EpicChase"
    assert int(row["horizon"]) == 180
    # The goal is several episodes away: that is the family's premise.
    assert int(row["optimal_actions"]) > 4 * int(row["horizon"])


@pytest.mark.parametrize("env_id", [
    "TopoGym/TopRP2-50-v0",        # Top slice
    "TopoGym/ClownChase-v0",       # Texture slice
    "TopoGym/Nested3-50-v0",       # GridWorld2D slice
])
def test_rows_carry_the_certified_metadata_for_every_slice(env_id):
    row = layout_row(env_id, 0)
    assert row["betti_z2"].count(" ") == 2  # three Betti numbers
    assert int(row["n_free_cells"]) > 0
    assert row["slice"] in ("Top", "Texture", "GridWorld2D")


def test_a_study_evaluates_frozen_after_learning(tmp_path):
    """The protocol: learn for a step budget, then evaluate with
    learning off. The evaluation is the headline, and it must not be
    the tail of the training run."""
    row = layout_row("TopoGym/TopRP2-50-v0", 0)
    baseline = get_baseline("random")(BaselineConfig(seed=0))
    result = run_single_layout(baseline, row, step_budget=2000,
                               eval_episodes=4,
                               telemetry_root=str(tmp_path))
    assert result.train_episodes == episodes_for(2000, result.horizon)
    assert result.eval_episodes == 4
    assert result.evaluation["episodes"] == 4
    assert result.layout == "TopRP2-50"
    assert result.wall_seconds > 0


def test_every_baseline_answers_the_single_layout_protocol():
    """It is a protocol method, not a script feature: a method that
    does not override it still runs."""
    from topogym.baselines.gridworld2dv1 import BASELINES

    for name in BASELINES:
        baseline = get_baseline(name)()
        assert callable(baseline.single_layout_train_test_run)
        assert isinstance(baseline.default_hyperparameters(), dict)


def test_hyperparameters_carry_over_rather_than_defaulting(tmp_path):
    """A single layout cannot supply a hold-out to tune against, so
    values come from outside it -- which is leak-free only because
    they were chosen on tune/train/val, never here."""
    row = layout_row("TopoGym/TopRP2-50-v0", 0)
    baseline = get_baseline("random")(BaselineConfig(seed=0))
    carried = {"w_a": 7.0}
    result = run_single_layout(baseline, row, step_budget=500,
                               eval_episodes=1, hyperparameters=carried)
    assert result.hyperparameters["values"] == carried
    assert result.hyperparameters["tuning_score"] is None
    assert result.hyperparameters["searched"] == []


def test_the_three_telemetry_tables_agree_on_the_study(tmp_path):
    """Three tables that cannot be joined are three problems."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    row = layout_row("TopoGym/TopRP2-50-v0", 0)
    baseline = get_baseline("go-explore-phase1")(BaselineConfig(seed=0))
    result = run_single_layout(baseline, row, step_budget=1000,
                               eval_episodes=5,
                               telemetry_root=str(tmp_path))
    steps = pd.read_parquet(tmp_path / "steps")
    episodes = pd.read_parquet(tmp_path / "episodes")
    instances = pd.read_parquet(tmp_path / "instances")

    # Training is recorded as its own phase, so the evaluation half has
    # to be selected rather than assumed to be the whole file.
    assert {"single-eval", "single-train"} >= set(steps["split"])
    assert "single-eval" in set(instances["split"])
    eval_steps = steps[steps["split"] == "single-eval"]
    eval_eps = episodes[episodes["split"] == "single-eval"]
    assert len(eval_eps) == 5
    assert len(eval_steps) == eval_eps["length"].sum()
    assert len(eval_steps) == result.evaluation["interactions"]


def test_go_explore_takes_the_archive_reset_it_is_offered(tmp_path):
    """The harness offers every method the same probe at every episode
    boundary; an archive method has to actually use it, or the
    single-layout protocol is measuring nothing it was built for."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    row = layout_row("TopoGym/TopRP2-50-v0", 0)
    baseline = get_baseline("go-explore-phase1")(BaselineConfig(seed=0))
    run_single_layout(baseline, row, step_budget=1000, eval_episodes=6,
                      telemetry_root=str(tmp_path), eval_archive=True)
    episodes = (pd.read_parquet(tmp_path / "episodes")
                .query("split == 'single-eval'").sort_values("episode"))
    # Never on the first episode -- there is no archive yet -- and
    # every time after.
    assert not episodes["archive_reset"].iloc[0]
    assert episodes["archive_reset"].iloc[1:].all()
    assert episodes["reset_cell"].iloc[1:].notna().all()


def test_a_random_walk_does_not_take_one(tmp_path):
    """The contrast that makes the previous test meaningful."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    row = layout_row("TopoGym/TopRP2-50-v0", 0)
    baseline = get_baseline("random")(BaselineConfig(seed=0))
    run_single_layout(baseline, row, step_budget=1000, eval_episodes=4,
                      telemetry_root=str(tmp_path))
    episodes = pd.read_parquet(tmp_path / "episodes")
    assert not episodes["archive_reset"].any()


def test_the_script_roster_resolves_to_real_layouts():
    """A roster naming an id that does not exist fails at hour nine of
    a cloud run rather than at import."""
    import pathlib
    import sys

    import gymnasium as gym

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]
                           / "scripts" / "benchmarks"))
    from run_single_layout import ROSTER

    for env_id in ROSTER:
        assert env_id in gym.registry, env_id
    # The spiral is in the roster and in no benchmark: that pairing is
    # the point of having a single-layout harness at all.
    from topogym import benchmarks

    assert any(benchmarks.is_standalone(e.split("/")[-1].removesuffix("-v0"))
               for e in ROSTER)


# -- the budget has to bind, not merely be announced ------------------

def test_the_step_budget_caps_a_method_counted_in_iterations():
    """RLlib trains in iterations of train_batch_size steps, so a
    100k-step study left at 40 iterations of 4,000 hands PPO 160k --
    60% more than the archive methods -- and every comparison drawn
    from it measures the discrepancy rather than the methods."""
    baseline = get_baseline("ppo")(
        BaselineConfig(max_iterations=40, train_batch_size=4000))
    baseline.apply_step_budget(100_000)
    assert baseline.config.max_iterations == 25


def test_it_rounds_down_rather_than_over():
    """Overrunning a budget is worse than underrunning it."""
    baseline = get_baseline("ppo")(
        BaselineConfig(max_iterations=100, train_batch_size=4000))
    baseline.apply_step_budget(99_000)
    assert baseline.config.max_iterations == 24
    assert baseline.config.max_iterations * 4000 <= 99_000


def test_a_smaller_iteration_cap_is_never_raised():
    """The budget is a ceiling, not a target."""
    baseline = get_baseline("ppo")(
        BaselineConfig(max_iterations=5, train_batch_size=4000))
    baseline.apply_step_budget(1_000_000)
    assert baseline.config.max_iterations == 5


def test_methods_counted_in_episodes_need_no_cap():
    for name in ("random", "go-explore-phase1"):
        baseline = get_baseline(name)(BaselineConfig(max_iterations=40))
        baseline.apply_step_budget(100_000)
        assert baseline.config.max_iterations == 40
        assert baseline.steps_per_iteration() is None


def test_the_budget_derives_the_episode_count_from_the_horizon():
    baseline = get_baseline("random")(BaselineConfig())
    assert baseline.apply_step_budget(1_000_000, 180) == 5555
    assert baseline.apply_step_budget(1_000_000, 6760) == 147
    assert baseline.episodes_in(100, 6760) == 1        # never zero
    assert baseline.episodes_in(1_000, 0) == 1_000     # no zero-division


# -- the evaluation horizon -------------------------------------------

def test_a_pinned_training_horizon_is_not_carried_into_evaluation():
    """EpicChase pins 180 so one episode reaches one chamber, which is
    what forces an archive. Carrying that into evaluation makes the
    goal unreachable by construction -- 937 actions of route against
    180 of budget -- so every policy scores zero and the metric
    distinguishes nothing."""
    from topogym.baselines.gridworld2dv1.single_layout import eval_horizon

    row = layout_row("TopoGym/EpicChase8-120-v0", 0)
    assert int(row["horizon"]) == 180
    assert eval_horizon(row) == 2820          # 3 x 937, to the ten
    assert eval_horizon(row) > int(row["optimal_actions"])


def test_an_unpinned_family_keeps_its_own_horizon():
    """The rule must not inflate worlds whose horizon already covers
    their route."""
    from topogym.baselines.gridworld2dv1.single_layout import eval_horizon

    for env_id in ("TopoGym/Decoys1-50-v0", "TopoGym/Maze-100-v0",
                   "TopoGym/ClownChase-v0"):
        row = layout_row(env_id, 0)
        assert eval_horizon(row) == int(row["horizon"]), env_id


# -- the single-episode ceiling ---------------------------------------

def test_the_ceiling_bounds_what_no_archive_can_exceed():
    """A method that never takes an archive reset restarts at the
    layout's start every episode, so this bounds its coverage however
    many steps it is given. Exceeding it is proof the archive carried
    the agent out of the region one episode covers."""
    from topogym.baselines.gridworld2dv1.single_layout import (
        single_episode_ceiling,
    )

    spiral = single_episode_ceiling("TopoGym/EpicChase8-120-v0", 0)
    assert 0.05 < spiral < 0.20        # ~10.4% of a 5,468-cell world
    # A world one episode can cover entirely has no meaningful ceiling.
    assert single_episode_ceiling("TopoGym/Decoys1-50-v0", 0) > 0.95


# -- seeds are separate studies ---------------------------------------

def test_each_layout_seed_gets_its_own_artefact_name():
    """A seed sweep writing every study to one filename leaves one
    result. Seed 0 keeps the bare name so existing paths do not move."""
    assert layout_row("TopoGym/EpicChase8-120-v0", 0)["unit"] \
        == "EpicChase8-120"
    assert layout_row("TopoGym/EpicChase8-120-v0", 7)["unit"] \
        == "EpicChase8-120@7"


# -- evaluation measures the policy, not the archive -------------------

def test_evaluation_takes_no_archive_reset_by_default(tmp_path):
    """The archive is a training artefact. Evaluating with it still
    available measures where the archive can drop you; without it, the
    thing training was supposed to produce."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    row = layout_row("TopoGym/Decoys1-50-v0", 0)
    baseline = get_baseline("go-explore-phase1")(BaselineConfig(seed=0))
    result = run_single_layout(baseline, row, step_budget=2000,
                               eval_episodes=5,
                               telemetry_root=str(tmp_path))
    episodes = pd.read_parquet(tmp_path / "episodes")
    evaluation = episodes[episodes["split"] == "single-eval"]
    assert not evaluation["archive_reset"].any()
    assert result.config["eval_archive"] is False


def test_evaluation_can_be_asked_to_keep_the_archive(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    row = layout_row("TopoGym/Decoys1-50-v0", 0)
    baseline = get_baseline("go-explore-phase1")(BaselineConfig(seed=0))
    run_single_layout(baseline, row, step_budget=2000, eval_episodes=6,
                      telemetry_root=str(tmp_path), eval_archive=True)
    episodes = (pd.read_parquet(tmp_path / "episodes")
                .query("split == 'single-eval'").sort_values("episode"))
    assert not episodes["archive_reset"].iloc[0]   # none to draw from
    assert episodes["archive_reset"].iloc[1:].all()


def test_training_is_recorded_as_its_own_phase(tmp_path):
    """Almost all the exploring happens during training on a
    single-layout study; recording only evaluation leaves the coverage
    curve invisible exactly where it was earned."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    row = layout_row("TopoGym/Decoys1-50-v0", 0)
    baseline = get_baseline("go-explore-phase1")(BaselineConfig(seed=0))
    run_single_layout(baseline, row, step_budget=4000, eval_episodes=3,
                      telemetry_root=str(tmp_path))
    episodes = pd.read_parquet(tmp_path / "episodes")
    assert set(episodes["split"]) == {"single-train", "single-eval"}
    training = episodes[episodes["split"] == "single-train"]
    assert len(training) > 3
    # Evaluation runs in a fresh world, so its coverage describes the
    # policy rather than inheriting what training uncovered.
    assert training["lifetime_coverage"].max() > \
        episodes[episodes["split"] == "single-eval"]["lifetime_coverage"].max()


def test_go_explore_phase_one_actually_trains_under_a_budget():
    """Its archive *is* what it learns. Without spending the training
    budget it arrives at evaluation empty and is measured on the last
    few percent of its budget."""
    row = layout_row("TopoGym/Decoys1-50-v0", 0)
    baseline = get_baseline("go-explore-phase1")(BaselineConfig(seed=0))
    result = run_single_layout(baseline, row, step_budget=6000,
                               eval_episodes=4)
    assert result.training["iterations"] > 0
    assert "explored" in result.training["stopped_because"]
    assert baseline.restorable_cells()


def test_without_a_budget_phase_one_still_trains_nothing():
    """The benchmark path is unchanged: each hold-out world is new, so
    its archive can only be built while that world is evaluated."""
    from topogym.baselines.gridworld2dv1.protocol import Hyperparameters

    rows = [layout_row("TopoGym/Decoys1-50-v0", 0)]
    baseline = get_baseline("go-explore-phase1")(BaselineConfig(seed=0))
    report = baseline.fit(rows, rows, Hyperparameters(values={}))
    assert report.iterations == 0
    assert "nothing is trained" in report.stopped_because
