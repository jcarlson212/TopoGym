"""Go-Explore: cell selection per Appendix A.5 of the paper."""

import pytest

from topogym.baselines.gridworld2dv1 import BaselineConfig, get_baseline
from topogym.baselines.gridworld2dv1.concrete_baselines.goexplore_phase1 import (
    ATTRIBUTES,
    DEFAULTS,
    Archive,
    GoExplorePhase1Baseline,
    GoExploreReset,
    GoExploreResetFactory,
)
from topogym.baselines.gridworld2dv1.evaluate import evaluate_instance
from topogym.baselines.gridworld2dv1.instances import load_split


def _neighbors_of(cell):
    x, y = cell
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def test_cell_score_follows_equations_1_2_and_4():
    """CellScore = sum_n NeighScore + sum_a CntScore + 1, with
    LevelWeight 1 (no levels here)."""
    params = {**DEFAULTS, "w_a": 2.0, "p_a": 1.0, "w_n": 5.0,
              "eps1": 0.001, "eps2": 0.00001}
    archive = Archive(params, neighbors=_neighbors_of)
    archive.observe({(0, 0)})               # seen once, chosen zero
    archive.cells[(0, 0)]["chosen"] = 3

    entry = archive.cells[(0, 0)]
    expected = 0.0
    for attribute in ATTRIBUTES:
        expected += 2.0 * (1.0 / (entry[attribute] + 0.001)) ** 1.0
        expected += 0.00001
    expected += 4 * 5.0   # all four neighbours absent from the archive
    expected += 1.0       # equation 4's constant
    assert archive.score((0, 0)) == pytest.approx(expected)
    assert archive.score((0, 0)) > 0  # always positive


def test_a_cell_with_missing_neighbours_scores_higher():
    """The frontier is what the neighbour subscore is for."""
    archive = Archive({**DEFAULTS, "w_n": 10.0}, neighbors=_neighbors_of)
    archive.observe({(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (5, 5)})
    enclosed = archive.score((0, 0))   # all present
    frontier = archive.score((5, 5))   # none present
    assert frontier > enclosed


def test_counts_lower_the_score():
    archive = Archive({**DEFAULTS, "w_n": 0.0}, neighbors=_neighbors_of)
    archive.observe({(0, 0), (9, 9)})
    for _ in range(10):
        archive.cells[(9, 9)]["chosen"] += 1
    assert archive.score((0, 0)) > \
        archive.score((9, 9))


def test_selection_is_proportional_and_updates_counts():
    archive = Archive(DEFAULTS, seed=0, neighbors=_neighbors_of)
    archive.observe({(0, 0), (1, 1)})
    chosen = archive.select()
    assert chosen in archive.cells
    assert archive.cells[chosen]["chosen"] == 1
    assert archive.cells[chosen]["chosen_since_new"] == 1
    assert archive.select() is not None
    assert Archive(DEFAULTS).select() is None  # empty archive


def test_discovering_something_new_resets_chosen_since_new():
    archive = Archive(DEFAULTS, seed=0, neighbors=_neighbors_of)
    archive.observe({(0, 0)})
    archive.cells[(0, 0)]["chosen_since_new"] = 7
    archive.observe({(0, 0), (1, 0)}, chosen_from=(0, 0))  # a new cell
    assert archive.cells[(0, 0)]["chosen_since_new"] == 0
    archive.cells[(0, 0)]["chosen_since_new"] = 4
    archive.observe({(0, 0)}, chosen_from=(0, 0))          # nothing new
    assert archive.cells[(0, 0)]["chosen_since_new"] == 4


def test_archive_resets_when_the_world_does():
    """An archive of another world's cells is meaningless."""
    hook = GoExploreReset(DEFAULTS, seed=0)
    rows = load_split("test")
    from topogym.baselines.gridworld2dv1.instances import make_instance

    first = make_instance(rows[0]).unwrapped
    first.reset(seed=0)
    for _ in range(10):
        first.step(2)
    hook(first, {})
    assert hook.archive.cells

    second = make_instance(rows[5]).unwrapped
    second.reset(seed=0)
    hook(second, {})
    assert hook._layout is second.layout
    assert set(hook.archive.cells) <= set(second.layout.free_cells)


def test_go_explore_uses_the_boundary_probe_and_explores_further():
    """The archive is the whole point: same random policy, more of the
    world seen, because episodes resume from chosen cells."""
    row = next(r for r in load_split("test") if r["unit"] == "Decoys4-50")
    baseline = GoExplorePhase1Baseline(BaselineConfig(seed=0))
    with_archive = evaluate_instance(
        row, baseline.policy(), episodes=15, trace=False,
        choose_reset=baseline.choose_reset,
        env_options=baseline.env_options(),
    )
    without = evaluate_instance(row, baseline.policy(), episodes=15,
                                trace=False,
                                env_options=baseline.env_options())
    assert with_archive["archive_resets"] == 14  # every boundary but the first
    assert without["archive_resets"] == 0
    assert with_archive["lifetime_coverage"] > without["lifetime_coverage"]


def test_pools_every_non_holdout_split_and_never_the_holdout():
    baseline = GoExplorePhase1Baseline()
    assert set(baseline.tuning_splits) == {"tune", "train", "val"}
    assert "test" not in baseline.tuning_splits


def test_reset_factory_is_picklable_for_parallel_sweeps():
    import pickle

    factory = GoExploreResetFactory(DEFAULTS, seed=3)
    restored = pickle.loads(pickle.dumps(factory))
    hook = restored(7)
    assert isinstance(hook, GoExploreReset)
    assert hook.seed == 7


def test_registered_under_its_name():
    assert get_baseline("go-explore-phase1") is GoExplorePhase1Baseline
    assert GoExplorePhase1Baseline.name == "go-explore-phase1"


def test_level_weight_is_one():
    """There are no levels here, so equation 3 is identically 1 and
    the score is the bracket of equation 4."""
    archive = Archive({**DEFAULTS, "w_n": 0.0, "w_a": 0.0, "eps2": 0.0},
                      neighbors=_neighbors_of)
    archive.observe({(0, 0)})
    assert archive.score((0, 0)) == pytest.approx(1.0)


def test_return_is_reported_per_episode_as_well_as_total():
    row = load_split("test")[0]
    record = evaluate_instance(row, GoExplorePhase1Baseline().policy(),
                               episodes=4, trace=False)
    assert record["return_per_episode"] == pytest.approx(
        record["cumulative_return"] / 4)


def test_successive_halving_thins_candidates_across_rungs():
    """Each rung scores only the survivors of the last, so the wide
    grid costs 28 sweeps rather than 48."""
    from topogym.baselines.gridworld2dv1.instances import load_split

    baseline = GoExplorePhase1Baseline(
        BaselineConfig(seed=0, tune_episodes=1, eval_workers=1))
    baseline.selection_rungs = (("tune", 3), ("train", 2), ("val", 1))
    baseline.tune_grid = GoExplorePhase1Baseline.tune_grid[:4]
    tuning = {name: load_split(name)[:2]
              for name in ("tune", "train", "val")}

    chosen = baseline.select_hyperparameters(tuning)
    per_rung = {}
    for entry in chosen.searched:
        per_rung[entry["split"]] = per_rung.get(entry["split"], 0) + 1
    assert per_rung == {"tune": 4, "train": 3, "val": 2}
    assert set(chosen.values) >= {"w_a", "p_a", "w_n", "eps1", "eps2"}
    # The hold-out is never a rung.
    assert "test" not in per_rung


def test_full_grid_and_rungs_are_declared():
    assert len(GoExplorePhase1Baseline.tune_grid) == 16
    assert GoExplorePhase1Baseline.selection_rungs == (
        ("tune", 8), ("train", 4), ("val", 1))
    sweeps, survivors = 0, len(GoExplorePhase1Baseline.tune_grid)
    for _split, keep in GoExplorePhase1Baseline.selection_rungs:
        sweeps += survivors
        survivors = min(survivors, keep)
    assert sweeps == 28 < 3 * len(GoExplorePhase1Baseline.tune_grid)
