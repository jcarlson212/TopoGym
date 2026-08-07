"""The baseline protocol: shared contracts, no algorithm required."""

import json
import pathlib

import numpy as np
import pytest

from topogym.baselines.gridworld2dv1 import (
    BASELINES,
    SPLIT_USAGE,
    Baseline,
    BaselineConfig,
    BaselineResult,
    Hyperparameters,
    TrainingReport,
    get_baseline,
)
from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import RandomBaseline
from topogym.baselines.gridworld2dv1.instances import (
    FlatObservation,
    load_split,
    make_instance,
)
from topogym.baselines.gridworld2dv1.protocol import Baseline as BaseClass
from topogym.baselines.gridworld2dv1.report import (
    _bootstrap_ci,
    _full_support,
    aggregate,
    mean_curves,
    write_benchmarks_md,
    write_result,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _splits(n=2):
    return {name: load_split(name)[:n]
            for name in ("tune", "train", "val", "test")}


# -- the shared contract --------------------------------------------

def test_registry_resolves_by_name():
    assert set(BASELINES) >= {"random", "ppo"}
    # Entries are import paths, not imported classes: listing the
    # baselines must not drag in a deep-learning stack.
    assert all(isinstance(v, str) and ":" in v
               for v in BASELINES.values())
    assert get_baseline("random") is RandomBaseline
    with pytest.raises(KeyError, match="unknown baseline"):
        get_baseline("nope")


def test_listing_baselines_never_imports_ray():
    """Checked in a fresh interpreter: in-process, another test may
    already have imported Ray."""
    import subprocess
    import sys

    code = (
        "import sys; "
        "from topogym.baselines.gridworld2dv1 import get_baseline; "
        "get_baseline('random'); "
        "print('ray' in sys.modules or 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout


def test_split_usage_is_declared_for_every_split():
    assert set(SPLIT_USAGE) == {"tune", "train", "val", "test"}
    assert "once" in SPLIT_USAGE["test"]


def test_baseline_requires_fit_and_policy():
    class Incomplete(BaseClass):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()


def test_subclass_inherits_the_protocol():
    """A variant overrides behaviour, not the split contract."""

    class Greedy(RandomBaseline):
        name = "greedy-zero"

        def policy(self):
            return lambda obs, env: 0

    baseline = Greedy(BaselineConfig(eval_episodes=1))
    assert isinstance(baseline, Baseline)
    result = baseline.run(_splits(1))
    assert result.algorithm == "greedy-zero"
    assert result.to_dict()["split_usage"] == SPLIT_USAGE


def test_dataclasses_round_trip():
    config = BaselineConfig(seed=3, run_dir=pathlib.Path("/tmp/x"))
    assert config.to_dict()["run_dir"] == "/tmp/x"
    assert Hyperparameters({"lr": 1e-3}, 0.5).to_dict()["values"]["lr"]
    assert TrainingReport(iterations=4).to_dict()["iterations"] == 4
    payload = BaselineResult("x").to_dict()
    assert set(payload) >= {"algorithm", "split_usage", "instances"}


# -- instances -------------------------------------------------------

def test_flat_observation_matches_what_rllib_accepts():
    row = load_split("test")[0]
    env = make_instance(row)
    assert isinstance(env, FlatObservation)
    assert len(env.observation_space.shape) == 1  # not 2D
    obs, _ = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    assert 0.0 <= obs.min() and obs.max() <= 1.0
    env.close()


def test_instances_are_reproducible_from_their_rows():
    row = load_split("test")[0]
    a, b = make_instance(row), make_instance(row)
    a.reset(seed=0)
    b.reset(seed=0)
    assert a.unwrapped.layout.cell_types == b.unwrapped.layout.cell_types
    a.close()
    b.close()


# -- evaluation ------------------------------------------------------

def test_evaluation_records_the_full_native_metric_set():
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_instance

    row = load_split("test")[0]
    record = evaluate_instance(row, RandomBaseline().policy(), episodes=1)
    assert record["episodes"] == 1
    assert 0.0 <= record["success_rate"] <= 1.0
    for key in ("state_coverage", "visitation_entropy", "mean_regret",
                "planning_efficiency", "steps_to_coverage",
                "steps_to_h1_holes", "unique_states"):
        assert key in record["metrics"], key
    assert set(record["curves"]) == {
        "coverage", "chambers_entered", "curvature_reached",
        "cumulative_return"}


# -- aggregation and reporting ---------------------------------------

def test_bootstrap_ci_brackets_the_point_estimate():
    values = np.arange(1.0, 21.0)
    point, (low, high) = _bootstrap_ci(values, np.median, 400)
    assert point == pytest.approx(10.5)
    assert low <= point <= high
    assert _bootstrap_ci(np.array([]), np.median, 10)[0] is None


def test_full_support_drops_the_survivorship_tail():
    curve = [[5, 1.0, 0.1, 4], [10, 2.0, 0.1, 4], [15, 9.0, 0.1, 1]]
    assert _full_support(curve) == curve[:2]
    assert _full_support([]) == []


def test_aggregate_reports_intervals_and_per_slice():
    instances = [
        {"median_steps_to_goal": 10.0, "success_rate": 1.0,
         "lifetime_coverage": 0.5, "optimal_actions": 5, "slice": "A"},
        {"median_steps_to_goal": 20.0, "success_rate": 0.5,
         "lifetime_coverage": 0.3, "optimal_actions": 5, "slice": "B"},
        {"median_steps_to_goal": None, "success_rate": 0.0,
         "lifetime_coverage": 0.1, "optimal_actions": 5, "slice": "B"},
    ]
    totals = aggregate(instances)
    assert totals["instances_evaluated"] == 3
    assert totals["instances_solved"] == 2
    assert totals["median_steps_to_goal"] == pytest.approx(15.0)
    assert len(totals["median_steps_to_goal_ci"]) == 2
    assert totals["success_rate"] == pytest.approx(0.5)
    assert set(totals["per_slice"]) == {"A", "B"}
    assert totals["per_slice"]["B"]["instances"] == 2
    # Unsolved instances score zero efficiency rather than vanishing.
    assert 0.0 <= totals["efficiency"]["iqm"] <= 1.0


def test_mean_curves_average_by_step_with_standard_error():
    instances = [
        {"curves": {"coverage": [[5, 0.10]], "chambers_entered": [],
                    "curvature_reached": []}},
        {"curves": {"coverage": [[5, 0.20]], "chambers_entered": [],
                    "curvature_reached": []}},
    ]
    curves = mean_curves(instances)
    step, mean, error, count = curves["coverage"][0]
    assert (step, mean, count) == (5, pytest.approx(0.15), 2)
    assert error > 0


def test_published_artifacts_are_written(tmp_path):
    result = BaselineResult("demo")
    result.instances = [{
        "median_steps_to_goal": 12.0, "success_rate": 1.0,
        "lifetime_coverage": 0.4, "optimal_actions": 6, "slice": "A",
    }]
    result.aggregates = aggregate(result.instances)
    path = write_result(result, tmp_path)
    payload = json.loads(path.read_text())
    assert payload["algorithm"] == "demo"
    assert payload["split_usage"]["test"].endswith("has stopped")

    document = write_benchmarks_md({"demo": payload},
                                   tmp_path / "BENCHMARKS.md")
    text = document.read_text()
    assert "TopoGym-v1 benchmark results" in text
    assert "`demo`" in text and "12.0" in text


# -- grouping --------------------------------------------------------

def test_groupings_partition_without_loss():
    from topogym.baselines.gridworld2dv1.protocol import GROUPINGS, group_rows

    rows = load_split("train")
    sizes = {}
    for grouping in GROUPINGS:
        groups = group_rows(rows, grouping)
        assert sum(len(v) for v in groups.values()) == len(rows)
        # No row lands in two groups.
        keys = [id(r) for group in groups.values() for r in group]
        assert len(keys) == len(set(keys))
        sizes[grouping] = len(groups)
    # Coarse to fine: one general explorer, then slices, then families,
    # then individual worlds.
    assert sizes["all"] == 1
    assert sizes["all"] < sizes["slice"] < sizes["family"] <= sizes["unit"]
    with pytest.raises(ValueError, match="unknown grouping"):
        group_rows(rows, "nope")


def test_hardware_knobs_reach_the_config():
    config = BaselineConfig(num_env_runners=16, num_envs_per_runner=4,
                            num_learners=1, gpus_per_learner=0)
    payload = config.to_dict()
    assert payload["num_env_runners"] == 16
    assert payload["num_envs_per_runner"] == 4
    assert payload["num_learners"] == 1
    assert payload["gpus_per_learner"] == 0


def test_aggregate_breaks_out_per_group():
    instances = [
        {"median_steps_to_goal": 10.0, "success_rate": 1.0,
         "lifetime_coverage": 0.5, "optimal_actions": 5, "slice": "A",
         "group": "Decoys"},
        {"median_steps_to_goal": 30.0, "success_rate": 0.5,
         "lifetime_coverage": 0.2, "optimal_actions": 5, "slice": "A",
         "group": "Maze"},
    ]
    totals = aggregate(instances)
    assert set(totals["per_group"]) == {"Decoys", "Maze"}
    assert totals["per_group"]["Maze"]["instances"] == 1


# -- archive methods -------------------------------------------------

def test_evaluation_offers_every_method_the_boundary_probe():
    """An archive method must be able to choose where each episode
    resumes; the harness must not decide that for it."""
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_instance

    row = load_split("test")[0]
    chosen = []

    def choose_reset(env, info):
        # Return somewhere already stood on: the archive's whole point.
        visited = sorted(env._ever_visited | env._visited)
        if not visited:
            return None
        target = visited[len(visited) // 2]
        chosen.append(target)
        return target

    record = evaluate_instance(row, RandomBaseline().policy(),
                               episodes=4, trace=False,
                               choose_reset=choose_reset)
    assert chosen, "the probe was never offered"
    assert record["archive_resets"] == len(chosen)
    assert record["episodes"] == 4


def test_archive_resets_need_no_hook_by_default():
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_instance

    record = evaluate_instance(load_split("test")[0],
                               RandomBaseline().policy(),
                               episodes=2, trace=False)
    assert record["archive_resets"] == 0  # inert unless asked for


def test_instances_enable_archive_resets():
    env = make_instance(load_split("test")[0])
    assert env.unwrapped.teleport is True


def test_contiguous_exposure_keeps_one_world_for_a_run():
    """Archive methods need a run of episodes on the same world, and
    the instance must survive the boundary so its archive does."""
    from topogym.baselines.gridworld2dv1.multitask import SplitEnv

    rows = load_split("train")[::40][:3]
    assert len({r["unit"] for r in rows}) == 3  # genuinely distinct
    env = SplitEnv({"rows": rows, "seed": 0, "sequential": True,
                    "episodes_per_instance": 3})
    seen = []
    for _ in range(9):
        env.reset(seed=0)
        seen.append(env.row["unit"])
    assert seen == [rows[0]["unit"]] * 3 + [rows[1]["unit"]] * 3 + \
        [rows[2]["unit"]] * 3
    env.close()

    # The default stays i.i.d. for gradient methods.
    plain = SplitEnv({"rows": rows, "seed": 0, "sequential": True})
    units = []
    for _ in range(3):
        plain.reset(seed=0)
        units.append(plain.row["unit"])
    assert units == [r["unit"] for r in rows]
    plain.close()


def test_lifetime_state_survives_a_contiguous_boundary():
    from topogym.baselines.gridworld2dv1.multitask import SplitEnv

    rows = load_split("train")[:1]
    env = SplitEnv({"rows": rows, "seed": 0, "episodes_per_instance": 3})
    env.reset(seed=0)
    for _ in range(20):
        env.step(2)
    core = env.env.unwrapped
    before = len(core._ever_visited | core._visited)
    env.reset(seed=1)  # same world: the archive must not be discarded
    core = env.env.unwrapped
    assert len(core._ever_visited | core._visited) >= before
    env.close()


def test_baselines_are_scoped_to_a_benchmark_version():
    """Each benchmark version owns its baselines, so a later version
    can change protocol or splits without disturbing published runs."""
    import pytest as _pytest

    from topogym.baselines import (
        BENCHMARK_PACKAGES,
        DEFAULT_BENCHMARK,
        baselines_for,
    )

    assert DEFAULT_BENCHMARK in BENCHMARK_PACKAGES
    package = baselines_for(DEFAULT_BENCHMARK)
    assert set(package.BASELINES) >= {"random", "ppo"}
    assert all(entry.startswith("topogym.baselines.gridworld2dv1")
               for entry in package.BASELINES.values())
    with _pytest.raises(KeyError, match="no baselines for benchmark"):
        baselines_for("nope-v9")


def test_published_artifacts_are_filed_under_the_benchmark_version():
    """Results from different benchmark versions must not share a
    directory, however similar their filenames."""
    root = pathlib.Path(__file__).resolve().parents[3]
    published = root / "benchmarks" / "gridworld2dv1"
    assert (published / "results").is_dir()
    assert (published / "plots").is_dir()
    # Nothing may sit loose at the top of benchmarks/ but the README.
    stray = [p.name for p in (root / "benchmarks").iterdir()
             if p.is_file() and p.name != "README.md"]
    assert not stray, f"unversioned artefacts: {stray}"


def test_curves_are_fractions_so_sizes_are_comparable():
    """Hold-out worlds differ in size by two orders of magnitude, so a
    mean of raw counts would mostly report which worlds are large."""
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_instance
    from topogym.baselines.gridworld2dv1.report import FIGURES

    rows = load_split("test")
    small = min(rows, key=lambda r: int(r["n_free_cells"]))
    large = max(rows, key=lambda r: int(r["n_free_cells"]))
    assert int(large["n_free_cells"]) > 50 * int(small["n_free_cells"])

    for row in (small, large):
        record = evaluate_instance(row, RandomBaseline().policy(),
                                   episodes=1)
        for key, _label, _title in FIGURES:
            values = [v for _step, v in record["curves"][key]]
            assert values, key
            if key == "cumulative_return":
                continue  # reward is not size-relative; see FIGURES
            assert all(0.0 <= v <= 1.0 for v in values), (key, row["unit"])


def test_baselines_declare_the_terms_they_are_measured_on():
    """A baseline states its action space, observation and reward
    modes rather than inheriting them silently, and the declaration is
    recorded with the results."""
    from topogym import ActionMode

    default = RandomBaseline(BaselineConfig())
    assert default.actions == ActionMode.EGOCENTRIC
    assert default.env_options() == {"actions": "egocentric"}

    class Fourway(RandomBaseline):
        name = "random-fourway"
        actions = ActionMode.FOURWAY
        obs_mode = "vector"
        reward_mode = "coverage"

    options = Fourway(BaselineConfig()).env_options()
    assert options == {"actions": "fourway", "obs_mode": "vector",
                       "reward_mode": "coverage"}

    result = Fourway(BaselineConfig(eval_episodes=1)).run(_splits(1))
    assert result.config["env_options"] == options  # recorded, not lost


def test_action_enums_are_usable_as_actions():
    import gymnasium as gym

    from topogym import ActionMode, EgocentricAction, FourwayAction

    assert ActionMode.EGOCENTRIC.actions is EgocentricAction
    assert ActionMode.FOURWAY.actions is FourwayAction
    env = gym.make("TopoGym/Dilution-50-v0").unwrapped
    env.reset(seed=0)
    env.step(EgocentricAction.FORWARD)  # an IntEnum *is* the action
    assert env._steps == 1
    assert env.action_space.contains(int(EgocentricAction.TURN_LEFT))


def test_observation_code_contract_is_slice_independent():
    """Hazards (8) and wormholes (9) appear only in Texture worlds, so
    a policy trained on GridWorld2D meets them first at evaluation. The
    declared space and the advertised code count must already account
    for that, or an embedding sized from training data aliases them."""
    import gymnasium as gym

    from topogym import OBS_CODE_COUNT, OBS_MAX
    from topogym.baselines.gridworld2dv1.protocol import Baseline

    assert OBS_CODE_COUNT == OBS_MAX + 1 == 10
    assert Baseline.observation_codes() == OBS_CODE_COUNT

    for env_id in ("TopoGym/Dilution-50-v0", "TopoGym/SpaceWarp-v0",
                   "TopoGym/TopKlein-50-v0"):
        env = gym.make(env_id).unwrapped
        env.reset(seed=0)
        # Same declared bound on every slice, whatever codes occur.
        assert env.observation_space.high.max() == OBS_MAX
        env.close()

    # And the baseline wrapper normalises by the largest code, not the
    # largest seen, so Texture codes stay inside [0, 1].
    wrapped = make_instance(load_split("test")[0])
    assert wrapped.observation_codes == OBS_CODE_COUNT
    assert wrapped.observation_space.high.max() == pytest.approx(1.0)
