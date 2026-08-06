"""The baseline protocol: shared contracts, no algorithm required."""

import json
import pathlib

import numpy as np
import pytest

from topogym.baselines import (
    BASELINES,
    SPLIT_USAGE,
    Baseline,
    BaselineConfig,
    BaselineResult,
    Hyperparameters,
    TrainingReport,
    get_baseline,
)
from topogym.baselines.instances import (
    FlatObservation,
    load_split,
    make_instance,
)
from topogym.baselines.protocol import Baseline as BaseClass
from topogym.baselines.random_walk import RandomBaseline
from topogym.baselines.report import (
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
        "import sys; from topogym.baselines import BASELINES, "
        "get_baseline; get_baseline('random'); "
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
    from topogym.baselines.evaluate import evaluate_instance

    row = load_split("test")[0]
    record = evaluate_instance(row, RandomBaseline().policy(), episodes=1)
    assert record["episodes"] == 1
    assert 0.0 <= record["success_rate"] <= 1.0
    for key in ("state_coverage", "visitation_entropy", "mean_regret",
                "planning_efficiency", "steps_to_coverage",
                "steps_to_h1_holes", "unique_states"):
        assert key in record["metrics"], key
    assert set(record["curves"]) == {
        "unique_states", "chambers_entered", "curvature_reached"}


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
        {"curves": {"unique_states": [[5, 10.0]], "chambers_entered": [],
                    "curvature_reached": []}},
        {"curves": {"unique_states": [[5, 20.0]], "chambers_entered": [],
                    "curvature_reached": []}},
    ]
    curves = mean_curves(instances)
    step, mean, error, count = curves["unique_states"][0]
    assert (step, mean, count) == (5, 15.0, 2)
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
