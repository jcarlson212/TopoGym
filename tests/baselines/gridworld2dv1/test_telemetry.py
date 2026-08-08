"""Parquet telemetry: the long-form tables written alongside results."""

from __future__ import annotations

import pytest

from topogym.baselines.gridworld2dv1 import telemetry

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


def _rows(n, **extra):
    return [{"episode": 0, "step": i, "interaction": i, "action": i % 3,
             "x": i, "y": 0, "reward": 0.0, **extra} for i in range(n)]


def test_tables_partition_by_algorithm_and_split(tmp_path):
    """Hive-style layout, so a query for one split reads one directory
    and every reader understands it without being told."""
    with telemetry.open_writer(str(tmp_path), "ppo") as writer:
        writer.add_steps(_rows(3), split="train", instance="A@1",
                         family="Maze", size=50, seed=1)
        writer.add_steps(_rows(2), split="test", instance="A@2",
                         family="Maze", size=50, seed=2)

    frame = pd.read_parquet(tmp_path / "steps")
    assert len(frame) == 5
    assert set(frame["split"]) == {"train", "test"}
    assert set(frame["algorithm"]) == {"ppo"}
    assert (tmp_path / "steps/algorithm=ppo/split=train").is_dir()


def test_partition_columns_are_not_duplicated_in_the_file(tmp_path):
    """Writing a partition key into both path and body makes readers
    infer two different types for it and refuse to merge them."""
    import pyarrow.parquet as pq

    with telemetry.open_writer(str(tmp_path), "ppo") as writer:
        writer.add_steps(_rows(2), split="test", instance="A@1",
                         family="Maze", size=50, seed=1)
    part = next((tmp_path / "steps").rglob("*.parquet"))
    names = pq.read_table(part).column_names
    assert "algorithm" not in names and "split" not in names
    assert "instance" in names  # non-partition keys stay in the body


def test_concurrent_writers_do_not_overwrite_each_other(tmp_path):
    """Every instance is evaluated in its own process with its own
    writer, so part numbering alone would collide."""
    for key in ("A@1", "A@2", "A@3"):
        with telemetry.open_writer(str(tmp_path), "ppo",
                                   part_prefix=f"{key}-") as writer:
            writer.add_steps(_rows(4), split="test", instance=key,
                             family="Maze", size=50, seed=1)
    frame = pd.read_parquet(tmp_path / "steps")
    assert len(frame) == 12
    assert frame["instance"].nunique() == 3


def test_batching_flushes_without_losing_the_tail(tmp_path):
    with telemetry.open_writer(str(tmp_path), "ppo",
                               batch_size=10) as writer:
        writer.add_steps(_rows(25), split="test", instance="A@1",
                         family="Maze", size=50, seed=1)
    assert len(pd.read_parquet(tmp_path / "steps")) == 25


def test_instance_rows_flatten_the_native_metric_set(tmp_path):
    """The whole point of the instances table: every metric TopoGym
    tracks becomes a column, including the ones no figure plots."""
    record = {
        "instance": "A@1", "lifetime_coverage": 0.5,
        "steps_to_goal": [None, 4],           # per-episode: dropped here
        "curves": {"coverage": [[0, 0.0]]},   # summary: dropped here
        "metrics": {"visitation_entropy": 8.3, "unique_states": 12,
                    "steps_to_coverage": {0.1: 5}},  # list-ish: dropped
    }
    with telemetry.open_writer(str(tmp_path), "ppo") as writer:
        writer.add_instance(record, split="test", instance="A@1",
                            family="Maze", size=50, seed=1)
    frame = pd.read_parquet(tmp_path / "instances")
    assert frame["metric_visitation_entropy"].iloc[0] == 8.3
    assert frame["metric_unique_states"].iloc[0] == 12
    assert "steps_to_goal" not in frame.columns
    assert "metric_steps_to_coverage" not in frame.columns


def test_telemetry_off_is_a_no_op(tmp_path):
    """A run without telemetry must not create anything, and must not
    make callers branch on whether the writer exists."""
    with telemetry.open_writer(None, "ppo") as writer:
        writer.add_steps(_rows(5), split="test")
        writer.add_instance({"a": 1}, split="test")
    assert not list(tmp_path.iterdir())


def test_a_missing_pyarrow_disables_telemetry_rather_than_failing(
        tmp_path, monkeypatch):
    """Losing the analysis file is bad; losing the ten-hour run that
    produced it is worse."""
    monkeypatch.setattr(telemetry, "is_available", lambda: False)
    writer = telemetry.open_writer(str(tmp_path), "ppo")
    writer.add_steps(_rows(3), split="test")
    writer.close()
    assert not list(tmp_path.iterdir())


def test_a_gcs_uri_is_accepted_without_touching_the_local_disk():
    """GKE pods do not keep their disks, so ``gs://`` has to be a
    first-class destination rather than a post-run copy."""
    writer = telemetry.open_writer("gs://topogym-runs/sweep-1", "ppo")
    assert isinstance(writer, telemetry.TelemetryWriter)
    assert writer._base.startswith("topogym-runs")  # bucket-relative


# -- integration with the evaluation loop -----------------------------

def test_an_evaluation_emits_all_three_tables(tmp_path):
    from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
        RandomPolicyFactory,
    )
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_split
    from topogym.baselines.gridworld2dv1.instances import load_split

    rows = load_split("val")[:2]
    records = evaluate_split(
        rows, None, episodes=2, seed=0,
        policy_factory=RandomPolicyFactory(0), workers=1,
        telemetry_root=str(tmp_path), algorithm="random", split="val",
    )
    steps = pd.read_parquet(tmp_path / "steps")
    episodes = pd.read_parquet(tmp_path / "episodes")
    instances = pd.read_parquet(tmp_path / "instances")

    assert len(instances) == len(records) == 2
    assert len(episodes) == 4  # two instances x two episodes
    # Every step of every episode, and the totals agree with the record.
    assert len(steps) == sum(r["interactions"] for r in records)
    assert episodes["length"].sum() == len(steps)
    # The expensive columns are present and per-episode, not per-step.
    assert episodes["observed_h1"].notna().all()
    assert "observed_h1" not in steps.columns


def test_step_stride_thins_the_largest_table_only(tmp_path):
    from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
        RandomPolicyFactory,
    )
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_split
    from topogym.baselines.gridworld2dv1.instances import load_split

    rows = load_split("val")[:1]
    records = evaluate_split(
        rows, None, episodes=2, seed=0,
        policy_factory=RandomPolicyFactory(0), workers=1,
        telemetry_root=str(tmp_path), algorithm="random", split="val",
        step_stride=10,
    )
    steps = pd.read_parquet(tmp_path / "steps")
    episodes = pd.read_parquet(tmp_path / "episodes")
    assert len(steps) == pytest.approx(records[0]["interactions"] / 10,
                                       abs=2)
    assert len(episodes) == 2  # unthinned


def test_a_column_null_in_one_part_merges_with_a_typed_one(tmp_path):
    """Inference is per file, and a column can be legitimately all-null
    in one part and populated in another -- reset_cell is null for every
    row of an evaluation taking no archive resets, and a string during
    training. Arrow then infers null for one and string for the other
    and refuses to merge, so the whole dataset becomes unreadable
    because of a column nobody was querying."""
    with telemetry.open_writer(str(tmp_path), "ge") as writer:
        writer.add_episodes(
            [{"episode": 0, "length": 5, "archive_reset": True,
              "reset_cell": "(3, 4)"}],
            split="single-train", instance="A@0", family="X",
            size=50, seed=0)
        writer.add_episodes(
            [{"episode": 0, "length": 5, "archive_reset": False,
              "reset_cell": None}],
            split="single-eval", instance="A@0", family="X",
            size=50, seed=0)
    frame = pd.read_parquet(tmp_path / "episodes")   # must not raise
    assert len(frame) == 2
    assert set(frame["split"]) == {"single-train", "single-eval"}


def test_fractions_stay_fractions_when_tracing_is_off(tmp_path):
    """The denominators are not a tracing feature. Leaving them at 1
    does not disable the fraction fields, it turns them into raw counts
    -- and a coverage column reading 111.0 will be plotted against one
    reading 0.11 without complaint."""
    from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
        RandomPolicyFactory,
    )
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_split
    from topogym.baselines.gridworld2dv1.instances import load_split

    rows = load_split("val")[:1]
    for trace in (True, False):
        folder = tmp_path / f"trace-{trace}"
        evaluate_split(rows, None, episodes=2, seed=0, trace=trace,
                       policy_factory=RandomPolicyFactory(0), workers=1,
                       telemetry_root=str(folder), algorithm="random",
                       split="val")
        episodes = pd.read_parquet(folder / "episodes")
        for column in ("episode_coverage", "lifetime_coverage",
                       "observed_frac"):
            values = episodes[column].dropna()
            assert (values <= 1.0).all(), (column, trace, list(values))


def test_chambers_entered_is_lifetime_like_coverage(tmp_path):
    """The environment clears its per-episode chamber record on every
    reset, so counting only what one evaluation loop saw reports "1
    chamber" beside a coverage figure including the six found during
    training."""
    from topogym.baselines.gridworld2dv1.concrete_baselines.random_walk import (
        RandomPolicyFactory,
    )
    from topogym.baselines.gridworld2dv1.evaluate import evaluate_instance
    from topogym.baselines.gridworld2dv1.instances import make_instance
    from topogym.baselines.gridworld2dv1.single_layout import layout_row
    from topogym.stats import StatsRecorder

    row = layout_row("TopoGym/Decoys0-50-v0", 0)
    env = StatsRecorder(make_instance(row, teleport=True))
    core = env.unwrapped
    policy = RandomPolicyFactory(0)()

    evaluate_instance(row, policy, episodes=6, seed=0, trace=False,
                      env=env)
    first = {i for c, i in core._chamber_of.items()
             if c in core.lifetime_visit_counts}
    record = evaluate_instance(row, policy, episodes=2, seed=99,
                               trace=False, env=env)
    env.close()
    assert record["chambers_entered"] >= len(first)
    assert record["chambers_entered"] == len(
        {i for c, i in core._chamber_of.items()
         if c in core.lifetime_visit_counts})
